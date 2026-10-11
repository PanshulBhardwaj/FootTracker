"""Persistent-identity multi-object tracker.

 * Constant-velocity (alpha-beta style) filter on the ground-contact point; dt-aware, so
   frame skipping widens the association gate instead of breaking it.
 * Cascade: (1) active tracks vs high-conf detections (Hungarian), (2) leftover active vs
   low-conf detections, (3) LOST tracks vs leftover detections using appearance + an
   ambiguity guard, (4) new tentative tracks.
 * player_id is minted only when a tentative track is confirmed and is never reused.
 * Ambiguous re-identification is not forced; the detection stays tentative.
 * Predicted positions are exposed separately and never flagged as confirmed.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from .config import TrackerConfig

BIG = 1e6

@dataclass
class Detection:
    bbox: tuple
    conf: float
    feat: Optional[np.ndarray] = None
    team_id: Optional[str] = None
    team_conf: float = 0.0
    @property
    def ground(self):  # feet midpoint, not bbox centre
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, y2], float)
    @property
    def height(self):
        return max(1.0, self.bbox[3] - self.bbox[1])

@dataclass
class Track:
    tid: int
    pos: np.ndarray
    vel: np.ndarray
    height: float
    feat: Optional[np.ndarray]
    last_ms: int
    first_ms: int
    hits: int = 1
    player_id: Optional[str] = None
    team_id: Optional[str] = None
    team_conf: float = 0.0
    status: str = "visible"
    last_bbox: Optional[tuple] = None
    last_conf: float = 0.0
    identity_conf: float = 0.5
    history: list = field(default_factory=list)
    pred: Optional[np.ndarray] = None
    def predict(self, dt):
        return self.pos + self.vel * dt

def _sim(a, b):
    if a is None or b is None:
        return 0.5
    return float(np.dot(a, b))

class Tracker:
    def __init__(self, cfg: TrackerConfig = None):
        self.cfg = cfg or TrackerConfig()
        self.tracks = []
        self._next_tmp = 1
        self._next_player = 1
        self.audit = []
        self.canonical = {}
        self.last_ms = None

    def _log(self, ev, ts, **kw):
        self.audit.append({"t_ms": ts, "event": ev, **kw})

    def canon(self, pid):
        while pid in self.canonical:
            pid = self.canonical[pid]
        return pid

    def _cost(self, tr, det, dt, use_app=True):
        d = float(np.linalg.norm(tr.predict(dt) - det.ground))
        g = self.cfg.max_speed_px_s * dt + self.cfg.gate_height_factor * max(tr.height, det.height) + 15
        if d > g:
            return BIG
        c = d / g
        if use_app and tr.feat is not None and det.feat is not None:
            w = self.cfg.appearance_weight
            c = (1 - w) * c + w * (1 - _sim(tr.feat, det.feat))
        return c

    def _assign(self, trs, dets, ts, use_app=True):
        if not trs or not dets:
            return [], list(range(len(trs))), list(range(len(dets)))
        M = np.array([[self._cost(t, d, max(0.0, (ts - t.last_ms) / 1000.0), use_app) for d in dets] for t in trs])
        r, c = linear_sum_assignment(M)
        pairs = [(i, j) for i, j in zip(r, c) if M[i, j] < BIG]
        mi = {i for i, _ in pairs}; mj = {j for _, j in pairs}
        return pairs, [i for i in range(len(trs)) if i not in mi], [j for j in range(len(dets)) if j not in mj]

    def _apply(self, tr, det, ts):
        dt = max(0.0, (ts - tr.last_ms) / 1000.0)
        new = det.ground
        if dt > 0:
            tr.vel = 0.6 * tr.vel + 0.4 * (new - tr.pos) / dt
        tr.pos = new; tr.height = 0.8 * tr.height + 0.2 * det.height
        tr.last_ms = ts; tr.hits += 1; tr.status = "visible"
        tr.last_bbox = tuple(det.bbox); tr.last_conf = det.conf; tr.pred = None
        if det.feat is not None:
            a = self.cfg.ema_appearance
            f = det.feat if tr.feat is None else a * tr.feat + (1 - a) * det.feat
            tr.feat = f / (np.linalg.norm(f) + 1e-9)
        if det.team_id is not None and (tr.team_id is None or det.team_conf >= tr.team_conf):
            tr.team_id, tr.team_conf = det.team_id, det.team_conf
        tr.history.append((ts, float(new[0]), float(new[1])))
        tr.identity_conf = min(1.0, tr.identity_conf + 0.05)
        if tr.player_id is None and tr.hits >= self.cfg.min_hits_to_confirm:
            tr.player_id = f"P{self._next_player:04d}"; self._next_player += 1
            self._log("track_created", ts, player_id=tr.player_id)

    def update(self, ts, dets):
        self.last_ms = ts
        hi = [d for d in dets if d.conf >= self.cfg.high_conf]
        lo = [d for d in dets if self.cfg.low_conf <= d.conf < self.cfg.high_conf]
        active = [t for t in self.tracks if t.player_id is None or t.status in ("visible", "occluded")]
        lost = [t for t in self.tracks if t.player_id is not None and t.status in ("temporarily_lost", "out_of_view")]
        p1, ua, uh = self._assign(active, hi, ts)
        for i, j in p1:
            self._apply(active[i], hi[j], ts)
        rem = [active[i] for i in ua]
        p2, _, _ = self._assign(rem, lo, ts, use_app=False)
        for i, j in p2:
            self._apply(rem[i], lo[j], ts)
        left = [hi[j] for j in uh]
        used = set()
        for tr in sorted(lost, key=lambda t: -t.last_ms):
            dt = max(0.0, (ts - tr.last_ms) / 1000.0)
            cands = []
            for k, d in enumerate(left):
                if k in used:
                    continue
                c = self._cost(tr, d, dt)
                if c < BIG and _sim(tr.feat, d.feat) >= self.cfg.reid_min_similarity and \
                        (tr.team_id is None or d.team_id is None or tr.team_id == d.team_id):
                    cands.append((c, k))
            cands.sort()
            if not cands:
                continue
            if len(cands) > 1 and cands[1][0] - cands[0][0] < self.cfg.reid_ambiguity_margin:
                self._log("reid_ambiguous", ts, player_id=tr.player_id)
                continue
            k = cands[0][1]; used.add(k)
            self._apply(tr, left[k], ts)
            tr.identity_conf = max(0.4, tr.identity_conf - 0.15)
            self._log("track_recovered", ts, player_id=tr.player_id)
        for k, d in enumerate(left):
            if k in used:
                continue
            self.tracks.append(Track(self._next_tmp, d.ground.copy(), np.zeros(2), d.height, d.feat, ts, ts,
                                     team_id=d.team_id, team_conf=d.team_conf, last_bbox=tuple(d.bbox), last_conf=d.conf,
                                     history=[(ts, float(d.ground[0]), float(d.ground[1]))]))
            self._next_tmp += 1
        for t in list(self.tracks):
            miss = ts - t.last_ms
            if miss <= 0:
                continue
            if t.player_id is None:
                if miss > 1000:
                    self.tracks.remove(t)  # tentative noise is dropped and never given an ID
                continue
            prev = t.status
            if miss > self.cfg.retire_ms: t.status = "retired"
            elif miss > self.cfg.lost_ms: t.status = "out_of_view"
            elif miss > 200: t.status = "temporarily_lost"
            else: t.status = "occluded"
            t.pred = t.predict(miss / 1000.0) if miss <= self.cfg.predict_ms else None
            t.identity_conf = max(0.1, t.identity_conf - 0.002 * (miss / 100))
            if prev != t.status and t.status == "temporarily_lost":
                self._log("track_lost", ts, player_id=t.player_id)
        return self.snapshot(ts)

    def merge(self, keep_id, drop_id, ts, reason):
        """Call only with strong evidence. Older (lower) ID wins; audit entry recorded."""
        if keep_id > drop_id:
            keep_id, drop_id = drop_id, keep_id
        self.canonical[drop_id] = keep_id
        self.tracks = [t for t in self.tracks if t.player_id != drop_id]
        self._log("track_merge", ts, keep=keep_id, drop=drop_id, reason=reason)

    def snapshot(self, ts):
        out = []
        for t in self.tracks:
            if t.player_id is None:
                continue
            conf = t.last_ms == ts
            out.append({
                "player_id": self.canon(t.player_id), "team_id": t.team_id, "status": t.status,
                "bbox": list(t.last_bbox) if conf and t.last_bbox else None,
                "ground": [float(t.pos[0]), float(t.pos[1])] if conf else None,
                "predicted_ground": None if conf or t.pred is None else [float(t.pred[0]), float(t.pred[1])],
                "det_conf": t.last_conf if conf else 0.0,
                "identity_conf": round(t.identity_conf, 3), "team_conf": round(t.team_conf, 3),
                "predicted": not conf,
            })
        return out
