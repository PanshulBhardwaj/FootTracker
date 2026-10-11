"""Ball candidate tracking + gap filling.

Stock detectors often return false balls (logos, socks, heads) and miss the real one for many frames.
 * BallTracker: keeps the candidate that fits the ball's motion (gated by speed), ignores jumps.
 * fill_ball_gaps: linearly bridges short gaps between two trusted detections and flags those
   frames `interp: True` so downstream logic can treat them as lower confidence.
Never invents a ball across long gaps.
"""
import numpy as np

class BallTracker:
    def __init__(self, max_speed_px_s=2500.0, lost_ms=700, start_conf=0.2, min_conf=0.05):
        self.max_speed, self.lost_ms, self.start_conf, self.min_conf = max_speed_px_s, lost_ms, start_conf, min_conf
        self.pos = None; self.vel = np.zeros(2); self.last_ms = None

    def update(self, ts, cands):
        """cands: list of (x, y, conf). Returns {"image":[x,y],"conf":c,"interp":False} or None."""
        cands = [c for c in cands if c[2] >= self.min_conf]
        if self.pos is not None and ts - self.last_ms > self.lost_ms:
            self.pos = None                                   # lost: must re-acquire with a confident candidate
        if self.pos is None:
            good = [c for c in cands if c[2] >= self.start_conf]
            if not good: return None
            x, y, c = max(good, key=lambda t: t[2])
            self.pos, self.vel, self.last_ms = np.array([x, y], float), np.zeros(2), ts
            return {"image": [float(x), float(y)], "conf": float(c), "interp": False}
        dt = max((ts - self.last_ms) / 1000.0, 1e-3)
        pred = self.pos + self.vel * min(dt, 0.3)
        gate = self.max_speed * dt + 40.0
        best, bs = None, -1.0
        for x, y, c in cands:
            d = float(np.hypot(x - pred[0], y - pred[1]))
            if d <= gate:
                s = c * (1.0 - 0.5 * d / gate)
                if s > bs: best, bs = (x, y, c), s
        if best is None: return None
        x, y, c = best
        self.vel = 0.5 * self.vel + 0.5 * (np.array([x, y]) - self.pos) / dt
        self.pos, self.last_ms = np.array([x, y], float), ts
        return {"image": [float(x), float(y)], "conf": float(c), "interp": False}

def fill_ball_gaps(frames, max_gap_ms=800, max_speed_px_s=2500.0, interp_conf=0.2):
    last = None
    for i, f in enumerate(frames):
        b = f.get("ball")
        if b is None or b.get("interp"): continue
        if last is not None and i - last > 1:
            a = frames[last]["ball"]; dt = f["t_ms"] - frames[last]["t_ms"]
            dist = float(np.hypot(b["image"][0] - a["image"][0], b["image"][1] - a["image"][1]))
            if dt <= max_gap_ms and dist <= max_speed_px_s * dt / 1000.0 + 40:
                for k in range(last + 1, i):
                    r = (frames[k]["t_ms"] - frames[last]["t_ms"]) / dt
                    frames[k]["ball"] = {"image": [a["image"][0] + (b["image"][0] - a["image"][0]) * r,
                                                   a["image"][1] + (b["image"][1] - a["image"][1]) * r],
                                         "conf": interp_conf, "interp": True}
        last = i
    return frames
