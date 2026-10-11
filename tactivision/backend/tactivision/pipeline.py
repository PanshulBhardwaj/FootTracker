"""Video -> state.json. Timestamps come from the video (POS_MSEC). Never fabricates detections."""
import json, os, time
import numpy as np, cv2
from .config import PipelineConfig, MODES
from .tracker import Tracker, Detection
from .teams import TeamClassifier, jersey_feature
from .pitch import PitchCalibrator, is_camera_cut
from .camera import CameraMotion
from .ball import BallTracker, fill_ball_gaps
from .metrics import heat_grid

def load_detector(cfg):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed: `pip install torch ultralytics`, then supply football weights.")
    if not os.path.exists(cfg.detector_weights):
        raise RuntimeError("Detector weights missing: %s (needs a model trained on football player+ball classes)" % cfg.detector_weights)
    return YOLO(cfg.detector_weights)

def process(video_path, out_json, cfg=None, landmarks_fn=None, progress=None):
    cfg = cfg or PipelineConfig()
    model = load_detector(cfg)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video (unsupported codec?): %s" % video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker, teams, calib, camera = Tracker(cfg.tracker), TeamClassifier(), PitchCalibrator(), CameraMotion()
    ball_tr = BallTracker(lost_ms=cfg.ball_gap_ms)
    frames, prev_hist, last_proc, skipped, lat = [], None, -1e9, 0, []
    gap = MODES[cfg.mode] * 1000
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        hist = cv2.calcHist([cv2.resize(frame, (64, 36))], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        cv2.normalize(hist, hist)
        cut = is_camera_cut(prev_hist, hist, cfg.cut_hist_threshold); prev_hist = hist
        if cut:
            calib.invalidate(); ball_tr.pos = None
        if ts - last_proc < gap and not cut:
            skipped += 1; continue
        t0 = time.time(); last_proc = ts
        res = model.predict(frame, imgsz=cfg.detector_imgsz, verbose=False, conf=min(cfg.tracker.low_conf, 0.05))[0]
        dets, cands = [], []
        for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            name = res.names[int(k)]
            if name in ("player", "person", "goalkeeper"):
                if c < cfg.tracker.low_conf: continue
                f = jersey_feature(frame, b); teams.observe(f); tid, tc = teams.classify(f)
                dets.append(Detection(tuple(float(v) for v in b), float(c), f, tid, tc))
            elif name in ("ball", "sports ball"):
                cands.append((float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2), float(c)))
        ball = ball_tr.update(ts, cands)
        seg, cam = camera.update(frame, [d.bbox for d in dets if d.conf >= 0.3], cut)
        if landmarks_fn:
            ip, pp = landmarks_fn(frame); calib.update(ip, pp, ts)
        players = tracker.update(ts, dets)
        for p in players:
            p["pitch"] = None if p["predicted"] else calib.to_pitch(p["ground"])
            p["pitch_conf"] = round(calib.conf, 3) if p["pitch"] else 0.0
        if ball:
            ball["pitch"] = calib.to_pitch(ball["image"])
        frames.append({"t_ms": ts, "players": players, "ball": ball, "calibration_conf": round(calib.conf, 3), "camera_cut": bool(cut), "seg": int(seg), "cam": [round(float(v), 6) for v in cam.ravel()]})
        lat.append(time.time() - t0)
        if progress and len(frames) % 25 == 0:
            progress(ts, total / fps * 1000)
    cap.release()
    fill_ball_gaps(frames, cfg.ball_gap_ms)
    pts = {"TEAM_A": [], "TEAM_B": []}
    for fr in frames:
        for p in fr["players"]:
            if p.get("pitch") and not p["predicted"] and p["team_id"] in pts:
                pts[p["team_id"]].append((*p["pitch"], p["det_conf"]))
    state = {
        "meta": {"video": os.path.basename(video_path), "fps": fps, "mode": cfg.mode, "frames_skipped_by_scheduler": skipped,
                 "median_latency_ms": float(np.median(lat) * 1000) if lat else None,
                 "p95_latency_ms": float(np.percentile(lat, 95) * 1000) if lat else None},
        "teams": {"TEAM_A": {"name": None}, "TEAM_B": {"name": None}},
        "frames": frames, "passes": [], "audit": tracker.audit,
        "metrics": {k: None for k in ("possession", "territory", "xthreat", "danger_entries", "pass_accuracy",
                                      "pressure_index", "progressive_passes", "ball_recoveries")},
        "heatmaps": {k: heat_grid(v) for k, v in pts.items()},
    }
    with open(out_json, "w") as f:
        json.dump(state, f)
    return state
