# TACTIVISION: AI-powered football intelligence

Dashboard + tracking backend built from the master prompt and reference screenshot.

## Status: read this first
This is a **working foundation, not the finished product.** It was built in a sandbox with no GPU, no model weights, no football video and no network, so only the pieces that need none of those were verified.

| Area | State |
|---|---|
| Persistent-ID tracker (Hungarian cascade, dt-aware gating, lost-track re-ID, ambiguity guard, merge audit) | Implemented, **5 unit tests pass** (synthetic data) |
| Predicted/lost tracks never shown as confirmed | Implemented + tested |
| Timestamped video pipeline (POS_MSEC, camera-cut check, quality modes, latency stats) | Written, **not run on real video** |
| Detector | Needs **your** weights (`backend/weights/football-players.pt`, classes `player`/`goalkeeper`/`ball`) and `pip install torch ultralytics` (AGPL-3.0, check licence) |
| Team classification (torso HSV + k-means) | Written, untested on real footage. Referees/goalkeepers not separated |
| Manual pitch calibration in the page (click 4+ points, homography, error in metres, projected pitch lines for checking; saved per video in the browser). Math tested on synthetic data only; valid only while the camera is still | Implemented |
| Camera-motion tracking (`backend/tactivision/camera.py`): one manual calibration follows pans/zooms within a shot; each camera cut starts a new shot that needs its own calibration | Implemented. Unit-tested on synthetic pans/zooms (3 px tolerance); page-level test with a synthetic panning state (0.1 m mapping error). **Not run on real footage; drift grows over long shots** |
| Automatic pitch calibration (homography, reprojection check, smoothing, invalidate on cut) | Written. **No pitch-landmark model included**; until you pass `landmarks_fn`, pitch coords are `null` and the minimap/heatmaps show empty states |
| OCR jersey numbers | Not implemented (labels use `P0001`-style IDs) |
| Ball ownership, pass detection, possession %, pass accuracy, progressive passes, ball recoveries (in `frontend/index.html`, rules in `docs/METRICS.md`) | Implemented; **7 logic tests pass on synthetic data (`node frontend/tests/passes.test.js`)**; never run on real footage. Depends on the ball being detected and on tracker IDs being stable |
| Pressure index and pressing chart (proxy, see METRICS.md) | Implemented, tested on synthetic data |
| Ball tracker (`ball.py`): picks the ball candidate that fits its motion, ignores jumps, bridges short gaps (<= 0.8 s, flagged interpolated) | Implemented; 3 unit tests; checked with a simulated bad detector. **Not run with a real detector** |
| xThreat, danger entries, territory | Not implemented; cards show `—` |
| Frontend (layout per screenshot, overlay, minimap, heatmaps, pass map/network, pitch-control proxy, diag panel) | Implemented; renders honest empty states; **not visually tested against the screenshot** |
| Benchmarks / accuracy / FPS claims | None. Nothing measured |

## Run
```
pip install -r requirements.txt
cd backend && python -m unittest discover -s tests     # tracker tests
uvicorn server:app --port 8000                          # then open http://localhost:8000
```
In the page: choose a video, then "Analyse on server" (needs detector weights), or load a pre-made `state.json` to explore the UI.

## Layout
- `backend/tactivision/` tracker, pitch, teams, metrics, pipeline, config
- `backend/server.py` FastAPI job API + serves the frontend
- `frontend/index.html` single-file dashboard (reads `state.json`; schema = `pipeline.process` output)
- `docs/METRICS.md` metric definitions and implementation status

## Next steps, in order
1. Obtain/train a football detector; run `process()` on a clip; check ID stability using the audit log in the diag panel.
2. Add a pitch-keypoint model and pass it as `landmarks_fn`; verify reprojection error on pans/zooms/cuts.
3. Add ball-ownership + pass event detection, then wire `metrics.py`.
4. Build a labelled test clip set and measure identity switches, FPS (end-to-end), p95 latency per mode.

## Using it
1. Re-run **Analyse on server** whenever you update the backend (camera data is created during analysis). Use Balanced mode; High throughput skips many frames and makes camera tracking less reliable.
2. Pause on a clear frame, **Calibrate**, 4+ points, **Apply**. Repeat once per camera shot (the status line shows shots vs calibrated shots).
3. Adjust **Ball link** (strict/normal/loose) and **Team A attacks** as needed.
4. Tests: `cd backend && python -m unittest discover -s tests`; `node frontend/tests/passes.test.js`; `frontend/tests/e2e_synthetic.py` (needs Playwright; paths inside are for the author's machine).
