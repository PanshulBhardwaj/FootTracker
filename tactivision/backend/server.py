"""Run from backend/:  uvicorn server:app --port 8000   then open http://localhost:8000"""
import os, uuid, threading
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from tactivision.config import PipelineConfig
from tactivision.pipeline import process

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)
app = FastAPI(title="Tactivision"); JOBS = {}

@app.post("/api/jobs")
async def create(video: UploadFile = File(...), mode: str = "balanced"):
    jid = uuid.uuid4().hex[:8]
    vp = os.path.join(DATA, jid + "_" + os.path.basename(video.filename))
    with open(vp, "wb") as f:
        while chunk := await video.read(1 << 20):
            f.write(chunk)
    JOBS[jid] = {"status": "running", "progress": 0, "video": vp, "state": os.path.join(DATA, jid + ".json")}
    def run():
        try:
            process(vp, JOBS[jid]["state"], PipelineConfig(mode=mode),
                    progress=lambda t, T: JOBS[jid].update(progress=round(t / max(T, 1), 3)))
            JOBS[jid]["status"] = "done"
        except Exception as e:
            JOBS[jid].update(status="error", error=str(e))
    threading.Thread(target=run, daemon=True).start()
    return {"job": jid}

def _job(jid):
    if jid not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[jid]

@app.get("/api/jobs/{jid}")
def status(jid: str):
    j = _job(jid); return {k: j.get(k) for k in ("status", "progress", "error")}

@app.get("/api/jobs/{jid}/state")
def state(jid: str): return FileResponse(_job(jid)["state"])

@app.get("/api/jobs/{jid}/video")
def video(jid: str): return FileResponse(_job(jid)["video"])

app.mount("/", StaticFiles(directory=os.path.join(HERE, "..", "frontend"), html=True), name="ui")
