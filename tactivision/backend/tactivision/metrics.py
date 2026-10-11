"""Metrics return None when evidence is insufficient (UI renders an em dash).
See docs/METRICS.md for definitions."""
import numpy as np

def pass_accuracy(passes, team):
    mine = [p for p in passes if p["team"] == team and p["outcome"] in ("completed", "failed")]
    return None if not mine else round(100 * sum(p["outcome"] == "completed" for p in mine) / len(mine), 1)

def possession(owner_samples):
    ctrl = [o for o in owner_samples if o]  # loose-ball samples (None) are excluded, not assigned
    if len(ctrl) < 30:
        return None
    a = sum(o == "TEAM_A" for o in ctrl) / len(ctrl) * 100
    return {"TEAM_A": round(a, 1), "TEAM_B": round(100 - a, 1)}

def heat_grid(points, nx=52, ny=34, sigma=1.4):
    """points: (x,y,w) in 0..1 from CONFIRMED calibrated observations only. None if no data."""
    if not points:
        return None
    from scipy.ndimage import gaussian_filter
    g = np.zeros((ny, nx))
    for x, y, w in points:
        g[min(ny - 1, int(y * ny)), min(nx - 1, int(x * nx))] += w
    g = gaussian_filter(g, sigma)
    return (g / g.max()).round(3).tolist() if g.max() > 0 else None
