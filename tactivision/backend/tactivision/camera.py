"""Camera-motion tracking so ONE manual calibration covers a panning/zooming shot.

A camera that pans/zooms (rotates about its centre) relates any two frames by a homography,
whatever the scene depth. We estimate frame-to-frame homographies from background features
(players masked out) and chain them: cam_t maps frame-t image coords -> the shot's first frame.
On a camera cut, or when tracking fails, a new shot (seg) starts and needs its own calibration.
Drift accumulates slowly; keep calibrations close in time to what you analyse.
"""
import numpy as np, cv2

class CameraMotion:
    def __init__(self, width=640, min_inliers=15, min_ratio=0.45):
        self.width, self.min_inliers, self.min_ratio = width, min_inliers, min_ratio
        self.prev = None; self.prev_mask = None
        self.cam = np.eye(3); self.seg = 0

    def _new_shot(self, g, mask):
        if self.prev is not None:
            self.seg += 1
        self.cam = np.eye(3); self.prev, self.prev_mask = g, mask
        return self.seg, self.cam.copy()

    def update(self, frame, boxes, cut=False):
        """boxes: player/ball boxes (x1,y1,x2,y2) in full-res px to exclude. Returns (seg, cam 3x3)."""
        h, w = frame.shape[:2]
        s = min(1.0, self.width / float(w))
        small = cv2.resize(frame, None, fx=s, fy=s) if s != 1.0 else frame
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = np.full(g.shape, 255, np.uint8)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(mask, (int(x1 * s) - 4, int(y1 * s) - 4), (int(x2 * s) + 4, int(y2 * s) + 4), 0, -1)
        if cut or self.prev is None:
            return self._new_shot(g, mask)
        p0 = cv2.goodFeaturesToTrack(self.prev, 500, 0.01, 8, mask=self.prev_mask)
        if p0 is None or len(p0) < self.min_inliers:
            return self._new_shot(g, mask)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, None, winSize=(31, 31), maxLevel=4)
        ok = st.ravel() == 1
        if ok.sum() < self.min_inliers:
            return self._new_shot(g, mask)
        a, b = p1[ok].reshape(-1, 2), p0[ok].reshape(-1, 2)      # current -> previous
        Hs, inl = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
        if Hs is None or inl.sum() < self.min_inliers or inl.sum() / len(a) < self.min_ratio:
            return self._new_shot(g, mask)
        det = np.linalg.det(Hs[:2, :2])
        if not (0.6 < det < 1.6) or abs(Hs[2, 0]) > 5e-3 or abs(Hs[2, 1]) > 5e-3:   # implausible motion
            return self._new_shot(g, mask)
        S = np.diag([s, s, 1.0])
        Hf = np.linalg.inv(S) @ Hs @ S
        self.cam = self.cam @ (Hf / Hf[2, 2])
        self.prev, self.prev_mask = g, mask
        return self.seg, self.cam.copy()
