"""Pitch calibration: homography image -> normalised pitch (x,y in 0..1).

No landmark detector is bundled (needs trained weights). Provide `landmarks_fn(frame) ->
(image_pts Nx2, pitch_pts Nx2 normalised)`. Without it, pitch coords are None, never guessed.
"""
import numpy as np, cv2

class PitchCalibrator:
    def __init__(self, max_reproj_px=6.0, smooth=0.7, max_stale_ms=1500):
        self.H = None; self.conf = 0.0; self.last_ms = None
        self.max_reproj_px, self.smooth, self.max_stale_ms = max_reproj_px, smooth, max_stale_ms

    def invalidate(self):
        self.H = None; self.conf = 0.0

    def update(self, image_pts, pitch_pts, ts):
        if image_pts is None or len(image_pts) < 4:
            return self._stale(ts)
        H, _ = cv2.findHomography(np.float32(image_pts), np.float32(pitch_pts), cv2.RANSAC, 0.02)
        if H is None:
            return self._stale(ts)
        back = cv2.perspectiveTransform(np.float32(pitch_pts)[None], np.linalg.inv(H))[0]
        err = float(np.mean(np.linalg.norm(back - np.float32(image_pts), axis=1)))  # reprojection error in px
        if err > self.max_reproj_px:
            return self._stale(ts)
        if self.H is not None:
            H = self.smooth * H / H[2, 2] + (1 - self.smooth) * self.H / self.H[2, 2]
        self.H, self.last_ms = H, ts
        self.conf = float(max(0.0, 1 - err / self.max_reproj_px)) * min(1.0, len(image_pts) / 8)
        return self.conf

    def _stale(self, ts):
        if self.last_ms is not None and ts - self.last_ms > self.max_stale_ms:
            self.invalidate()
        else:
            self.conf *= 0.9
        return self.conf

    def to_pitch(self, ground_xy):
        """Takes the ground-contact point. Returns None when calibration is uncertain."""
        if self.H is None or self.conf < 0.3 or ground_xy is None:
            return None
        p = cv2.perspectiveTransform(np.float32([[ground_xy]]), self.H)[0, 0]
        if not (-0.02 <= p[0] <= 1.02 and -0.02 <= p[1] <= 1.02):
            return None
        return [float(np.clip(p[0], 0, 1)), float(np.clip(p[1], 0, 1))]

def is_camera_cut(prev_hist, hist, threshold):
    return prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > threshold
