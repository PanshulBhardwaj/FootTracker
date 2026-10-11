import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, cv2
from tactivision.camera import CameraMotion

def scene(seed=3, w=1600, h=900):
    rng = np.random.default_rng(seed); img = np.full((h, w, 3), 60, np.uint8)
    for _ in range(900):
        c = tuple(int(v) for v in rng.integers(20, 255, 3)); p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(img, p, int(rng.integers(3, 14)), c, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)

class T(unittest.TestCase):
    def test_pan_zoom_recovered(self):
        big = scene(); W, H = 1280, 720
        def frame(t):  # camera pans right 6 px/frame and zooms in slowly
            z = 1 + 0.002 * t; M = np.array([[z, 0, 100 + 6 * t], [0, z, 80 + 1.5 * t]], float)  # crop -> big
            return cv2.warpAffine(big, M, (W, H), flags=cv2.WARP_INVERSE_MAP), np.vstack([M, [0, 0, 1]])
        cm = CameraMotion(); f0, M0 = frame(0); cm.update(f0, [])
        for t in range(1, 25):
            ft, Mt = frame(t); seg, cam = cm.update(ft, [])
            self.assertEqual(seg, 0)
            # truth: frame-t pixel -> big -> frame-0 pixel
            true = np.linalg.inv(M0) @ Mt   # frame-t coords -> big? (M maps crop->big), so crop_t -> crop_0 = M0^-1 Mt
            for p in ([200, 300, 1], [900, 500, 1], [640, 360, 1]):
                a = cam @ p; a = a[:2] / a[2]; b = true @ p; b = b[:2] / b[2]
                self.assertLess(np.linalg.norm(a - b), 3.0, (t, p, a, b))
    def test_cut_starts_new_shot(self):
        cm = CameraMotion(); cm.update(scene(1)[:720, :1280], [])
        seg, cam = cm.update(scene(2)[:720, :1280], [], cut=True)
        self.assertEqual(seg, 1); self.assertTrue(np.allclose(cam, np.eye(3)))
    def test_unrelated_frame_starts_new_shot(self):
        cm = CameraMotion(); cm.update(scene(1)[:720, :1280], [])
        seg, _ = cm.update(scene(9)[:720, :1280], []); self.assertEqual(seg, 1)
if __name__ == "__main__": unittest.main()
