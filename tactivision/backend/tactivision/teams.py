"""Team classification from torso colour (grass masked). Goalkeepers/referees need a 3rd+
cluster or manual correction (not implemented)."""
import numpy as np, cv2

def jersey_feature(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = y2 - y1, x2 - x1
    if h < 24 or w < 8:
        return None
    crop = frame[y1 + int(.18 * h): y1 + int(.55 * h), x1 + int(.25 * w): x2 - int(.25 * w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 85) & (hsv[..., 1] > 60)
    px = hsv[~grass]
    if len(px) < 20:
        return None
    hist = cv2.calcHist([px.reshape(-1, 1, 3)], [0, 1], None, [12, 6], [0, 180, 0, 256]).flatten()
    n = np.linalg.norm(hist)
    return hist / n if n else None

class TeamClassifier:
    def __init__(self, min_samples=40):
        self.samples, self.centres, self.min_samples = [], None, min_samples
    def observe(self, feat):
        if feat is not None and self.centres is None:
            self.samples.append(feat)
            if len(self.samples) >= self.min_samples:
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
                _, _, c = cv2.kmeans(np.float32(self.samples), 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
                self.centres = c
    def classify(self, feat):
        if self.centres is None or feat is None:
            return None, 0.0
        d = np.linalg.norm(self.centres - feat, axis=1)
        k = int(np.argmin(d))
        return ("TEAM_A", "TEAM_B")[k], float(1 - d[k] / (d.sum() + 1e-9))
