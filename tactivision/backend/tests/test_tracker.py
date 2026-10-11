import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tactivision.tracker import Tracker, Detection
from tactivision.metrics import pass_accuracy, possession

def feat(i, n=16):
    v = np.zeros(n); v[i] = 1; v += 0.02; return v / np.linalg.norm(v)
def det(x, y, f, conf=0.9, team="TEAM_A"):
    return Detection((x - 10, y - 80, x + 10, y), conf, f, team, 0.9)
def run(tr, t, ds): return {p["player_id"]: p for p in tr.update(t, ds)}

class T(unittest.TestCase):
    def warm(self, tr, n=5):
        for k in range(n):
            s = run(tr, k * 40, [det(100 + k, 300, feat(0)), det(600 + k, 300, feat(5), team="TEAM_B")])
        return s
    def test_brief_disappearance_keeps_id(self):
        tr = Tracker(); ids = set(self.warm(tr)); self.assertEqual(len(ids), 2)
        for k in range(5, 30):
            s = run(tr, k * 40, [det(600 + k, 300, feat(5), team="TEAM_B")])
        lost = [p for p in s.values() if p["status"] != "visible"]
        self.assertEqual(len(lost), 1)
        self.assertTrue(lost[0]["predicted"]); self.assertIsNone(lost[0]["bbox"])
        s = run(tr, 30 * 40, [det(130, 300, feat(0)), det(630, 300, feat(5), team="TEAM_B")])
        self.assertEqual(set(s), ids)
    def test_noise_gets_no_id(self):
        tr = Tracker(); self.warm(tr)
        s = run(tr, 400, [det(105, 300, feat(0)), det(605, 300, feat(5), team="TEAM_B"), det(900, 100, feat(9))])
        self.assertEqual(len(s), 2)
    def test_ambiguous_reid_not_forced(self):
        tr = Tracker(); self.warm(tr)
        for k in range(5, 30): run(tr, k * 40, [det(600 + k, 300, feat(5), team="TEAM_B")])
        s = run(tr, 1200, [det(125, 300, feat(0)), det(127, 300, feat(0)), det(630, 300, feat(5), team="TEAM_B")])
        self.assertTrue(any(e["event"] == "reid_ambiguous" for e in tr.audit))
        self.assertEqual(len(s), 2)  # no third ID invented
    def test_merge_audit(self):
        tr = Tracker(); a, b = sorted(self.warm(tr))
        tr.merge(b, a, 500, "test"); self.assertEqual(tr.canon(b), a)
        self.assertTrue(any(e["event"] == "track_merge" for e in tr.audit))
    def test_metrics_unavailable(self):
        self.assertIsNone(pass_accuracy([], "TEAM_A")); self.assertIsNone(possession([]))
if __name__ == "__main__": unittest.main()
