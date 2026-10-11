import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tactivision.ball import BallTracker, fill_ball_gaps

class T(unittest.TestCase):
    def test_false_positive_is_ignored(self):
        bt = BallTracker(); out = []
        for k in range(20):
            x = 100 + 8 * k; cands = [(x, 300, 0.4)]
            if k % 3 == 1: cands.append((900, 100, 0.6))    # higher-confidence false ball far away
            out.append(bt.update(k * 40, cands))
        self.assertTrue(all(o and abs(o["image"][0] - (100 + 8 * k)) < 1 for k, o in enumerate(out)))
    def test_lost_ball_needs_confident_reacquire(self):
        bt = BallTracker(); bt.update(0, [(100, 100, .5)])
        self.assertIsNone(bt.update(2000, [(500, 500, .1)]))
        self.assertIsNotNone(bt.update(2040, [(500, 500, .5)]))
    def test_gap_filling_short_only(self):
        mk = lambda k, b: {"t_ms": k * 40, "ball": b}
        d = lambda x: {"image": [x, 200.0], "conf": .5, "interp": False}
        fr = [mk(0, d(100))] + [mk(k, None) for k in range(1, 5)] + [mk(5, d(200))] + [mk(k, None) for k in range(6, 40)] + [mk(40, d(400))]
        fill_ball_gaps(fr, max_gap_ms=800)
        self.assertTrue(all(fr[k]["ball"] and fr[k]["ball"]["interp"] for k in range(1, 5)))
        self.assertAlmostEqual(fr[2]["ball"]["image"][0], 140.0, places=3)
        self.assertTrue(all(fr[k]["ball"] is None for k in range(6, 40)))   # 1.4 s gap: not invented
if __name__ == "__main__": unittest.main()
