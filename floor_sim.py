"""What does a hard minimum keep (min_keep) cost in savings, and what does it buy? (simulated users)"""
import sys

from fccl import FCCL
from fccl_sim import simulate, SCENARIOS

KNEE = 0.45     # the keep level below which the blind tests showed quality dropping


class Tracker:
    """Wraps a policy and counts how many compression plans fall below the knee."""
    def __init__(self, inner, stats):
        self.inner, self.stats = inner, stats

    def plan(self, req):
        p = self.inner.plan(req)
        if p.apply:
            self.stats["n"] += 1
            self.stats["low"] += p.rate < KNEE
        return p

    def feedback(self, fb):
        self.inner.feedback(fb)


n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
makers = {}
for floor in (0.1, 0.3, 0.45, 0.6):
    makers[f"strength 1.0, min_keep {floor}"] = (
        lambda f: (lambda s: FCCL(seed=s, ctx_strength=1.0, min_keep=f)))(floor)
for floor in (0.1, 0.45):
    makers[f"adaptive trust, min_keep {floor}"] = (
        lambda f: (lambda s: FCCL(seed=s, adaptive_trust=True, min_keep=f)))(floor)

for scen, kw in SCENARIOS.items():
    print(f"\n{scen}")
    print(f"{'policy':<32}{'tokens saved %':>15}{'bad % (first 10)':>18}{'bad % (last 20)':>17}{'plans < 0.45 %':>16}")
    for name, mk in makers.items():
        stats = {"n": 0, "low": 0}
        rs = [simulate(lambda s, mk=mk: Tracker(mk(s), stats), seed, **kw) for seed in range(n)]
        avg = lambda k: sum(r[k] for r in rs) / len(rs)
        low = 100 * stats["low"] / max(1, stats["n"])
        print(f"{name:<32}{avg('saved'):>15.1f}{avg('early'):>18.1f}{avg('late'):>17.1f}{low:>16.1f}")