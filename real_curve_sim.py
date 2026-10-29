"""Simulation that uses the MEASURED quality curve (blind tests E13/E14) instead of invented thresholds.

Measured: how much answer quality survives at each keep fraction.
Still invented: how much each simulated user cares (sensitivity) and the dial K.
Reports tokens saved, quality kept (from the measured curve), and bad reactions.
"""
import random
import sys

from fccl import FCCL, Request, Feedback, CompressionPlan

CHATTY_TEXTS = ["Ugh I feel so stressed about work today",
                "I'm so tired and annoyed with everything lately",
                "Honestly I just need to rant about my week"]
LOGISTICS_TEXTS = ["Book the 4 PM meeting with Rahul and send the invoice",
                   "What is the deadline for the order delivery",
                   "Schedule the ticket pickup at the address"]

# keep fraction -> answer quality kept (mean blind-test score / 6, both local methods averaged).
# The 0.10 point is an extrapolation (we never measured below 0.3).
QUALITY_POINTS = [(0.10, 0.52), (0.30, 0.76), (0.44, 0.925), (0.60, 0.99), (1.00, 1.00)]


def quality(keep):
    if keep >= 1.0:
        return 1.0
    pts = QUALITY_POINTS
    if keep <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if keep <= x1:
            return y0 + (y1 - y0) * (keep - x0) / (x1 - x0)
    return 1.0


class SimUser:
    """How much this user cares (sensitivity, 0 to 1) about lost context, per conversation type."""

    def __init__(self, rng, pattern_share, inverted_share):
        r = rng.random()
        if r < pattern_share:             # venting users care more, logistics users less
            self.sens = {"chatty": rng.uniform(0.5, 1.0), "logistics": rng.uniform(0.1, 0.5)}
        elif r < pattern_share + inverted_share:    # the opposite
            self.sens = {"chatty": rng.uniform(0.1, 0.5), "logistics": rng.uniform(0.5, 1.0)}
        else:                             # no pattern
            self.sens = {"chatty": rng.uniform(0.1, 1.0), "logistics": rng.uniform(0.1, 1.0)}

    def react(self, mode, keep, rng, K):
        p = min(1.0, K * self.sens[mode] * (1.0 - quality(keep)))    # chance the lost context hurts
        if rng.random() < p:
            return "correction"
        return "thumbs_up" if rng.random() < 0.1 else "none"


class FixedRate:
    def __init__(self, rate):
        self.rate = rate

    def plan(self, req):
        return CompressionPlan(apply=self.rate < 1.0, rate=self.rate)


def simulate(make_policy, seed, K, pattern_share, inverted_share, n_users=100, n_turns=40):
    rng = random.Random(seed)
    users = [SimUser(rng, pattern_share, inverted_share) for _ in range(n_users)]
    policy = make_policy(seed)
    kept = total = qsum = 0.0
    neg = {"early": 0, "late": 0}
    cnt = {"early": 0, "late": 0}
    for t in range(n_turns):
        phase = "early" if t < 10 else ("late" if t >= 20 else None)
        for u, user in enumerate(users):
            mode = "chatty" if rng.random() < 0.4 else "logistics"
            text = rng.choice(CHATTY_TEXTS if mode == "chatty" else LOGISTICS_TEXTS)
            ctx = rng.randint(800, 4000)
            plan = policy.plan(Request(f"u{u}", text, ctx))
            rate = plan.rate if plan.apply else 1.0
            kept += ctx * rate
            total += ctx
            qsum += ctx * quality(rate)
            kind = user.react(mode, rate, rng, K)
            if hasattr(policy, "feedback"):
                policy.feedback(Feedback(f"u{u}", kind))
            if phase:
                cnt[phase] += 1
                neg[phase] += kind == "correction"
    return dict(saved=100 * (1 - kept / total), quality=100 * qsum / total,
                early=100 * neg["early"] / cnt["early"], late=100 * neg["late"] / cnt["late"])


SCENARIOS = {
    "pattern holds for 60%": dict(pattern_share=0.6, inverted_share=0.0),
    "no pattern at all": dict(pattern_share=0.0, inverted_share=0.0),
    "pattern is backwards for 60%": dict(pattern_share=0.0, inverted_share=0.6),
}

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    makers = {
        "fixed keep 0.8": lambda s: FixedRate(0.8),
        "fixed keep 0.6": lambda s: FixedRate(0.6),
        "fixed keep 0.45": lambda s: FixedRate(0.45),
        "fixed keep 0.3": lambda s: FixedRate(0.3),
        "FCCL strength 0.0": lambda s: FCCL(seed=s, ctx_strength=0.0, min_keep=0.1),
        "FCCL strength 1.0": lambda s: FCCL(seed=s, ctx_strength=1.0, min_keep=0.1),
        "FCCL strength 1.0, min_keep 0.45": lambda s: FCCL(seed=s, ctx_strength=1.0, min_keep=0.45),
        "FCCL adaptive, min_keep 0.45": lambda s: FCCL(seed=s, adaptive_trust=True, min_keep=0.45),
    }
    for K in (3, 6):
        for scen, kw in SCENARIOS.items():
            print(f"\n[K={K}: users care {'moderately' if K == 3 else 'strongly'}]  {scen}")
            print(f"{'policy':<34}{'tokens saved %':>15}{'quality kept %':>16}{'bad % (first 10)':>18}{'bad % (last 20)':>17}")
            for name, mk in makers.items():
                rs = [simulate(mk, s, K, **kw) for s in range(n)]
                avg = lambda k: sum(r[k] for r in rs) / len(rs)
                print(f"{name:<34}{avg('saved'):>15.1f}{avg('quality'):>16.1f}{avg('early'):>18.1f}{avg('late'):>17.1f}")