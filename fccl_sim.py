"""Test FCCL against simulated users who have a hidden compression tolerance."""
import sys
import random
from fccl import FCCL, Request, Feedback

CHATTY_TEXTS = ["Ugh I feel so stressed about work today",
                "I'm so tired and annoyed with everything lately",
                "Honestly I just need to rant about my week"]
LOGISTICS_TEXTS = ["Book the 4 PM meeting with Rahul and send the invoice",
                   "What is the deadline for the order delivery",
                   "Schedule the ticket pickup at the address"]


class SimUser:
    """Hidden 'need': the least fraction of the context they can lose before the answer gets worse."""

    def __init__(self, rng, pattern_share=0.6, inverted_share=0.0):
        r = rng.random()
        if r < pattern_share:    # follows the pattern: chatty needs more, logistics needs less
            self.need = {"chatty": rng.uniform(0.5, 0.9), "logistics": rng.uniform(0.2, 0.6)}
        elif r < pattern_share + inverted_share:   # the opposite of the pattern
            self.need = {"chatty": rng.uniform(0.2, 0.6), "logistics": rng.uniform(0.5, 0.9)}
        else:                    # no pattern at all
            self.need = {"chatty": rng.uniform(0.2, 0.9), "logistics": rng.uniform(0.2, 0.9)}

    def react(self, mode, rate, rng):
        gap = self.need[mode] - rate
        if gap > 0 and rng.random() < min(1.0, gap * 4):
            return "correction"
        return "thumbs_up" if rng.random() < 0.1 else "none"


class FixedRate:
    def __init__(self, rate):
        self.rate = rate

    def plan(self, req):
        from fccl import CompressionPlan
        return CompressionPlan(apply=self.rate < 1.0, rate=self.rate)


def simulate(make_policy, seed, n_users=100, n_turns=40, pattern_share=0.6, inverted_share=0.0):
    rng = random.Random(seed)
    users = [SimUser(rng, pattern_share, inverted_share) for _ in range(n_users)]
    policy = make_policy(seed)
    kept = total = 0.0
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
            kind = user.react(mode, rate, rng)
            if hasattr(policy, "feedback"):
                policy.feedback(Feedback(f"u{u}", kind))
            if phase:
                cnt[phase] += 1
                neg[phase] += kind == "correction"
    return dict(saved=100 * (1 - kept / total),
                early=100 * neg["early"] / cnt["early"],
                late=100 * neg["late"] / cnt["late"])


SCENARIOS = {
    "pattern holds for 60% (the earlier test)": dict(pattern_share=0.6, inverted_share=0.0),
    "no pattern at all": dict(pattern_share=0.0, inverted_share=0.0),
    "pattern is backwards for 60%": dict(pattern_share=0.0, inverted_share=0.6),
}

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    makers = {
        "fixed keep 0.7": lambda s: FixedRate(0.7),
        "FCCL strength 0.0": lambda s: FCCL(seed=s, ctx_strength=0.0),
        "FCCL strength 0.25": lambda s: FCCL(seed=s, ctx_strength=0.25),
        "FCCL strength 0.5": lambda s: FCCL(seed=s, ctx_strength=0.5),
        "FCCL strength 1.0": lambda s: FCCL(seed=s, ctx_strength=1.0),
    }
    for scen, kw in SCENARIOS.items():
        print(f"\n{scen}")
        print(f"{'policy':<20}{'tokens saved %':>15}{'bad reactions % (first 10 turns)':>34}{'(last 20 turns)':>17}")
        for name, mk in makers.items():
            rs = [simulate(mk, s, **kw) for s in range(n)]
            avg = lambda k: sum(r[k] for r in rs) / len(rs)
            print(f"{name:<20}{avg('saved'):>15.1f}{avg('early'):>34.1f}{avg('late'):>17.1f}")