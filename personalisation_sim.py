"""Simulation: does FCCL's per-user learning help when users differ in how much compression they tolerate? (E32)

This is a SIMULATION. It tests whether the mechanism can work, not whether real people behave this way.

Each simulated user has a tolerance t: they are unhappy when a plan keeps less than the fraction t of the text.
t is drawn from a normal distribution (mean M, spread S), clipped to 0.3 to 1.0; the report sweeps M over 0.55, 0.65, 0.75
(FCCL's default plan keeps about 0.65 for a neutral message, so M decides whether the default is too hard, about right or too soft
for the typical user) and S over 0, 0.1, 0.2. The first run used only M = 0.75; the sweep over M was added AFTER seeing that run. A user who is unhappy complains
with probability 0.5 (show_more, correction, regenerate or thumbs_down); a happy user thumbs up with probability 0.1
and falsely complains with probability 0.03. Utility per request = (1 - keep) - 1.0 * unhappy, so saving all the
text is worth 1 and an unhappy answer costs 1. (The unhappy rate and the saving are printed separately, so the
conclusion does not hinge on that cost.)

Policies, all on the same users, the same request texts and the same random numbers:
  oracle   knows each user's tolerance and plans exactly that (an upper bound nobody can reach)
  global   one fixed keep level for everybody, the best one chosen with hindsight (a strong non-personal baseline)
  fixed    FCCL's own plan with learning switched off (no personalisation)
  learn    FCCL with its normal per-user learning

PRE-REGISTERED PREDICTIONS (written before the first run):
  P1  spread 0, any setting: learn is no better than the best global level (nothing to personalise).
  P2  spread 0.15, 40 requests per user, min_keep 0.3: learn beats fixed on mean utility, and the paired 95% interval excludes 0.
  P3  with FCCL's default min_keep 0.6 the gain of learn over fixed is small (under 0.02 utility), because the plan can
      only move between keeping 60% and 100%, so it can personalise only toward keeping MORE text.
  P4  at 40 requests per user, learn does NOT beat the best global level at spread 0.15 (one global level is hard to beat
      when each user gives about 4 complaints).

  python personalisation_sim.py                         40 requests per user, 500 users, both min_keep settings
  python personalisation_sim.py requests=200 users=300
  python personalisation_sim.py minkeep=0.6             one setting only
Writes personalisation_sim_results.txt. Needs only fccl.py (no API key, no downloads).
"""
import random
import statistics
import sys

from fccl import FCCL, Request, Feedback

TEXTS = [
    "Can you check the invoice 4821 and the delivery schedule for the meeting with Rahul tomorrow at 4 PM?",   # logistics
    "Ugh I feel so stressed and annoyed today, just venting about everything that went wrong.",                 # chatty
    "Tell me more about how the new feature works in the app.",                                                 # neutral
]
NEGATIVE = ["show_more", "correction", "regenerate", "thumbs_down"]
LAMBDA = 1.0
P_COMPLAIN, P_FALSE, P_UP = 0.5, 0.03, 0.1


def make_user(rng, spread, n_requests, mean=0.75):
    tol = min(1.0, max(0.3, rng.gauss(mean, spread)))
    seq = [(rng.randrange(len(TEXTS)), rng.random(), rng.random()) for _ in range(n_requests)]
    return tol, seq


def reaction(unhappy, u1, u2):
    """The user's feedback to one answer, from two fixed uniform numbers (common random numbers across policies)."""
    if unhappy:
        return NEGATIVE[int(u2 * len(NEGATIVE)) % len(NEGATIVE)] if u1 < P_COMPLAIN else "none"
    if u1 < P_FALSE:
        return NEGATIVE[int(u2 * len(NEGATIVE)) % len(NEGATIVE)]
    return "thumbs_up" if u1 > 1 - P_UP else "none"


def utility(keep, tol):
    unhappy = keep < tol
    return (1 - keep) - LAMBDA * unhappy, unhappy


def run_fccl(users, learn, min_keep, seed=0):
    layer = FCCL(min_tokens=0, seed=seed, min_keep=min_keep, learn=learn)
    per_user = []
    for i, (tol, seq) in enumerate(users):
        uid = f"u{i}"
        tot, bad, kept = 0.0, 0, 0.0
        for k, (ti, u1, u2) in enumerate(seq):
            text = TEXTS[ti]
            plan = layer.plan(Request(uid, text, int(len(text.split()) * 1.3)))
            u, unhappy = utility(plan.rate, tol)
            tot += u
            bad += unhappy
            kept += plan.rate
            layer.feedback(Feedback(uid, reaction(unhappy, u1, u2)))
        n = len(seq)
        per_user.append((tot / n, bad / n, kept / n))
    return per_user


def run_fixed_keep(users, keep):
    out = []
    for tol, seq in users:
        u, unhappy = utility(keep, tol)
        out.append((u, float(unhappy), keep))
    return out


def run_oracle(users, min_keep):
    return [(utility(max(min_keep, tol), tol)[0], float(max(min_keep, tol) < tol), max(min_keep, tol)) for tol, _ in users]


def mean_col(rows, j):
    return statistics.mean(r[j] for r in rows)


def paired_ci(a, b, reps=2000, seed=0):
    """Mean of a-b over users with a bootstrap 95% interval (resampling users)."""
    d = [x[0] - y[0] for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(reps))
    return statistics.mean(d), means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def one_setting(spread, n_users, n_requests, min_keep, seed=1, mean=0.75):
    rng = random.Random(seed)
    users = [make_user(rng, spread, n_requests, mean) for _ in range(n_users)]
    grid = [round(min_keep + 0.05 * i, 2) for i in range(int(round((1.0 - min_keep) / 0.05)) + 1)]
    glob_all = {g: run_fixed_keep(users, g) for g in grid}
    best_g = max(grid, key=lambda g: mean_col(glob_all[g], 0))
    res = {"oracle": run_oracle(users, min_keep), "global": glob_all[best_g],
           "fixed": run_fccl(users, False, min_keep), "learn": run_fccl(users, True, min_keep)}
    return res, best_g


def report(settings):
    lines = []
    for min_keep, n_requests, n_users in settings:
        lines.append(f"\nmin_keep {min_keep}, {n_requests} requests per user, {n_users} simulated users")
        lines.append(f"{'mean':>5} {'spread':>6} {'policy':>9} {'utility':>8} {'unhappy':>8} {'keep':>6}   learn - fixed [95% CI]   learn - global [95% CI]")
        for mean in (0.55, 0.65, 0.75):
            for spread in (0.0, 0.1, 0.2):
                res, best_g = one_setting(spread, n_users, n_requests, min_keep, mean=mean)
                lf = paired_ci(res["learn"], res["fixed"])
                lg = paired_ci(res["learn"], res["global"])
                for name in ("oracle", "global", "fixed", "learn"):
                    r = res[name]
                    extra = ""
                    if name == "learn":
                        extra = f"   {lf[0]:+.3f} [{lf[1]:+.3f}, {lf[2]:+.3f}]   {lg[0]:+.3f} [{lg[1]:+.3f}, {lg[2]:+.3f}]"
                    label = f"global{best_g:.2f}" if name == "global" else name
                    lines.append(f"{mean:>5.2f} {spread:>6.2f} {label:>9} {mean_col(r, 0):>8.3f} {mean_col(r, 1):>8.1%} {mean_col(r, 2):>6.2f}{extra}")
    return "\n".join(lines)


def main(argv):
    n_requests, n_users, only = 40, 500, None
    for a in argv[1:]:
        if a.startswith("requests=") and a[9:].isdigit():
            n_requests = int(a[9:])
        elif a.startswith("users=") and a[6:].isdigit():
            n_users = int(a[6:])
        elif a.startswith("minkeep="):
            only = float(a[8:])
        else:
            raise SystemExit("usage: python personalisation_sim.py [requests=N] [users=K] [minkeep=X]")
    mks = [only] if only is not None else [0.6, 0.3]
    text = ("E32 simulation of per-user learning (not evidence about real people).\n" +
            report([(mk, n_requests, n_users) for mk in mks]))
    print(text)
    open("personalisation_sim_results.txt", "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main(sys.argv)