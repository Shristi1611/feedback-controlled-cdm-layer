"""PRISM test: does learning one user's length preference beat a one-size-fits-all rule?

Real people rated 1-100 several model responses to the same prompt. For each user we go through their
turns in order; for each turn we predict which response they rated higher using ONLY earlier turns
of that user (plus other users for the global rule), from response length alone.

  python prism_length_test.py

Public research data (human text CC-BY-4.0). Writes prism_results.txt.
"""
import json
from collections import defaultdict

import numpy as np
from huggingface_hub import hf_hub_download

REPO = "HannahRoseKirk/prism-alignment"
KS = [0, 1, 3, 10, 30, 100]          # blend strength: bigger = trust the population more
MIN_PRIOR = 2                        # need this many earlier turns before predicting
BUCKETS = [(2, 3), (4, 6), (7, 10), (11, 999)]

path = hf_hub_download(REPO, "utterances.jsonl", repo_type="dataset")
groups = defaultdict(list)
with open(path, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("score") is None or not r.get("model_response"):
            continue
        try:
            conv = int(str(r["conversation_id"]).lstrip("c"))
        except ValueError:
            conv = 0
        groups[(r["user_id"], r["interaction_id"])].append(
            (conv, r["turn"], len(r["model_response"].split()), float(r["score"])))

by_user = defaultdict(list)
for (u, _), rows in groups.items():
    if len(rows) < 2:
        continue
    x = np.log1p(np.array([w for _, _, w, _ in rows], float))
    y = np.array([s for _, _, _, s in rows], float)
    x, y = x - x.mean(), y - y.mean()
    if not x.any():
        continue
    iu = np.triu_indices(len(x), 1)
    dx = (x[:, None] - x[None, :])[iu]
    dy = (y[:, None] - y[None, :])[iu]
    m = dy != 0
    by_user[u].append((rows[0][0], rows[0][1], float((x * y).sum()), float((x * x).sum()), dx[m], dy[m]))
for u in by_user:
    by_user[u].sort(key=lambda t: (t[0], t[1]))

SXY = sum(t[2] for ts in by_user.values() for t in ts)
SXX = sum(t[3] for ts in by_user.values() for t in ts)


def score_pairs(b, dx, dy):
    if len(dx) == 0:
        return 0.0, 0
    pred = b * dx
    corr = np.where(pred == 0, 0.5, (np.sign(pred) == np.sign(dy)).astype(float))
    return float(corr.sum()), len(dx)


methods = ["global"] + [f"user k={k}" for k in KS]
user_acc = {m: {} for m in methods}
bucket = {m: {bk: [0.0, 0] for bk in BUCKETS} for m in methods}

for u, turns in by_user.items():
    uxy = sum(t[2] for t in turns)
    uxx = sum(t[3] for t in turns)
    b_g = (SXY - uxy) / (SXX - uxx)                  # global rule WITHOUT this user
    cxy = cxx = 0.0
    tot = {m: [0.0, 0] for m in methods}
    for t, (_, _, sxy, sxx, dx, dy) in enumerate(turns):
        if t >= MIN_PRIOR:
            for m in methods:
                if m == "global":
                    b = b_g
                else:
                    k = float(m.split("=")[1])
                    b = (cxy + k * b_g) / (cxx + k) if (cxx + k) > 0 else 0.0
                c, n = score_pairs(b, dx, dy)
                tot[m][0] += c
                tot[m][1] += n
                for bk in BUCKETS:
                    if bk[0] <= t <= bk[1]:
                        bucket[m][bk][0] += c
                        bucket[m][bk][1] += n
        cxy += sxy
        cxx += sxx
    for m in methods:
        if tot[m][1] > 0:
            user_acc[m][u] = tot[m][0] / tot[m][1]

users = sorted(user_acc["global"])
rng = np.random.default_rng(0)
idx = rng.integers(0, len(users), (3000, len(users)))
lines = [f"PRISM length-preference test. {len(users)} users with enough turns. Accuracy = share of within-turn "
         "response pairs ordered correctly (50% = chance).",
         f"global length slope (all users): {SXY / SXX:+.3f}  (positive = people rate longer answers higher)", "",
         f"{'method':<14}{'accuracy %':>11}{'vs global (pts)':>17}{'95% interval':>20}{'users better/worse':>20}"]
g = np.array([user_acc["global"][u] for u in users])
for m in methods:
    a = np.array([user_acc[m][u] for u in users])
    if m == "global":
        lines.append(f"{m:<14}{100 * a.mean():>11.1f}{'-':>17}{'-':>20}{'-':>20}")
        continue
    d = a - g
    boots = d[idx].mean(1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    lines.append(f"{m:<14}{100 * a.mean():>11.1f}{100 * d.mean():>+17.1f}{f'[{lo:+.1f}, {hi:+.1f}]':>20}"
                 f"{f'{int((d > 0).sum())}/{int((d < 0).sum())}':>20}")

lines += ["", "Learning curve (pooled accuracy % by number of earlier turns the user had):",
          f"{'method':<14}" + "".join(f"{(str(bk[0]) + '-' + str(bk[1])) if bk[1] < 999 else '11+':>9}" for bk in BUCKETS)]
for m in methods:
    lines.append(f"{m:<14}" + "".join(f"{100 * bucket[m][bk][0] / max(1, bucket[m][bk][1]):>9.1f}" for bk in BUCKETS))

slopes, odd_s, even_s = [], [], []
for u, ts in by_user.items():
    if len(ts) >= 8:
        a = [t for i, t in enumerate(ts) if i % 2 == 0]
        b = [t for i, t in enumerate(ts) if i % 2 == 1]
        sa, sb = sum(t[3] for t in a), sum(t[3] for t in b)
        if sa > 0 and sb > 0:
            odd_s.append(sum(t[2] for t in a) / sa)
            even_s.append(sum(t[2] for t in b) / sb)
    if sum(t[3] for t in ts) > 0:
        slopes.append(sum(t[2] for t in ts) / sum(t[3] for t in ts))
slopes = np.array(slopes)
lines += ["", f"users who rate longer answers higher: {100 * (slopes > 0).mean():.0f}%, shorter higher: {100 * (slopes < 0).mean():.0f}%",
          f"stability of a user's length preference (odd vs even turns, {len(odd_s)} users): "
          f"correlation {np.corrcoef(odd_s, even_s)[0, 1]:+.2f}  (0 = no stable personal preference)"]
print("\n".join(lines))
with open("prism_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))