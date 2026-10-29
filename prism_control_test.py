"""PRISM follow-up: (1) control for which model wrote the response, (2) does the length preference depend on conversation type?

Variants of the same online per-user test as before:
  length only   : as before
  score adj.    : subtract each model's average score (removes 'this model is just better')
  both adj.     : also subtract each model's usual length (only within-model length differences remain)
Accuracies are only comparable between 'global' and 'user' WITHIN a variant (the target differs per variant).

  python prism_control_test.py
"""
import json
from collections import defaultdict

import numpy as np
from huggingface_hub import hf_hub_download

REPO = "HannahRoseKirk/prism-alignment"
KS = [0, 3, 10, 30]
MIN_PRIOR = 2

path = hf_hub_download(REPO, "utterances.jsonl", repo_type="dataset")
groups = defaultdict(list)
msum = defaultdict(lambda: [0.0, 0.0, 0])
with open(path, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("score") is None or not r.get("model_response"):
            continue
        lw = float(np.log1p(len(r["model_response"].split())))
        sc = float(r["score"])
        try:
            conv = int(str(r["conversation_id"]).lstrip("c"))
        except ValueError:
            conv = 0
        groups[(r["user_id"], r["interaction_id"])].append(
            (conv, r["turn"], lw, sc, r["model_name"], r["conversation_type"]))
        m = msum[r["model_name"]]
        m[0] += lw
        m[1] += sc
        m[2] += 1
mlw = {k: v[0] / v[2] for k, v in msum.items()}
msc = {k: v[1] / v[2] for k, v in msum.items()}


def build(variant):
    by_user = defaultdict(list)
    for (u, _), rows in groups.items():
        if len(rows) < 2:
            continue
        x = np.array([r[2] for r in rows])
        y = np.array([r[3] for r in rows])
        ms = [r[4] for r in rows]
        if variant in ("score", "both"):
            y = y - np.array([msc[m] for m in ms])
        if variant == "both":
            x = x - np.array([mlw[m] for m in ms])
        x, y = x - x.mean(), y - y.mean()
        if not x.any():
            continue
        iu = np.triu_indices(len(x), 1)
        dx = (x[:, None] - x[None, :])[iu]
        dy = (y[:, None] - y[None, :])[iu]
        k = dy != 0
        by_user[u].append((rows[0][0], rows[0][1], float((x * y).sum()), float((x * x).sum()),
                           dx[k], dy[k], rows[0][5]))
    for u in by_user:
        by_user[u].sort(key=lambda t: (t[0], t[1]))
    return by_user


def score_pairs(b, dx, dy):
    if len(dx) == 0:
        return 0.0, 0
    pred = b * dx
    corr = np.where(pred == 0, 0.5, (np.sign(pred) == np.sign(dy)).astype(float))
    return float(corr.sum()), len(dx)


def evaluate(by_user, title):
    SXY = sum(t[2] for ts in by_user.values() for t in ts)
    SXX = sum(t[3] for ts in by_user.values() for t in ts)
    methods = ["global"] + [f"user k={k}" for k in KS]
    acc = {m: {} for m in methods}
    for u, turns in by_user.items():
        uxy, uxx = sum(t[2] for t in turns), sum(t[3] for t in turns)
        b_g = (SXY - uxy) / (SXX - uxx)
        cxy = cxx = 0.0
        tot = {m: [0.0, 0] for m in methods}
        for t, (_, _, sxy, sxx, dx, dy, _) in enumerate(turns):
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
            cxy += sxy
            cxx += sxx
        for m in methods:
            if tot[m][1] > 0:
                acc[m][u] = tot[m][0] / tot[m][1]
    users = sorted(acc["global"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(users), (3000, len(users)))
    g = np.array([acc["global"][u] for u in users])
    out = [f"--- {title} --- ({len(users)} users, global slope {SXY / SXX:+.3f})",
           f"{'method':<12}{'accuracy %':>11}{'vs global':>11}{'95% interval':>20}"]
    for m in methods:
        a = np.array([acc[m][u] for u in users])
        if m == "global":
            out.append(f"{m:<12}{100 * a.mean():>11.1f}{'-':>11}{'-':>20}")
            continue
        d = a - g
        lo, hi = np.percentile(d[idx].mean(1) * 100, [2.5, 97.5])
        out.append(f"{m:<12}{100 * a.mean():>11.1f}{100 * d.mean():>+11.1f}{f'[{lo:+.1f}, {hi:+.1f}]':>20}")
    odd, even = [], []
    for ts in by_user.values():
        if len(ts) >= 8:
            a_ = [t for i, t in enumerate(ts) if i % 2 == 0]
            b_ = [t for i, t in enumerate(ts) if i % 2 == 1]
            sa, sb = sum(t[3] for t in a_), sum(t[3] for t in b_)
            if sa > 0 and sb > 0:
                odd.append(sum(t[2] for t in a_) / sa)
                even.append(sum(t[2] for t in b_) / sb)
    out.append(f"stability of a user's preference (odd vs even turns): r = {np.corrcoef(odd, even)[0, 1]:+.2f}")
    return out, by_user


lines = ["PRISM follow-up. 'vs global' > 0 means learning the user beats the one-size rule.", ""]
base = None
for variant, title in [("length", "length only (as before)"), ("score", "model-adjusted score"),
                       ("both", "model-adjusted score and length")]:
    out, bu = evaluate(build(variant), title)
    if variant == "length":
        base = bu
    lines += out + [""]

# reference: how well does knowing only WHICH MODEL wrote it predict ratings?
c = n = 0.0
for rows in groups.values():
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dy = rows[i][3] - rows[j][3]
            if dy == 0:
                continue
            p = msc[rows[i][4]] - msc[rows[j][4]]
            c += 0.5 if p == 0 else float(np.sign(p) == np.sign(dy))
            n += 1
lines.append(f"reference: predicting from model identity alone orders {100 * c / n:.1f}% of pairs correctly "
             "(compare with length alone ~59%)")

# context: does the length preference depend on conversation type?
lines += ["", "By conversation type (length only): does 'longer is better' hold everywhere?",
          f"{'type':<22}{'turns':>8}{'slope':>9}{'longer-wins accuracy %':>25}"]
by_type = defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0])
for turns in base.values():
    for (_, _, sxy, sxx, dx, dy, ct) in turns:
        s = by_type[ct]
        s[0] += sxy
        s[1] += sxx
        s[2] += 1
        cc, nn = score_pairs(1.0, dx, dy)
        s[3] += cc
        s[4] += nn
for ct, s in sorted(by_type.items()):
    lines.append(f"{ct:<22}{s[2]:>8}{s[0] / s[1]:>+9.2f}{100 * s[3] / max(1, s[4]):>25.1f}")

print("\n".join(lines))
with open("prism_control_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))