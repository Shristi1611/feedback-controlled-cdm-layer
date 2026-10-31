"""PRISM: do people differ on features other than length? (formatting, disclaimers, hedging, questions, ...)

Part 1: per-feature table (global weight, share of users leaning each way, odd/even-turn stability of a user's lean).
Part 2: online test, as in prism_length_test.py but with all features at once: for each user, in order, predict which
response in a turn they rate higher using only that user's earlier turns, vs one global rule.

  python prism_multi_test.py

Public research data. Writes prism_multi_results.txt.
"""
import json
import math
import re
from collections import defaultdict

import numpy as np
from huggingface_hub import hf_hub_download

REPO = "HannahRoseKirk/prism-alignment"
MIN_PRIOR = 2
KS_BLEND = [3, 10, 30, 100, 300]
KS_USER = [1, 3, 10]
LAMBDA0 = 1.0
BUCKETS = [(2, 3), (4, 6), (7, 10), (11, 999)]

LIST_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", re.M)
DISC = re.compile(r"as an ai|language model|i'm sorry|i am sorry|i cannot|i can't|i'm unable|i am unable|"
                  r"i don't have (?:personal|the ability)|i do not have (?:personal|the ability)", re.I)
HEDGE = re.compile(r"\b(might|may|could|perhaps|possibly|probably|depends|generally|typically|often|"
                   r"some people|it's possible|it is possible)\b", re.I)
HEAD_RE = re.compile(r"^#{1,4} ", re.M)
NAMES = ["length (log words)", "bullet list", "disclaimer/refusal", "hedging density", "asks a question",
         "bold/headings", "word complexity"]


def features(text):
    words = text.split()
    n = max(1, len(words))
    return [math.log1p(len(words)), float(len(LIST_RE.findall(text)) >= 2), float(bool(DISC.search(text))),
            100 * len(HEDGE.findall(text)) / n, float("?" in text),
            float("**" in text or bool(HEAD_RE.search(text))), sum(len(w) for w in words) / n]


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
            (conv, r["turn"], features(r["model_response"]), float(r["score"])))

turns_raw = []
for (u, _), rows_ in groups.items():
    if len(rows_) < 2:
        continue
    X = np.array([r[2] for r in rows_], float)
    y = np.array([r[3] for r in rows_], float)
    X -= X.mean(0)
    y -= y.mean()
    turns_raw.append((u, rows_[0][0], rows_[0][1], X, y))
sd = np.vstack([t[3] for t in turns_raw]).std(0)
sd[sd == 0] = 1.0

F = len(NAMES)
by_user = defaultdict(list)
for u, conv, turn, X, y in turns_raw:
    X = X / sd
    iu = np.triu_indices(len(y), 1)
    dX = X[iu[0]] - X[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    m = dy != 0
    by_user[u].append((conv, turn, X.T @ X, X.T @ y, dX[m], dy[m]))
for u in by_user:
    by_user[u].sort(key=lambda t: (t[0], t[1]))

A_tot = sum(t[2] for ts in by_user.values() for t in ts)
c_tot = sum(t[3] for ts in by_user.values() for t in ts)
I = np.eye(F)
b_all = np.linalg.solve(A_tot + LAMBDA0 * I, c_tot)

# ---------------- part 1: per-feature table
lean_pos = np.zeros(F)
n_lean = np.zeros(F)
odd_s, even_s = [[] for _ in range(F)], [[] for _ in range(F)]
for u, ts in by_user.items():
    A_u = sum(t[2] for t in ts)
    c_u = sum(t[3] for t in ts)
    for f in range(F):
        if A_u[f, f] > 0:
            n_lean[f] += 1
            lean_pos[f] += (c_u[f] / A_u[f, f]) > 0
    if len(ts) >= 8:
        a = [t for i, t in enumerate(ts) if i % 2 == 0]
        b = [t for i, t in enumerate(ts) if i % 2 == 1]
        Aa, ca = sum(t[2] for t in a), sum(t[3] for t in a)
        Ab, cb = sum(t[2] for t in b), sum(t[3] for t in b)
        for f in range(F):
            if Aa[f, f] > 0 and Ab[f, f] > 0:
                odd_s[f].append(ca[f] / Aa[f, f])
                even_s[f].append(cb[f] / Ab[f, f])
lines = [f"PRISM multi-feature test. {len(by_user)} users.", "",
         "Part 1: what people reward, and how stable each person's lean is (odd vs even turns; 0 = no stable personal trait)",
         f"{'feature':<22}{'global weight':>15}{'users leaning positive':>24}{'stability r':>13}"]
for f in range(F):
    r = np.corrcoef(odd_s[f], even_s[f])[0, 1] if len(odd_s[f]) > 3 else float("nan")
    lines.append(f"{NAMES[f]:<22}{b_all[f]:>+15.2f}{100 * lean_pos[f] / max(1, n_lean[f]):>23.0f}%{r:>+13.2f}")
lines.append("global weight = score points (1 to 100 scale) per one standard deviation of the feature within a turn.")

# ---------------- part 2: online test
methods = ["global"] + [f"user-only k={k}" for k in KS_USER] + [f"blend k={k}" for k in KS_BLEND]
acc = {m: {} for m in methods}
bucket = {m: {bk: [0.0, 0] for bk in BUCKETS} for m in methods}


def score(b, dX, dy):
    if len(dy) == 0:
        return 0.0, 0
    p = dX @ b
    return float(np.where(p == 0, 0.5, (np.sign(p) == np.sign(dy)).astype(float)).sum()), len(dy)


for u, ts in by_user.items():
    if len(ts) <= MIN_PRIOR:
        continue
    A_u = sum(t[2] for t in ts)
    c_u = sum(t[3] for t in ts)
    b_g = np.linalg.solve(A_tot - A_u + LAMBDA0 * I, c_tot - c_u)          # global rule without this user
    cumA, cumc = np.zeros((F, F)), np.zeros(F)
    tot = {m: [0.0, 0] for m in methods}
    for t, (_, _, A, c, dX, dy) in enumerate(ts):
        if t >= MIN_PRIOR:
            for m in methods:
                if m == "global":
                    b = b_g
                elif m.startswith("user-only"):
                    b = np.linalg.solve(cumA + float(m.split("=")[1]) * I, cumc)
                else:
                    k = float(m.split("=")[1])
                    b = np.linalg.solve(cumA + k * I, cumc + k * b_g)
                cc, nn = score(b, dX, dy)
                tot[m][0] += cc
                tot[m][1] += nn
                for bk in BUCKETS:
                    if bk[0] <= t <= bk[1]:
                        bucket[m][bk][0] += cc
                        bucket[m][bk][1] += nn
        cumA += A
        cumc += c
    for m in methods:
        if tot[m][1] > 0:
            acc[m][u] = tot[m][0] / tot[m][1]

users = sorted(acc["global"])
g = np.array([acc["global"][u] for u in users])
rng = np.random.default_rng(0)
idx = rng.integers(0, len(users), (3000, len(users)))
lines += ["", f"Part 2: online test, {len(users)} users. Accuracy = share of within-turn response pairs ordered correctly (50% = chance).",
          f"{'method':<16}{'accuracy %':>11}{'vs global (pts)':>17}{'95% interval':>20}{'users better/worse':>20}"]
for m in methods:
    a = np.array([acc[m][u] for u in users])
    if m == "global":
        lines.append(f"{m:<16}{100 * a.mean():>11.1f}")
        continue
    d = a - g
    lo, hi = np.percentile(d[idx].mean(1) * 100, [2.5, 97.5])
    lines.append(f"{m:<16}{100 * a.mean():>11.1f}{100 * d.mean():>+17.1f}{f'[{lo:+.1f}, {hi:+.1f}]':>20}"
                 f"{f'{int((d > 0).sum())}/{int((d < 0).sum())}':>20}")
lines += ["", "Learning curve (pooled accuracy % by number of earlier turns):",
          f"{'method':<16}" + "".join(f"{(str(b[0]) + '-' + str(b[1])) if b[1] < 999 else '11+':>9}" for b in BUCKETS)]
for m in methods:
    lines.append(f"{m:<16}" + "".join(f"{100 * bucket[m][b][0] / max(1, bucket[m][b][1]):>9.1f}" for b in BUCKETS))
print("\n".join(lines))
open("prism_multi_results.txt", "w", encoding="utf-8").write("\n".join(lines))