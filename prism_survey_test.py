"""PRISM: does anything in the survey predict a user's length preference? (cold-start test)

Target: each user's own length slope (positive = rates longer answers higher), from all their ratings.
Predictors: stated preferences, AI usage, free-text 'brief/detail' mentions, demographics (reference only).
Out-of-sample: 5-fold over users. Reports correlation, permutation p-value, top-vs-bottom quintile gap,
and pairwise accuracy vs the global rule. Also prints the reliability ceiling.

  python prism_survey_test.py

Public research data. Writes prism_survey_results.txt.
"""
import json
import math
from collections import Counter, defaultdict

import numpy as np
from huggingface_hub import hf_hub_download

REPO = "HannahRoseKirk/prism-alignment"
MIN_TURNS = 6
LAMBDAS = [10, 100, 1000]
PERMS = 200
BRIEF = ["concise", "short", "brief", "direct", "to the point", "succinct", "straight", "simple", "quick"]
DETAIL = ["detail", "thorough", "comprehensive", "in-depth", "in depth", "elaborate", "extensive", "depth"]


def read(name):
    path = hf_hub_download(REPO, name, repo_type="dataset")
    return [json.loads(line) for line in open(path, encoding="utf-8")]


survey = {r["user_id"]: r for r in read("survey.jsonl")}
utt = read("utterances.jsonl")

# ---- per-user turns, as in prism_length_test.py
groups = defaultdict(list)
for r in utt:
    if r.get("score") is None or not r.get("model_response"):
        continue
    try:
        conv = int(str(r["conversation_id"]).lstrip("c"))
    except ValueError:
        conv = 0
    groups[(r["user_id"], r["interaction_id"])].append(
        (conv, r["turn"], len(r["model_response"].split()), float(r["score"])))
by_user = defaultdict(list)
for (u, _), rows_ in groups.items():
    if len(rows_) < 2:
        continue
    x = np.log1p(np.array([w for _, _, w, _ in rows_], float))
    y = np.array([s for _, _, _, s in rows_], float)
    x, y = x - x.mean(), y - y.mean()
    if not x.any():
        continue
    iu = np.triu_indices(len(x), 1)
    dx = (x[:, None] - x[None, :])[iu]
    dy = (y[:, None] - y[None, :])[iu]
    m = dy != 0
    by_user[u].append((rows_[0][0], rows_[0][1], float((x * y).sum()), float((x * x).sum()), dx[m], dy[m]))
for u in by_user:
    by_user[u].sort(key=lambda t: (t[0], t[1]))

users = sorted(u for u in by_user if len(by_user[u]) >= MIN_TURNS and u in survey)
N = len(users)
sxy = np.array([sum(t[2] for t in by_user[u]) for u in users])
sxx = np.array([sum(t[3] for t in by_user[u]) for u in users])
y_slope = sxy / sxx
w = np.minimum(sxx, np.percentile(sxx, 95))
w = w / w.mean()
DX = [np.concatenate([t[4] for t in by_user[u]]) for u in users]
DY = [np.concatenate([t[5] for t in by_user[u]]) for u in users]

# ---- reliability ceiling (odd vs even turns)
odd, even = [], []
for u in users:
    ts = by_user[u]
    if len(ts) >= 8:
        a = [t for i, t in enumerate(ts) if i % 2 == 0]
        b = [t for i, t in enumerate(ts) if i % 2 == 1]
        sa, sb = sum(t[3] for t in a), sum(t[3] for t in b)
        if sa > 0 and sb > 0:
            odd.append(sum(t[2] for t in a) / sa)
            even.append(sum(t[2] for t in b) / sb)
r_half = float(np.corrcoef(odd, even)[0, 1])
r_full = 2 * r_half / (1 + r_half)
ceiling = math.sqrt(max(r_full, 0.0))


# ---- features
def as_dict(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return {}
    return v if isinstance(v, dict) else {}


def numeric_block(field):
    dicts = [as_dict(survey[u].get(field)) for u in users]
    counts = Counter(k for d in dicts for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
    keys = sorted(k for k, c in counts.items() if c >= 0.8 * N)
    cols = []
    for k in keys:
        vals = [d.get(k) if isinstance(d.get(k), (int, float)) else np.nan for d in dicts]
        med = np.nanmedian(vals)
        cols.append([med if (isinstance(v, float) and np.isnan(v)) else v for v in vals])
    return np.array(cols, float).T if cols else np.zeros((N, 0)), [f"{field}.{k}" for k in keys]


def onehot_block(field, min_count=25):
    vals = [str(survey[u].get(field)) for u in users]
    cats = [c for c, n in Counter(vals).items() if n >= min_count]
    return (np.array([[1.0 if v == c else 0.0 for c in cats] for v in vals], float) if cats else np.zeros((N, 0)),
            [f"{field}={c}" for c in cats])


def flag_block():
    txt = [str(survey[u].get("system_string") or "").lower() for u in users]
    brief = np.array([float(any(k in t for k in BRIEF)) for t in txt])
    detail = np.array([float(any(k in t for k in DETAIL)) for t in txt])
    return np.stack([brief, detail], 1), ["mentions brevity", "mentions detail"]


def stack(*blocks):
    return np.hstack([b[0] for b in blocks]), sum([b[1] for b in blocks], [])


GROUPS = {
    "stated preferences": stack(numeric_block("stated_prefs")),
    "AI usage": stack(numeric_block("lm_usecases"), onehot_block("lm_familiarity"), onehot_block("lm_frequency_use")),
    "free text: brief/detail": stack(flag_block()),
    "demographics (reference)": stack(onehot_block("age"), onehot_block("gender"), onehot_block("education")),
}
GROUPS["all except demographics"] = (np.hstack([GROUPS[k][0] for k in ("stated preferences", "AI usage", "free text: brief/detail")]),
                                     sum([GROUPS[k][1] for k in ("stated preferences", "AI usage", "free text: brief/detail")], []))


def standardize(X):
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - X.mean(0)) / sd


def ridge_fit(X, y, wt, lam):
    sw = wt.sum()
    xm = (wt[:, None] * X).sum(0) / sw
    ym = (wt * y).sum() / sw
    Xc, yc = X - xm, y - ym
    beta = np.linalg.solve(Xc.T @ (wt[:, None] * Xc) + lam * np.eye(X.shape[1]), Xc.T @ (wt * yc))
    return beta, xm, ym


def oof_predict(X, y, lam, folds):
    pred = np.zeros(N)
    for f in range(5):
        te = folds == f
        beta, xm, ym = ridge_fit(X[~te], y[~te], w[~te], lam)
        pred[te] = (X[te] - xm) @ beta + ym
    return pred


def wcorr(a, b):
    ma, mb = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
    cov = (w * (a - ma) * (b - mb)).sum()
    return cov / math.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum() + 1e-12)


def pair_acc(b, dx, dy):
    if len(dx) == 0:
        return np.nan
    p = b * dx
    return float(np.where(p == 0, 0.5, (np.sign(p) == np.sign(dy)).astype(float)).mean())


rng = np.random.default_rng(0)
folds = rng.permutation(N) % 5
# the global rule for each fold, learned only from the other users
b_glob = np.zeros(N)
for f in range(5):
    te = folds == f
    b_glob[te] = sxy[~te].sum() / sxx[~te].sum()
acc_glob = np.array([pair_acc(b_glob[i], DX[i], DY[i]) for i in range(N)])

lines = [f"PRISM survey test: {N} users with at least {MIN_TURNS} rated turns. Target = each user's own length slope.",
         f"Reliability: odd/even-turn correlation of the slope is {r_half:+.2f}, so no predictor can correlate with the "
         f"observed slope by more than about {ceiling:.2f} (the ceiling).",
         f"Share of users who rate longer answers higher: {100 * (y_slope > 0).mean():.0f}%", ""]

brief_users = GROUPS["free text: brief/detail"][0]
for j, label in enumerate(["mentions brevity", "mentions detail"]):
    m = brief_users[:, j] == 1
    lines.append(f"  users whose free text {label}: {int(m.sum())}; mean slope {y_slope[m].mean() if m.any() else float('nan'):+.2f}"
                 f" vs others {y_slope[~m].mean():+.2f}")
lines += ["", f"{'predictors':<26}{'features':>9}{'lambda':>8}{'out-of-sample r':>17}{'perm. p':>9}"
          f"{'top-vs-bottom 20%':>19}{'accuracy vs global':>24}"]
boot_idx = rng.integers(0, N, (2000, N))
for name, (X, names) in GROUPS.items():
    if X.shape[1] == 0:
        lines.append(f"{name:<26}{0:>9}   (no usable features)")
        continue
    Xs = standardize(X)
    for lam in LAMBDAS:
        pred = oof_predict(Xs, y_slope, lam, folds)
        r = wcorr(pred, y_slope)
        if lam == 100:
            perm = [wcorr(oof_predict(Xs, rng.permutation(y_slope), lam, folds), y_slope) for _ in range(PERMS)]
            # null: shuffled targets predicted, then correlated with the real targets
            perm_p = f"{(1 + sum(abs(p) >= abs(r) for p in perm)) / (PERMS + 1):.3f}"
            order = np.argsort(pred)
            k = N // 5
            gap = y_slope[order[-k:]].mean() - y_slope[order[:k]].mean()
            acc = np.array([pair_acc(pred[i], DX[i], DY[i]) for i in range(N)])
            d = acc - acc_glob
            ok = ~np.isnan(d)
            dd = d[ok]
            boots = np.sort(dd[rng.integers(0, len(dd), (2000, len(dd)))].mean(1)) * 100
            extra = (f"{gap:>+19.2f}", f"{f'{100 * dd.mean():+.1f} [{boots[50]:+.1f}, {boots[1949]:+.1f}]':>24}")
        else:
            perm_p, extra = "", ("", "")
        lines.append(f"{name:<26}{Xs.shape[1]:>9}{lam:>8}{r:>+17.3f}{perm_p:>9}{extra[0]:>19}{extra[1]:>24}")
lines += ["", "out-of-sample r: correlation between predicted and observed slope for users the model never saw.",
          "perm. p: how often shuffled targets do at least as well (small = real signal).",
          "top-vs-bottom 20%: observed slope of the 20% predicted to like longer answers most, minus the 20% predicted least.",
          "accuracy vs global: pairwise accuracy using the survey-predicted slope minus the global rule (points, 95% interval).",
          "Best-of-three lambda is slightly optimistic; the permutation p uses lambda = 100 only.",
          "Demographics are shown for reference only; I would not build a product on them."]
print("\n".join(lines))
open("prism_survey_results.txt", "w", encoding="utf-8").write("\n".join(lines))