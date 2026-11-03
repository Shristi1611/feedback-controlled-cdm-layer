"""Simulate 'retry with the original context when the model says unknown' using the cached answers.
No API calls. Reads squad_llm_cache.json (E18) and multiwoz_llm_cache.json (E22).

  python cascade_analysis.py

Token cost counts context tokens only: compressed call = fraction kept; a retry adds the full original context.
"""
import json
import math
import os
from collections import defaultdict

FILES = [("SQuAD (E18)", "squad_llm_cache.json"), ("MultiWOZ (E22)", "multiwoz_llm_cache.json")]


def load(path):
    cache = json.load(open(path, encoding="utf-8"))
    qs = defaultdict(dict)
    for k, v in cache.items():
        parts = k.split("|")
        qs["|".join(parts[:-2])][(parts[-2], parts[-1])] = v
    return qs


def is_unknown(v):
    return bool(v.get("unknown", v.get("noans", False)))


def sign_p(lost, gained):
    m = lost + gained
    if m == 0:
        return 1.0
    k = min(lost, gained)
    return min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m)


results = {}                                    # (method, rate) -> pooled lists across datasets
for name, path in FILES:
    if not os.path.exists(path):
        print(f"{path} not found, skipping {name}")
        continue
    qs = {q: v for q, v in load(path).items() if ("original", "1.0") in v}
    variants = [k for k in set().union(*[set(v) for v in qs.values()])
                if k[0] not in ("original", "no dialogue") and all(k in v for v in qs.values())]
    variants.sort(key=lambda k: (k[0], -float(k[1])))
    print(f"\n=== {name}: {len(qs)} questions ===")
    print(f"{'method':<13}{'keep':>5} | {'plain: tokens %':>16}{'change':>22} | {'retried %':>10}{'cascade: tokens %':>19}"
          f"{'change':>22}{'p':>7} | {'wrong, no abstain':>18}")
    for var in variants:
        o = [int(v[("original", "1.0")]["correct"]) for v in qs.values()]
        c = [int(v[var]["correct"]) for v in qs.values()]
        u = [is_unknown(v[var]) for v in qs.values()]
        kept = [v[var]["kept"] for v in qs.values()]
        casc = [oo if uu else cc for oo, cc, uu in zip(o, c, u)]
        d_plain = [a - b for a, b in zip(c, o)]
        d_casc = [a - b for a, b in zip(casc, o)]
        n = len(o)
        cost_plain = 100 * sum(kept) / n
        cost_casc = 100 * (sum(kept) + sum(u)) / n
        wrong_confident = sum(1 for cc, uu, oo in zip(c, u, o) if oo and not cc and not uu)
        lost, gained = d_casc.count(-1), d_casc.count(1)
        print(f"{var[0]:<13}{var[1]:>5} | {cost_plain:>16.0f}"
              f"{f'{100 * sum(d_plain) / n:+.1f} ({d_plain.count(-1)}/{d_plain.count(1)})':>22} | "
              f"{100 * sum(u) / n:>10.0f}{cost_casc:>19.0f}"
              f"{f'{100 * sum(d_casc) / n:+.1f} ({lost}/{gained})':>22}{sign_p(lost, gained):>7.3f} | {wrong_confident:>18}")
        agg = results.setdefault(var, dict(o=[], c=[], casc=[], kept=[], u=[]))
        agg["o"] += o
        agg["c"] += c
        agg["casc"] += casc
        agg["kept"] += kept
        agg["u"] += u

print("\n=== both datasets pooled (same answerer; informal) ===")
print(f"{'method':<13}{'keep':>5} | {'plain: tokens %':>16}{'change':>22} | {'cascade: tokens %':>18}{'change':>22}{'p':>7}")
for var, a in sorted(results.items(), key=lambda kv: (kv[0][0], -float(kv[0][1]))):
    n = len(a["o"])
    if n < 100:                                   # skip variants present in only one dataset
        continue
    dp = [x - y for x, y in zip(a["c"], a["o"])]
    dc = [x - y for x, y in zip(a["casc"], a["o"])]
    lost, gained = dc.count(-1), dc.count(1)
    print(f"{var[0]:<13}{var[1]:>5} | {100 * sum(a['kept']) / n:>16.0f}"
          f"{f'{100 * sum(dp) / n:+.1f} ({dp.count(-1)}/{dp.count(1)})':>22} | "
          f"{100 * (sum(a['kept']) + sum(a['u'])) / n:>18.0f}"
          f"{f'{100 * sum(dc) / n:+.1f} ({lost}/{gained})':>22}{sign_p(lost, gained):>7.3f}")
print("\nchange = accuracy points vs the original (lost/gained questions). 'wrong, no abstain' = questions the "
      "compression broke where the model still gave a confident answer (the retry cannot catch these).")