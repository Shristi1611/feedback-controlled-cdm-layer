"""Paired analysis of the SQuAD LLM test: is each compressed variant really worse than the original?

Uses the same 60 questions as the real run. For each variant it counts, per question:
  lost   = original answered correctly, compressed did not
  gained = original wrong, compressed right
and runs an exact sign test (McNemar) plus a bootstrap 95% interval on the difference.

  python paired_analysis.py          (uses the 60 questions)
  python paired_analysis.py 60
"""
import json
import math
import random
import sys

CACHE = "squad_llm_cache.json"
SQUAD_VAL_SIZE = 10570            # size of the SQuAD v1.1 validation split
n = int(sys.argv[1]) if len(sys.argv) > 1 else 60

cache = json.load(open(CACHE, encoding="utf-8"))
idxs = random.Random(0).sample(range(SQUAD_VAL_SIZE), n)

VARIANTS = [("LLMLingua-2", 0.8), ("LLMLingua-2", 0.6), ("LLMLingua-2", 0.45),
            ("LLMLingua-2", 0.3), ("SimpleCDM", 0.6), ("SimpleCDM", 0.45)]


def get(i, method, rate):
    r = cache.get(f"{i}|{method}|{rate}")
    return None if r is None else bool(r["correct"])


def sign_test_p(lost, gained):
    """Exact two-sided p-value: if there were no real difference, lost and gained would split 50/50."""
    m = lost + gained
    if m == 0:
        return 1.0
    k = min(lost, gained)
    p = sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m
    return min(1.0, 2 * p)


rng = random.Random(1)
lines = [f"Paired comparison vs original, {n} questions (difference = compressed minus original, in points).",
         f"{'method':<13}{'keep':>5}{'lost':>6}{'gained':>8}{'diff':>8}{'95% interval':>20}{'p-value':>10}"]
for method, rate in VARIANTS:
    pairs = []
    for i in idxs:
        o, c = get(i, "original", 1.0), get(i, method, rate)
        if o is not None and c is not None:
            pairs.append((o, c))
    if not pairs:
        lines.append(f"{method:<13}{rate:>5}  no data")
        continue
    lost = sum(1 for o, c in pairs if o and not c)
    gained = sum(1 for o, c in pairs if (not o) and c)
    diffs = [int(c) - int(o) for o, c in pairs]
    mean = 100 * sum(diffs) / len(diffs)
    boots = sorted(100 * sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(5000))
    lo, hi = boots[int(0.025 * 5000)], boots[int(0.975 * 5000)]
    lines.append(f"{method:<13}{rate:>5}{lost:>6}{gained:>8}{mean:>+8.1f}{f'[{lo:+.1f}, {hi:+.1f}]':>20}"
                 f"{sign_test_p(lost, gained):>10.3f}")

lines.append("")
lines.append("p < 0.05 means the drop is unlikely to be chance. An interval that stays below 0 says the same.")
print("\n".join(lines))
with open("paired_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))