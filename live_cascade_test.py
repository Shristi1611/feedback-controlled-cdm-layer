"""Live end-to-end test of the wired retry net (E27).

Runs the REAL code path on fresh SQuAD questions:
  FCCL.plan -> compressor (LLMLingua-2) -> cdm.answer_with_retry -> live Groq answers -> "abstained" feedback to FCCL

Per question it measures, against the uncompressed answer to the same question:
  - accuracy of the compressed answer WITHOUT the retry, and WITH the retry (what the pipeline really returns)
  - the real cost: words sent as a fraction of the original (a retry adds 100%), and requests per question
It reuses the E26 prediction: with the retry, loss of at most 3 points at 72% of the tokens or less.
(FCCL's own plan for neutral questions is keep 0.65, so expect about 65% of the tokens plus the retries.)

  python live_cascade_test.py 60            LLMLingua-2 (needs about 1-2 GB of free RAM)
  python live_cascade_test.py 60 simple     the free local SimpleCDM instead (no model; not the E26 compressor)

Needs GROQ_API_KEY. Resumable: every model answer is cached in live_cascade_cache.json, so rerunning replays
finished questions for free and stops cleanly at the Groq free limit. The question sample excludes every
question used in E18 and E26. Writes live_cascade_results.txt.
"""
import hashlib
import json
import math
import os
import random
import re
import string
import sys
import time

import requests

from fccl import FCCL, Request, Feedback
from cdm import LLMLinguaCDM, SimpleCDM, answer_with_retry

ANSWER_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"
PAUSE = 3.0
URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE_PATH = "live_cascade_cache.json"
OUT_PATH = "live_cascade_results.txt"

# Same prompt as the E26 replication, so the numbers are comparable (it tells the model to say "unknown").
SQUAD_SYSTEM = ("Answer the question using ONLY the passage. Reply with the shortest possible answer "
                "(a few words), with no explanation. If the passage does not contain the answer, reply: unknown")
JUDGE_SYSTEM = "You grade answers. Reply with exactly one word: YES or NO."


class RateLimited(Exception):
    pass


cache = {}


def save_cache():
    json.dump(cache, open(CACHE_PATH, "w", encoding="utf-8"))


def chat(model, system, user, max_tokens=300):
    """One Groq call, cached by content so reruns (and the baseline/retry on the same text) cost nothing."""
    key = hashlib.sha1(json.dumps([model, system, user, max_tokens]).encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]
    for _ in range(8):
        r = requests.post(URL, headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}, timeout=90, json={
            "model": model, "temperature": 0, "max_completion_tokens": max_tokens, "reasoning_effort": "low",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", 10))
            if wait > 180:
                raise RateLimited(f"Groq asks for a {wait / 60:.0f} minute wait (probably the free daily token limit).")
            time.sleep(wait + 1)
            continue
        if r.status_code != 200:
            raise SystemExit(f"Groq error {r.status_code}: {r.text[:400]}")
        out = (r.json()["choices"][0]["message"].get("content") or "").strip()
        cache[key] = out
        save_cache()
        time.sleep(PAUSE)
        return out
    raise RateLimited("Too many rate-limit retries.")


def normalize_squad(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_unknown(ans):
    na = normalize_squad(ans)
    return na == "" or na.startswith("unknown")


def grade(question, golds, ans):
    """Same rule as E18/E26: containment of a gold answer, else the judge model."""
    if is_unknown(ans):
        return False
    na = normalize_squad(ans)
    if any(normalize_squad(g) and (normalize_squad(g) in na or na == normalize_squad(g)) for g in golds):
        return True
    verdict = chat(JUDGE_MODEL, JUDGE_SYSTEM,
                   f"Question: {question}\nCorrect answers: {golds}\nModel answer: {ans}\n"
                   "Does the model answer give the same information as at least one correct answer?", max_tokens=100)
    return verdict.upper().startswith("YES")


def sign_p(lost, gained):
    m = lost + gained
    if m == 0:
        return 1.0
    k = min(lost, gained)
    return min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m)


def boot_ci(diffs, reps=4000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(100 * sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def load_squad_questions(n):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation")
    N = len(ds)
    used = set(random.Random(0).sample(range(N), 60)) | set(random.Random(0).sample(range(N), 5))
    pool = [i for i in range(N) if i not in used]
    used |= set(random.Random(1).sample(pool, 100))            # the E26 replication sample
    pool2 = [i for i in range(N) if i not in used]
    out = []
    for i in random.Random(2).sample(pool2, n):
        it = ds[i]
        out.append((str(i), it["question"], it["answers"]["text"], it["context"]))
    return out


def report(rows, layer, backend_name, total, stopped):
    n = len(rows)
    lines = [f"Live end-to-end test of the wired retry net (E27). Backend: {backend_name}. "
             f"{n} of {total} fresh SQuAD questions finished" + (" (stopped early by the rate limit)." if stopped else ".")]
    if n == 0:
        lines.append("No question finished.")
        return "\n".join(lines)
    base = [int(r["base_ok"]) for r in rows]
    plain = [int(r["first_ok"]) for r in rows]
    final = [int(r["final_ok"]) for r in rows]
    d_plain = [a - b for a, b in zip(plain, base)]
    d_final = [a - b for a, b in zip(final, base)]
    cost = 100 * sum(r["cost"] for r in rows) / n
    plain_cost = 100 * sum(r["kept"] for r in rows) / n
    retried = sum(r["retried"] for r in rows)
    lines += [
        "",
        f"{'':<28}{'accuracy':>10}{'change':>9}{'lost/gained':>13}{'p':>8}{'context words used':>20}",
        f"{'uncompressed':<28}{100 * sum(base) / n:>9.1f}%{'':>9}{'':>13}{'':>8}{'100%':>20}",
        f"{'compressed, no retry':<28}{100 * sum(plain) / n:>9.1f}%{100 * sum(d_plain) / n:>+9.1f}"
        f"{f'{d_plain.count(-1)}/{d_plain.count(1)}':>13}{sign_p(d_plain.count(-1), d_plain.count(1)):>8.3f}{plain_cost:>19.0f}%",
        f"{'compressed + retry (real)':<28}{100 * sum(final) / n:>9.1f}%{100 * sum(d_final) / n:>+9.1f}"
        f"{f'{d_final.count(-1)}/{d_final.count(1)}':>13}{sign_p(d_final.count(-1), d_final.count(1)):>8.3f}{cost:>19.0f}%",
        "",
        f"Retried {retried} of {n} questions ({100 * retried / n:.0f}%). Requests per question: {1 + retried / n:.2f} "
        f"(uncompressed: 1.00). The guard fell back (no compression) on {sum(r['fell'] for r in rows)} questions.",
        f"FCCL's chosen keep: mean {sum(r['rate'] for r in rows) / n:.2f}, range "
        f"{min(r['rate'] for r in rows):.2f} to {max(r['rate'] for r in rows):.2f}; measured words kept (before any retry) "
        f"{plain_cost:.0f}%.",
        f"Abstained feedback sent: {sum(r['abstained'] for r in rows)}; the user's satisfaction weight went 1.00 -> "
        f"{layer.sat_weight.get('u1', 1.0):.2f}.",
    ]
    lo, hi = boot_ci(d_final)
    loss = 100 * sum(d_final) / n
    ok = loss >= -3.0 and cost <= 72.0
    lines += ["",
              f"PREDICTION CHECK (E26 prediction, this run): with the retry, accuracy changes by {loss:+.1f} points "
              f"(95% interval [{lo:+.1f}, {hi:+.1f}]) at {cost:.0f}% of the context words. "
              f"Prediction was: loss of at most 3 points at 72% or less.  Result: {'PASS' if ok else 'FAIL'}"
              + ("  (small sample: wide interval)" if n < 100 else ""),
              "Not counted: latency of the extra request, and the cost of the answer tokens."]
    return "\n".join(lines)


def main(argv):
    if len(argv) < 2 or not argv[1].isdigit():
        raise SystemExit("usage: python live_cascade_test.py N [simple]")
    n = int(argv[1])
    use_simple = len(argv) > 2 and argv[2] == "simple"
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set in this terminal. Run setx, then open a NEW terminal.")
    if os.path.exists(CACHE_PATH):
        cache.update(json.load(open(CACHE_PATH, encoding="utf-8")))
    questions = load_squad_questions(n)
    print(f"{len(questions)} fresh SQuAD questions (none used in E18 or E26).", flush=True)

    layer = FCCL(min_tokens=0, seed=0)                    # defaults: min_keep 0.6, keyword context rule
    backend = SimpleCDM() if use_simple else LLMLinguaCDM()
    backend_name = "SimpleCDM" if use_simple else "LLMLingua-2"
    rows, stopped = [], False
    try:
        for k, (qid, q, golds, context) in enumerate(questions, 1):
            def ask(text, q=q):
                return chat(ANSWER_MODEL, SQUAD_SYSTEM, f"Passage:\n{text}\n\nQuestion: {q}")

            base_ans = ask(context)
            plan = layer.plan(Request("u1", q, int(len(context.split()) * 1.3)))
            res = answer_with_retry(ask, backend, context, plan, prompt=q)
            first_ans = ask(res.cdm.text)                 # answer on the compressed text alone (cached, free)
            if res.abstained:
                layer.feedback(Feedback("u1", "abstained"))
            rows.append(dict(base_ok=grade(q, golds, base_ans), first_ok=grade(q, golds, first_ans),
                             final_ok=grade(q, golds, res.answer), retried=int(res.retried), cost=res.cost,
                             kept=res.cdm.kept_words, fell=int(res.cdm.fell_back), rate=plan.rate,
                             abstained=int(res.abstained)))
            print(f"  {k}/{len(questions)} questions done", flush=True)
    except RateLimited as e:
        stopped = True
        print(f"\nSTOPPED: {e} Progress is saved; rerun the same command later to continue.")
    text = report(rows, layer, backend_name, len(questions), stopped)
    print("\n" + text)
    open(OUT_PATH, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main(sys.argv)