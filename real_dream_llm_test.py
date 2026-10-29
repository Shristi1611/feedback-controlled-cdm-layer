"""Real DREAM dialogue questions (multiple choice), answered by a generative LLM (Groq free tier),
with the dialogue compressed to different levels. Resumes if interrupted (dream_llm_cache.json).
If the free rate limit forces a long wait, it stops cleanly and reports on the questions finished so far.

  python real_dream_llm_test.py 5      smoke test (a few minutes)
  python real_dream_llm_test.py 100    the real run (roughly 40 minutes if no limit is hit)

Needs GROQ_API_KEY in the environment. DREAM is public research data.
"""
import json
import math
import os
import random
import re
import sys
import time

import requests

from fccl import CompressionPlan, extract_preserve
from cdm import LLMLinguaCDM, SimpleCDM, apply_plan

ANSWER_MODEL = "openai/gpt-oss-20b"
PAUSE = 3.0                  # seconds between calls, to stay under the rate limits
MIN_WORDS = 100              # only dialogues at least this long, so there is something to compress
URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE = "dream_llm_cache.json"

KEY = os.environ.get("GROQ_API_KEY")
if not KEY:
    raise SystemExit("GROQ_API_KEY is not set in this terminal. Run setx, then open a NEW terminal.")

SYSTEM = ("Read the conversation and answer the multiple-choice question about it. "
          "Reply with only the letter of the best answer: A, B or C.")


class RateLimited(Exception):
    pass


def chat(system, user, max_tokens=300):
    for _ in range(8):
        r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}"}, timeout=90, json={
            "model": ANSWER_MODEL, "temperature": 0, "max_completion_tokens": max_tokens,
            "reasoning_effort": "low",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", 10))
            if wait > 180:
                raise RateLimited(f"Groq asks for a {wait / 60:.0f} minute wait (probably the free daily token limit).")
            time.sleep(wait + 1)
            continue
        if r.status_code != 200:
            raise SystemExit(f"Groq error {r.status_code}: {r.text[:400]}")
        time.sleep(PAUSE)
        return (r.json()["choices"][0]["message"].get("content") or "").strip()
    raise RateLimited("Too many rate-limit retries.")


def format_dialogue(turns):
    """'M: hi' -> 'Man: hi', 'W: hi' -> 'Woman: hi'. Returns the text and the speaker labels to protect."""
    out, labels = [], set()
    for t in turns:
        m = re.match(r"^\s*([MW])(\d?)\s*:\s*(.*)$", t, re.S)
        if m:
            name = ("Man" if m.group(1) == "M" else "Woman") + m.group(2)
            labels.add(name)
            out.append(f"{name}: {m.group(3).strip()}")
        else:
            out.append(t.strip())
    return "\n".join(out), labels


def ask(text, question, shown):
    options = "\n".join(f"{'ABC'[k]}. {c}" for k, c in enumerate(shown))
    resp = chat(SYSTEM, f"Conversation:\n{text}\n\nQuestion: {question}\n{options}\nAnswer:")
    m = re.search(r"\b([ABC])\b", resp.upper())
    return m.group(1) if m else None


n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
from datasets import load_dataset

ds = load_dataset("dataset-org/dream", name="default", revision="refs/convert/parquet", split="validation")
eligible = [i for i in range(len(ds)) if len(" ".join(ds[i]["dialogue"]).split()) >= MIN_WORDS]
idxs = random.Random(0).sample(eligible, n)
print(f"{len(eligible)} questions have a dialogue of at least {MIN_WORDS} words; using {n}.", flush=True)

VARIANTS = [("no dialogue", 0.0), ("original", 1.0), ("LLMLingua-2", 0.8), ("LLMLingua-2", 0.6),
            ("LLMLingua-2", 0.45), ("LLMLingua-2", 0.3), ("SimpleCDM", 0.6), ("SimpleCDM", 0.45)]
backends = {"LLMLingua-2": LLMLinguaCDM(), "SimpleCDM": SimpleCDM()}
cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
stopped = None

try:
    for k, i in enumerate(idxs, 1):
        it = ds[i]
        question, choices, answer = it["question"], it["choice"], it["answer"]
        order = [0, 1, 2]
        random.Random(i).shuffle(order)                  # shuffle choices so letter position carries no signal
        shown = [choices[j] for j in order]
        gold = "ABC"[order.index(choices.index(answer))]
        dialogue, labels = format_dialogue(it["dialogue"])
        for method, rate in VARIANTS:
            key = f"{i}|{method}|{rate}"
            if key in cache:
                continue
            if method == "no dialogue":
                text, kept, fell = "(no conversation provided)", 0.0, False
            elif method == "original":
                text, kept, fell = dialogue, 1.0, False
            else:
                plan = CompressionPlan(apply=True, rate=rate,
                                       preserve=sorted(labels | set(extract_preserve(question))))
                res = apply_plan(backends[method], dialogue, plan, prompt=question)
                text, kept, fell = res.text, res.kept_words, res.fell_back
            pred = ask(text, question, shown)
            cache[key] = dict(correct=pred == gold, noans=pred is None, kept=kept, fell=fell)
            json.dump(cache, open(CACHE, "w", encoding="utf-8"))
        print(f"  {k}/{n} questions done", flush=True)
except RateLimited as e:
    stopped = str(e)

done = [i for i in idxs if all(f"{i}|{m}|{r}" in cache for m, r in VARIANTS)]
if not done:
    raise SystemExit("No question is fully finished yet. " + (stopped or ""))


def sign_test_p(lost, gained):
    m = lost + gained
    if m == 0:
        return 1.0
    kk = min(lost, gained)
    return min(1.0, 2 * sum(math.comb(m, j) for j in range(kk + 1)) / 2 ** m)


lines = []
if stopped:
    lines += [f"STOPPED EARLY: {stopped} Results below cover the {len(done)} of {n} questions finished "
              "(the first ones of a fixed random sample, so still a fair sample). Rerun the same command later to continue.", ""]
lines += [f"{len(done)} real DREAM questions (dialogues of {MIN_WORDS}+ words); answerer {ANSWER_MODEL}. "
          "Chance is about 33%. Correct % has a 95% margin.",
          f"{'method':<13}{'keep':>5}{'words kept %':>14}{'guard fell back %':>19}{'no answer %':>13}{'correct %':>16}"]
for method, rate in VARIANTS:
    rs = [cache[f"{i}|{method}|{rate}"] for i in done]
    p = sum(r["correct"] for r in rs) / len(rs)
    margin = 100 * 1.96 * math.sqrt(p * (1 - p) / len(rs))
    avg = lambda f: 100 * sum(r[f] for r in rs) / len(rs)
    lines.append(f"{method:<13}{rate:>5}{avg('kept'):>14.0f}{avg('fell'):>19.0f}{avg('noans'):>13.0f}"
                 f"{100 * p:>10.1f} +-{margin:<4.1f}")

lines += ["", "Paired against the original on the same questions (difference = variant minus original, in points):",
          f"{'method':<13}{'keep':>5}{'lost':>6}{'gained':>8}{'diff':>8}{'95% interval':>20}{'p-value':>10}"]
rng = random.Random(1)
for method, rate in VARIANTS:
    if method == "original":
        continue
    d = [int(cache[f"{i}|{method}|{rate}"]["correct"]) - int(cache[f"{i}|original|1.0"]["correct"]) for i in done]
    lost, gained = d.count(-1), d.count(1)
    boots = sorted(100 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(3000))
    lo, hi = boots[75], boots[2924]
    lines.append(f"{method:<13}{rate:>5}{lost:>6}{gained:>8}{100 * sum(d) / len(d):>+8.1f}"
                 f"{f'[{lo:+.1f}, {hi:+.1f}]':>20}{sign_test_p(lost, gained):>10.3f}")
print("\n" + "\n".join(lines))
with open("dream_llm_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))