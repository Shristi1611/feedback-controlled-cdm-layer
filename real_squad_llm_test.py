"""Real SQuAD questions, answered by a generative LLM (Groq free tier), judged by a bigger LLM.

Compares answers from the original passage vs compressed passages. Resumes if interrupted
(progress is saved in squad_llm_cache.json).

  python real_squad_llm_test.py 5      smoke test (a few minutes)
  python real_squad_llm_test.py 60     the real run (roughly 30 minutes)

Needs GROQ_API_KEY in the environment. SQuAD is public text, so nothing private is sent.
"""
import json
import math
import os
import random
import re
import string
import sys
import time

import requests

from fccl import CompressionPlan, extract_preserve
from cdm import LLMLinguaCDM, SimpleCDM, apply_plan

# Model IDs from this account's own model list. Edit if they change.
ANSWER_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"
PAUSE = 3.0                  # seconds between calls, to stay under the rate limits
URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE = "squad_llm_cache.json"

KEY = os.environ.get("GROQ_API_KEY")
if not KEY:
    raise SystemExit("GROQ_API_KEY is not set in this terminal. Run setx, then open a NEW terminal.")


def chat(model, system, user, max_tokens=300):
    for _ in range(8):
        r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}"}, timeout=90, json={
            "model": model, "temperature": 0, "max_completion_tokens": max_tokens,
            "reasoning_effort": "low",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        if r.status_code == 429:                       # rate limited: wait and retry
            time.sleep(float(r.headers.get("retry-after", 10)) + 1)
            continue
        if r.status_code != 200:
            raise SystemExit(f"Groq error {r.status_code}: {r.text[:400]}")
        time.sleep(PAUSE)
        return (r.json()["choices"][0]["message"].get("content") or "").strip()
    raise SystemExit("Too many rate-limit retries. Wait a few minutes and rerun; progress is saved.")


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


ANSWER_SYSTEM = ("Answer the question using ONLY the passage. Reply with the shortest possible answer "
                 "(a few words), with no explanation. If the passage does not contain the answer, reply: unknown")
JUDGE_SYSTEM = "You grade answers. Reply with exactly one word: YES or NO."


def answer_and_grade(question, golds, passage):
    ans = chat(ANSWER_MODEL, ANSWER_SYSTEM, f"Passage:\n{passage}\n\nQuestion: {question}")
    na = normalize(ans)
    unknown = (na == "") or na.startswith("unknown")
    correct = (not unknown) and any(normalize(g) and (normalize(g) in na or na == normalize(g)) for g in golds)
    if not correct and not unknown:                     # ask the judge only when the simple check fails
        verdict = chat(JUDGE_MODEL, JUDGE_SYSTEM,
                       f"Question: {question}\nCorrect answers: {golds}\nModel answer: {ans}\n"
                       "Does the model answer give the same information as at least one correct answer?",
                       max_tokens=100)
        correct = verdict.upper().startswith("YES")
    return correct, unknown


n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
from datasets import load_dataset

ds = load_dataset("rajpurkar/squad", split="validation")
idxs = random.Random(0).sample(range(len(ds)), n)

VARIANTS = [("original", 1.0), ("LLMLingua-2", 0.8), ("LLMLingua-2", 0.6), ("LLMLingua-2", 0.45),
            ("LLMLingua-2", 0.3), ("SimpleCDM", 0.6), ("SimpleCDM", 0.45)]
backends = {"LLMLingua-2": LLMLinguaCDM(), "SimpleCDM": SimpleCDM()}
cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}

for k, i in enumerate(idxs, 1):
    it = ds[i]
    context, question, golds = it["context"], it["question"], it["answers"]["text"]
    for method, rate in VARIANTS:
        key = f"{i}|{method}|{rate}"
        if key in cache:
            continue
        if method == "original":
            text, kept, fell = context, 1.0, False
        else:
            plan = CompressionPlan(apply=True, rate=rate, preserve=extract_preserve(question))
            res = apply_plan(backends[method], context, plan, prompt=question)
            text, kept, fell = res.text, res.kept_words, res.fell_back
        correct, unknown = answer_and_grade(question, golds, text)
        cache[key] = dict(correct=correct, unknown=unknown, kept=kept, fell=fell)
    json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    print(f"  {k}/{n} questions done", flush=True)

lines = [f"{n} real SQuAD questions; answerer {ANSWER_MODEL}, judge {JUDGE_MODEL}. Correct % has a 95% margin.",
         f"{'method':<13}{'keep':>5}{'words kept %':>14}{'guard fell back %':>19}{'says unknown %':>16}{'correct %':>16}"]
for method, rate in VARIANTS:
    rs = [cache[f"{i}|{method}|{rate}"] for i in idxs]
    p = sum(r["correct"] for r in rs) / len(rs)
    margin = 100 * 1.96 * math.sqrt(p * (1 - p) / len(rs))
    avg = lambda f: 100 * sum(r[f] for r in rs) / len(rs)
    lines.append(f"{method:<13}{rate:>5}{avg('kept'):>14.0f}{avg('fell'):>19.0f}{avg('unknown'):>16.0f}"
                 f"{100 * p:>10.1f} +-{margin:<4.1f}")
print("\n" + "\n".join(lines))
with open("squad_llm_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))