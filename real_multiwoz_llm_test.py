"""Real MultiWOZ 2.2 booking conversations (logistics-style, with natural filler and repetition).
Question: what value did the customer ask for? (e.g. restaurant area). Answers come from human annotations.
The conversation is compressed to different levels and answered by a generative LLM (Groq free tier).
Questions with numeric answers are excluded (FCCL's guard protects number-like tokens by design).
Resumes if interrupted (multiwoz_llm_cache.json); stops cleanly at a long rate limit and reports on finished questions.

  python real_multiwoz_llm_test.py 6      smoke test
  python real_multiwoz_llm_test.py 60     the real run (about 30 questions per day on the free tier if limits bite)

Needs GROQ_API_KEY in the environment. MultiWOZ is public research data (Apache-2.0).
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
PAUSE = 3.0
URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE = "multiwoz_llm_cache.json"

KEY = os.environ.get("GROQ_API_KEY")
if not KEY:
    raise SystemExit("GROQ_API_KEY is not set in this terminal. Run setx, then open a NEW terminal.")

SYSTEM = ("You will read a conversation between a customer and a booking assistant, then answer a question about it. "
          "Reply with only the requested value (a few words, no explanation). "
          "If the conversation does not say, reply: unknown")

PHRASES = {"area": "area", "food": "type of food", "name": "name", "pricerange": "price range",
           "type": "type of place", "day": "day", "departure": "departure place", "destination": "destination",
           "bookday": "booking day"}


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


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", "", str(s).lower()).split())


def rows(x):
    """Hugging Face stores nested sequences as dict-of-lists; turn that into a list of dicts."""
    if isinstance(x, dict):
        keys = list(x.keys())
        n = len(x[keys[0]]) if keys else 0
        return [{k: x[k][i] for k in keys} for i in range(n)]
    return list(x or [])


def final_state(turns, user_code):
    state = {}
    for t in turns:
        if t["speaker"] != user_code:
            continue
        for fr in rows(t.get("frames")):
            st = fr.get("state") or {}
            for sv in rows(st.get("slots_values")):
                vals = sv.get("slots_values_list")
                if vals:
                    state[sv["slots_values_name"]] = list(vals)
    return state


def question_for(name):
    dom, _, slot = name.partition("-")
    phrase = PHRASES.get(slot.replace(" ", ""), slot)
    return f"For the {dom} part of the conversation, what was the {phrase} that the customer asked for or agreed to?"


def is_correct(answer, golds):
    a = norm(answer)
    for g in golds:
        g = norm(g)
        if not g:
            continue
        if f" {g} " in f" {a} " or (len(g) >= 5 and any(w.startswith(g) for w in a.split())):
            return True
        if g == "centre" and "center" in a.split():
            return True
    return False


n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
n += n % 2                                            # keep the two groups equal
from datasets import load_dataset

ds = load_dataset("pfb30/multi_woz_v22", name="default", revision="refs/convert/parquet", split="validation")
per_dialogue, texts, seen = {}, {}, set()
for d in ds:
    did = d["dialogue_id"]
    turns = rows(d["turns"])
    if not turns or did in seen:
        continue
    user_code = turns[0]["speaker"]
    lines_ = [f"{'Customer' if t['speaker'] == user_code else 'Agent'}: {t['utterance'].strip()}" for t in turns]
    text = "\n".join(lines_)
    ntext = " " + norm(" ".join(t["utterance"] for t in turns)) + " "
    opts = []
    for name, vals in final_state(turns, user_code).items():
        good = [v for v in vals if norm(v) and norm(v) not in ("dontcare", "none", "yes", "no")
                and not re.search(r"\d", str(v))]
        found = [v for v in good if f" {norm(v)} " in ntext]
        if found:
            opts.append((name, good, ntext.count(f" {norm(found[0])} ")))
    if opts:
        seen.add(did)
        per_dialogue[did] = opts
        texts[did] = text

rng = random.Random(0)
ids = sorted(per_dialogue)
rng.shuffle(ids)
need = {"1 mention": n // 2, "2+ mentions": n // 2}
chosen = []
for did in ids:
    opts = per_dialogue[did][:]
    rng.shuffle(opts)
    for name, vals, m in opts:
        g = "1 mention" if m == 1 else "2+ mentions"
        if need[g] > 0:
            need[g] -= 1
            chosen.append((did, name, vals, m, g))
            break
    if not any(need.values()):
        break
rng.shuffle(chosen)
print(f"{len(per_dialogue)} conversations have usable word-answer questions; using {len(chosen)} "
      f"({n // 2} single-mention, {n // 2} repeated).", flush=True)

VARIANTS = [("no dialogue", 0.0), ("original", 1.0), ("LLMLingua-2", 0.8), ("LLMLingua-2", 0.6),
            ("LLMLingua-2", 0.45), ("SimpleCDM", 0.6)]
backends = {"LLMLingua-2": LLMLinguaCDM(), "SimpleCDM": SimpleCDM()}
cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}


def key(q, method, rate):
    return f"{q[0]}|{q[1]}|{method}|{rate}"


stopped = None
try:
    for k, q in enumerate(chosen, 1):
        did, name, golds, mentions, grp = q
        question = question_for(name)
        dialogue = texts[did]
        for method, rate in VARIANTS:
            if key(q, method, rate) in cache:
                continue
            if method == "no dialogue":
                text, kept, fell = "(no conversation provided)", 0.0, False
            elif method == "original":
                text, kept, fell = dialogue, 1.0, False
            else:
                plan = CompressionPlan(apply=True, rate=rate,
                                       preserve=sorted({"Customer", "Agent"} | set(extract_preserve(question))))
                res = apply_plan(backends[method], dialogue, plan, prompt=question)
                text, kept, fell = res.text, res.kept_words, res.fell_back
            ans = chat(SYSTEM, f"Conversation:\n{text}\n\nQuestion: {question}\nAnswer:")
            cache[key(q, method, rate)] = dict(correct=is_correct(ans, golds), noans=norm(ans).startswith("unknown")
                                               or norm(ans) == "", kept=kept, fell=fell)
            json.dump(cache, open(CACHE, "w", encoding="utf-8"))
        print(f"  {k}/{len(chosen)} questions done", flush=True)
except RateLimited as e:
    stopped = str(e)

done = [q for q in chosen if all(key(q, m, r) in cache for m, r in VARIANTS)]
if not done:
    raise SystemExit("No question is fully finished yet. " + (stopped or ""))


def sign_test_p(lost, gained):
    m = lost + gained
    if m == 0:
        return 1.0
    kk = min(lost, gained)
    return min(1.0, 2 * sum(math.comb(m, j) for j in range(kk + 1)) / 2 ** m)


def diffs(qs, method, rate):
    return [int(cache[key(q, method, rate)]["correct"]) - int(cache[key(q, "original", 1.0)]["correct"]) for q in qs]


lines = []
if stopped:
    lines += [f"STOPPED EARLY: {stopped} Results cover the {len(done)} of {len(chosen)} questions finished "
              "(a random mix of both groups). Rerun the same command later to continue.", ""]
g1 = [q for q in done if q[4] == "1 mention"]
g2 = [q for q in done if q[4] == "2+ mentions"]
lines += [f"{len(done)} real MultiWOZ questions ({len(g1)} single-mention answers, {len(g2)} repeated); "
          f"answerer {ANSWER_MODEL}. Word answers only. Correct % has a 95% margin.",
          f"{'method':<13}{'keep':>5}{'words kept %':>14}{'guard fell back %':>19}{'unknown %':>11}{'correct %':>16}"]
for method, rate in VARIANTS:
    rs = [cache[key(q, method, rate)] for q in done]
    p = sum(r["correct"] for r in rs) / len(rs)
    margin = 100 * 1.96 * math.sqrt(p * (1 - p) / len(rs))
    avg = lambda f: 100 * sum(r[f] for r in rs) / len(rs)
    lines.append(f"{method:<13}{rate:>5}{avg('kept'):>14.0f}{avg('fell'):>19.0f}{avg('noans'):>11.0f}"
                 f"{100 * p:>10.1f} +-{margin:<4.1f}")

lines += ["", "Paired change vs the original (points; lost/gained = questions the compression broke/fixed).",
          "Does repetition protect? Compare 'answer stated once' with 'answer stated 2+ times':"]
for method, rate in VARIANTS:
    if method in ("original", "no dialogue"):
        continue
    parts = []
    for label, qs in (("all", done), ("once", g1), ("2+ times", g2)):
        d = diffs(qs, method, rate)
        if not d:
            continue
        lost, gained = d.count(-1), d.count(1)
        extra = f", p={sign_test_p(lost, gained):.3f}" if label == "all" else ""
        parts.append(f"{label} (n={len(d)}): {100 * sum(d) / len(d):+.1f} ({lost}/{gained}{extra})")
    lines.append(f"{method:<12}{rate:>5}  " + "   ".join(parts))
orig = {lab: 100 * sum(cache[key(q, 'original', 1.0)]['correct'] for q in qs) / max(1, len(qs))
        for lab, qs in (("once", g1), ("2+ times", g2))}
lines.append(f"original accuracy: answer stated once {orig['once']:.0f}%, stated 2+ times {orig['2+ times']:.0f}%")
print("\n" + "\n".join(lines))
with open("multiwoz_llm_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))