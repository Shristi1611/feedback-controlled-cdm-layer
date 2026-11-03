"""Question-aware compression on the same 60 MultiWOZ questions as real_multiwoz_llm_test.py.
Keeps the conversation turns most relevant to the user's question until a word budget is reached, in original order.
Scorers: embedding similarity (Ollama nomic-embed-text, local) and BM25 (plain word matching).
Reuses multiwoz_llm_cache.json (run real_multiwoz_llm_test.py 60 first), adds only the new variants.

  python qaware_multiwoz_test.py 6      smoke test (first 6 questions)
  python qaware_multiwoz_test.py 60     all 60 questions (about 240 new calls; resumable)

Needs GROQ_API_KEY and a running Ollama with nomic-embed-text.
"""
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import requests

ANSWER_MODEL = "openai/gpt-oss-20b"
PAUSE = 3.0
URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE = "multiwoz_llm_cache.json"
OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

KEY = os.environ.get("GROQ_API_KEY")
if not KEY:
    raise SystemExit("GROQ_API_KEY is not set in this terminal. Run setx, then open a NEW terminal.")

SYSTEM = ("You will read a conversation between a customer and a booking assistant, then answer a question about it. "
          "Reply with only the requested value (a few words, no explanation). "
          "If the conversation does not say, reply: unknown")

PHRASES = {"area": "area", "food": "type of food", "name": "name", "pricerange": "price range",
           "type": "type of place", "day": "day", "departure": "departure place", "destination": "destination",
           "bookday": "booking day"}
STOP = set("the a an of to for and or in on at is was were what which that this with did do does part".split())


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


# ---------- question-aware selection ----------
def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=120)
    if r.status_code == 404:                                   # older Ollama versions
        out = []
        for t in texts:
            rr = requests.post(f"{OLLAMA}/api/embeddings", json={"model": EMBED_MODEL, "prompt": t}, timeout=120)
            rr.raise_for_status()
            out.append(rr.json()["embedding"])
        return np.array(out)
    r.raise_for_status()
    return np.array(r.json()["embeddings"])


def scores_embed(turns, question):
    vecs = embed(["search_document: " + t for t in turns] + ["search_query: " + question])
    d, q = vecs[:-1], vecs[-1]
    return list((d @ q) / (np.linalg.norm(d, axis=1) * np.linalg.norm(q) + 1e-9))


def scores_bm25(turns, question, k1=1.5, b=0.75):
    docs = [norm(t).split() for t in turns]
    q = [w for w in norm(question).split() if w not in STOP]
    n = len(docs)
    avg = sum(len(d) for d in docs) / max(1, n)
    out = []
    for d in docs:
        s = 0.0
        for w in set(q):
            df = sum(1 for dd in docs if w in dd)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = d.count(w)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / max(1e-9, avg)))
        out.append(s)
    return out


def select(turns, scores, rate):
    total = sum(len(t.split()) for t in turns)
    budget = rate * total
    order = sorted(range(len(turns)), key=lambda i: -scores[i])     # ties keep original order
    chosen, used = [], 0
    for i in order:
        w = len(turns[i].split())
        if used + w <= budget * 1.1 or not chosen:
            chosen.append(i)
            used += w
    return "\n".join(turns[i] for i in sorted(chosen)), used / max(1, total)


limit = int(sys.argv[1]) if len(sys.argv) > 1 else 60

try:
    embed(["test"])
except Exception as e:
    raise SystemExit(f"Could not reach Ollama at {OLLAMA} ({type(e).__name__}). Open the Ollama app, "
                     f"and make sure '{EMBED_MODEL}' is installed (ollama list).")

# ---------- rebuild the same 60 questions as real_multiwoz_llm_test.py ----------
n = 60
from datasets import load_dataset

ds = load_dataset("pfb30/multi_woz_v22", name="default", revision="refs/convert/parquet", split="validation")
per_dialogue, turn_lists, seen = {}, {}, set()
for d in ds:
    did = d["dialogue_id"]
    turns = rows(d["turns"])
    if not turns or did in seen:
        continue
    user_code = turns[0]["speaker"]
    lines_ = [f"{'Customer' if t['speaker'] == user_code else 'Agent'}: {t['utterance'].strip()}" for t in turns]
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
        turn_lists[did] = lines_

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
chosen = chosen[:limit]

if not os.path.exists(CACHE):
    raise SystemExit(f"{CACHE} not found. Run: python real_multiwoz_llm_test.py 60  first.")
cache = json.load(open(CACHE, encoding="utf-8"))


def key(q, method, rate):
    return f"{q[0]}|{q[1]}|{method}|{rate}"


BASE = [("original", 1.0), ("LLMLingua-2", 0.6), ("LLMLingua-2", 0.45), ("SimpleCDM", 0.6)]
NEW = [("Q-aware embeddings", 0.6), ("Q-aware embeddings", 0.45), ("Q-aware BM25", 0.6), ("Q-aware BM25", 0.45)]
missing = [q for q in chosen if any(key(q, m, r) not in cache for m, r in BASE)]
if missing:
    raise SystemExit(f"{len(missing)} questions lack the baseline answers. Run real_multiwoz_llm_test.py 60 to completion first.")

stopped = None
try:
    for k, q in enumerate(chosen, 1):
        did, name, golds, mentions, grp = q
        question = question_for(name)
        turns = turn_lists[did]
        emb_scores = None
        for method, rate in NEW:
            if key(q, method, rate) in cache:
                continue
            if method == "Q-aware embeddings":
                if emb_scores is None:
                    emb_scores = scores_embed(turns, question)
                text, kept = select(turns, emb_scores, rate)
            else:
                text, kept = select(turns, scores_bm25(turns, question), rate)
            ans = chat(SYSTEM, f"Conversation:\n{text}\n\nQuestion: {question}\nAnswer:")
            cache[key(q, method, rate)] = dict(correct=is_correct(ans, golds), noans=norm(ans).startswith("unknown")
                                               or norm(ans) == "", kept=kept, fell=False)
            json.dump(cache, open(CACHE, "w", encoding="utf-8"))
        print(f"  {k}/{len(chosen)} questions done", flush=True)
except RateLimited as e:
    stopped = str(e)

ALL = BASE + NEW
done = [q for q in chosen if all(key(q, m, r) in cache for m, r in ALL)]
if not done:
    raise SystemExit("No question is fully finished yet. " + (stopped or ""))


def sign_test_p(a, b):
    m = a + b
    if m == 0:
        return 1.0
    kk = min(a, b)
    return min(1.0, 2 * sum(math.comb(m, j) for j in range(kk + 1)) / 2 ** m)


def corr(q, m, r):
    return int(cache[key(q, m, r)]["correct"])


lines = []
if stopped:
    lines += [f"STOPPED EARLY: {stopped} Results cover the {len(done)} of {len(chosen)} questions finished. "
              "Rerun the same command later to continue.", ""]
lines += [f"{len(done)} MultiWOZ questions; answerer {ANSWER_MODEL}. Question-aware = keep the turns most relevant to the question.",
          f"{'method':<20}{'keep':>5}{'words kept %':>14}{'correct %':>16}{'change vs original':>22}"]
for m, r in ALL:
    rs = [cache[key(q, m, r)] for q in done]
    p = sum(x["correct"] for x in rs) / len(rs)
    margin = 100 * 1.96 * math.sqrt(p * (1 - p) / len(rs))
    kept = 100 * sum(x["kept"] for x in rs) / len(rs)
    d = [corr(q, m, r) - corr(q, "original", 1.0) for q in done]
    lines.append(f"{m:<20}{r:>5}{kept:>14.0f}{100 * p:>10.1f} +-{margin:<4.1f}"
                 f"{f'{100 * sum(d) / len(d):+.1f} ({d.count(-1)} lost, {d.count(1)} gained)':>22}")

lines += ["", "Direct comparison at the same keep level (question by question):",
          f"{'question-aware method':<22}{'vs':>4}{'keep':>6}  {'aware right, LLMLingua wrong':>30}{'LLMLingua right, aware wrong':>30}{'p-value':>9}"]
for m, r in NEW:
    a = sum(1 for q in done if corr(q, m, r) == 1 and corr(q, "LLMLingua-2", r) == 0)
    b = sum(1 for q in done if corr(q, m, r) == 0 and corr(q, "LLMLingua-2", r) == 1)
    lines.append(f"{m:<22}{'':>4}{r:>6}  {a:>30}{b:>30}{sign_test_p(a, b):>9.3f}")
print("\n" + "\n".join(lines))
with open("qaware_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))