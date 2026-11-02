"""Replication of the retry-on-unknown cascade (E23) on FRESH questions, run live.

PREDICTION (written before running): pooled over both datasets, LLMLingua-2 at keep 0.6 with a retry on 'unknown'
loses no more than 3 accuracy points vs the original, using no more than 72% of the original context tokens.

  python replicate_cascade.py squad 100        fresh SQuAD questions (resumable; stops cleanly at the free limit)
  python replicate_cascade.py multiwoz 100     fresh MultiWOZ conversations (resumable)
  python replicate_cascade.py report           results so far, with the PASS/FAIL check

Excludes every question used in E18/E21/E22. Needs GROQ_API_KEY. The retry reuses the original-context answer
(temperature 0), which is exactly what a live retry would return.
"""
import json
import math
import os
import random
import re
import string
import sys
import time
from collections import defaultdict

import requests

ANSWER_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"
PAUSE = 3.0
URL = "https://api.groq.com/openai/v1/chat/completions"
CACHES = {"squad": "rep_squad_cache.json", "multiwoz": "rep_multiwoz_cache.json"}
VARIANTS = [("original", 1.0), ("LLMLingua-2", 0.8), ("LLMLingua-2", 0.6), ("LLMLingua-2", 0.45)]
NEED = [(m, str(r)) for m, r in VARIANTS]


# ----------------------------------------------------------------- report
def load_complete(path):
    if not os.path.exists(path):
        return {}
    qs = defaultdict(dict)
    for k, v in json.load(open(path, encoding="utf-8")).items():
        parts = k.split("|")
        qs["|".join(parts[:-2])][(parts[-2], parts[-1])] = v
    return {q: v for q, v in qs.items() if all(k in v for k in NEED)}


def sign_p(lost, gained):
    m = lost + gained
    if m == 0:
        return 1.0
    k = min(lost, gained)
    return min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m)


def stats(recs, var):
    o = [int(r[("original", "1.0")]["correct"]) for r in recs]
    c = [int(r[var]["correct"]) for r in recs]
    u = [bool(r[var]["unknown"]) for r in recs]
    kept = [r[var]["kept"] for r in recs]
    casc = [oo if uu else cc for oo, cc, uu in zip(o, c, u)]
    n = len(o)
    dp = [a - b for a, b in zip(c, o)]
    dc = [a - b for a, b in zip(casc, o)]
    return dict(n=n, tok_plain=100 * sum(kept) / n, d_plain=100 * sum(dp) / n, lg_plain=(dp.count(-1), dp.count(1)),
                tok_casc=100 * (sum(kept) + sum(u)) / n, d_casc=100 * sum(dc) / n, lg_casc=(dc.count(-1), dc.count(1)),
                retried=100 * sum(u) / n, conf_wrong=sum(1 for oo, cc, uu in zip(o, c, u) if oo and not cc and not uu),
                dc=dc, kept=kept, u=u)


def report():
    sets = {name: list(load_complete(path).values()) for name, path in CACHES.items()}
    lines = ["Replication of the retry-on-unknown cascade on fresh questions (tokens % = context tokens used vs the original)."]
    pooled = []
    for name, recs in list(sets.items()) + [("pooled", None)]:
        if name == "pooled":
            recs = [r for rs in sets.values() for r in rs]
            pooled = recs
        if not recs:
            continue
        lines += ["", f"=== {name}: {len(recs)} questions ===",
                  f"{'method':<13}{'keep':>5} | {'plain tokens %':>15}{'change (lost/gained)':>24} | {'retried %':>10}"
                  f"{'cascade tokens %':>18}{'change (lost/gained)':>24}{'p':>7} | {'confident wrong':>16}"]
        for m, r in VARIANTS[1:]:
            s = stats(recs, (m, str(r)))
            lines.append(f"{m:<13}{r:>5} | {s['tok_plain']:>15.0f}"
                         f"{f'{s['d_plain']:+.1f} ({s['lg_plain'][0]}/{s['lg_plain'][1]})':>24} | {s['retried']:>10.0f}"
                         f"{s['tok_casc']:>18.0f}{f'{s['d_casc']:+.1f} ({s['lg_casc'][0]}/{s['lg_casc'][1]})':>24}"
                         f"{sign_p(*s['lg_casc']):>7.3f} | {s['conf_wrong']:>16}")
    if pooled:
        s = stats(pooled, ("LLMLingua-2", "0.6"))
        rng = random.Random(0)
        n = s["n"]
        boots = sorted(100 * sum(rng.choice(s["dc"]) for _ in range(n)) / n for _ in range(3000))
        lo, hi = boots[75], boots[2924]
        ok = s["d_casc"] >= -3.0 and s["tok_casc"] <= 72.0
        lines += ["", f"PREDICTION CHECK (pooled, {n} fresh questions): cascade at keep 0.6 changes accuracy by "
                      f"{s['d_casc']:+.1f} points (95% interval [{lo:+.1f}, {hi:+.1f}]) at {s['tok_casc']:.0f}% of the tokens.",
                  f"Prediction was: loss of at most 3 points at 72% of the tokens or less.  Result: {'PASS' if ok else 'FAIL'}."]
    print("\n".join(lines))
    open("replication_results.txt", "w", encoding="utf-8").write("\n".join(lines))


if len(sys.argv) < 2 or sys.argv[1] not in ("squad", "multiwoz", "report"):
    raise SystemExit("usage: python replicate_cascade.py squad|multiwoz N   or   python replicate_cascade.py report")
if sys.argv[1] == "report":
    report()
    raise SystemExit(0)

# ----------------------------------------------------------------- live run
from fccl import CompressionPlan, extract_preserve
from cdm import LLMLinguaCDM, apply_plan

KEY = os.environ.get("GROQ_API_KEY")
if not KEY:
    raise SystemExit("GROQ_API_KEY is not set in this terminal. Run setx, then open a NEW terminal.")


class RateLimited(Exception):
    pass


def chat(model, system, user, max_tokens=300):
    for _ in range(8):
        r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}"}, timeout=90, json={
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
        time.sleep(PAUSE)
        return (r.json()["choices"][0]["message"].get("content") or "").strip()
    raise RateLimited("Too many rate-limit retries.")


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", "", str(s).lower()).split())


def normalize_squad(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


SQUAD_SYSTEM = ("Answer the question using ONLY the passage. Reply with the shortest possible answer "
                "(a few words), with no explanation. If the passage does not contain the answer, reply: unknown")
JUDGE_SYSTEM = "You grade answers. Reply with exactly one word: YES or NO."
MW_SYSTEM = ("You will read a conversation between a customer and a booking assistant, then answer a question about it. "
             "Reply with only the requested value (a few words, no explanation). "
             "If the conversation does not say, reply: unknown")
PHRASES = {"area": "area", "food": "type of food", "name": "name", "pricerange": "price range",
           "type": "type of place", "day": "day", "departure": "departure place", "destination": "destination",
           "bookday": "booking day"}


def squad_ask(question, golds, passage):
    ans = chat(ANSWER_MODEL, SQUAD_SYSTEM, f"Passage:\n{passage}\n\nQuestion: {question}")
    na = normalize_squad(ans)
    unknown = (na == "") or na.startswith("unknown")
    correct = (not unknown) and any(normalize_squad(g) and (normalize_squad(g) in na or na == normalize_squad(g))
                                    for g in golds)
    if not correct and not unknown:
        verdict = chat(JUDGE_MODEL, JUDGE_SYSTEM,
                       f"Question: {question}\nCorrect answers: {golds}\nModel answer: {ans}\n"
                       "Does the model answer give the same information as at least one correct answer?", max_tokens=100)
        correct = verdict.upper().startswith("YES")
    return correct, unknown


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


def mw_question(name):
    dom, _, slot = name.partition("-")
    return (f"For the {dom} part of the conversation, what was the {PHRASES.get(slot.replace(' ', ''), slot)} "
            "that the customer asked for or agreed to?")


def mw_correct(answer, golds):
    a = norm(answer)
    for g in golds:
        g = norm(g)
        if g and (f" {g} " in f" {a} " or (len(g) >= 5 and any(w.startswith(g) for w in a.split()))):
            return True
        if g == "centre" and "center" in a.split():
            return True
    return False


which = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
from datasets import load_dataset

questions = []                                   # (qid, ask_fn(text)->(correct, unknown), context, question, preserve)
if which == "squad":
    ds = load_dataset("rajpurkar/squad", split="validation")
    N = len(ds)
    used = set(random.Random(0).sample(range(N), 60)) | set(random.Random(0).sample(range(N), 5))
    pool = [i for i in range(N) if i not in used]
    for i in random.Random(1).sample(pool, n):
        it = ds[i]
        q, golds = it["question"], it["answers"]["text"]
        questions.append((str(i), (lambda text, q=q, golds=golds: squad_ask(q, golds, text)), it["context"], q,
                          set(extract_preserve(q))))
else:
    prev = "multiwoz_llm_cache.json"
    used_dids = {k.split("|")[0] for k in json.load(open(prev, encoding="utf-8"))} if os.path.exists(prev) else set()
    ds = load_dataset("pfb30/multi_woz_v22", name="default", revision="refs/convert/parquet", split="validation")
    per_dialogue, texts, seen = {}, {}, set()
    for d in ds:
        did = d["dialogue_id"]
        turns = rows(d["turns"])
        if not turns or did in seen or did in used_dids:
            continue
        user_code = turns[0]["speaker"]
        text = "\n".join(f"{'Customer' if t['speaker'] == user_code else 'Agent'}: {t['utterance'].strip()}" for t in turns)
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
    rng = random.Random(1)
    ids = sorted(per_dialogue)
    rng.shuffle(ids)
    n += n % 2
    need = {"1": n // 2, "2+": n // 2}
    chosen = []
    for did in ids:
        opts = per_dialogue[did][:]
        rng.shuffle(opts)
        for name, vals, m in opts:
            g = "1" if m == 1 else "2+"
            if need[g] > 0:
                need[g] -= 1
                chosen.append((did, name, vals))
                break
        if not any(need.values()):
            break
    rng.shuffle(chosen)
    for did, name, golds in chosen:
        qtext = mw_question(name)

        def ask(text, qtext=qtext, golds=golds):
            a = chat(ANSWER_MODEL, MW_SYSTEM, f"Conversation:\n{text}\n\nQuestion: {qtext}\nAnswer:")
            return mw_correct(a, golds), (norm(a) == "" or norm(a).startswith("unknown"))
        questions.append((f"{did}|{name}", ask, texts[did], qtext, {"Customer", "Agent"} | set(extract_preserve(qtext))))

print(f"{len(questions)} fresh {which} questions (none used in earlier experiments).", flush=True)
path = CACHES[which]
cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
backend = LLMLinguaCDM()
try:
    for k, (qid, ask, context, question, preserve) in enumerate(questions, 1):
        for method, rate in VARIANTS:
            key = f"{qid}|{method}|{rate}"
            if key in cache:
                continue
            if method == "original":
                text, kept, fell = context, 1.0, False
            else:
                plan = CompressionPlan(apply=True, rate=rate, preserve=sorted(preserve))
                res = apply_plan(backend, context, plan, prompt=question)
                text, kept, fell = res.text, res.kept_words, res.fell_back
            correct, unknown = ask(text)
            cache[key] = dict(correct=bool(correct), unknown=bool(unknown), kept=kept, fell=fell)
            json.dump(cache, open(path, "w", encoding="utf-8"))
        print(f"  {k}/{len(questions)} questions done", flush=True)
except RateLimited as e:
    print(f"\nSTOPPED: {e} Progress is saved; rerun the same command later to continue.")
print("\nRun:  python replicate_cascade.py report")