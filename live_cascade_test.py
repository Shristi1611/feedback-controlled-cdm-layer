"""Live end-to-end test of the wired retry net (E27, E28).

Runs the REAL code path on fresh questions:
  FCCL.plan -> compressor (LLMLingua-2) -> cdm.answer_with_retry -> live Groq answers -> "abstained" feedback to FCCL

Per question it measures, against the uncompressed answer to the same question:
  - accuracy of the compressed answer WITHOUT the retry, and WITH the retry (what the pipeline really returns)
  - the real cost: words sent as a fraction of the original (a retry adds 100%), and requests per question
It reuses the E26 prediction: with the retry, loss of at most 3 points at 72% of the tokens or less.
(FCCL's own plan for neutral questions is keep 0.65, so expect about 65% of the tokens plus the retries.)

  python live_cascade_test.py squad 100              fresh SQuAD questions (the first 60 are the E27 sample)
  python live_cascade_test.py multiwoz 100           fresh MultiWOZ booking conversations (not used in E22 or E26)
  python live_cascade_test.py squad 100 users=10     spread the questions over 10 simulated users (default 5)
  python live_cascade_test.py squad 100 simple       the free local SimpleCDM instead of LLMLingua-2
  python live_cascade_test.py squad 100 skip=100 strength=0.8
        E29: take questions 101 to 200 of the fixed sample (never used before) and plan with compress_strength 0.8,
        so a neutral question is planned at keep 0.6 instead of 0.65. PRE-REGISTERED prediction, written before any run:
        pooled over the two fresh 100-question sets (SQuAD and MultiWOZ), the retry pipeline loses at most 3 points and
        uses 72% of the context words or less.
  (a bare number, "python live_cascade_test.py 60", still means SQuAD)

Questions are dealt to the users in turn, so each user has their own satisfaction weight and an abstention on
one passage no longer pushes down the weight that every other question uses (E27 sent everything through one user).
Needs GROQ_API_KEY. Resumable: every model answer is cached in live_cascade_cache.json, so rerunning replays
finished questions for free and stops cleanly at the Groq free limit. Writes live_cascade_<dataset>_<answerer>_s<skip>_results.txt.
Other answerers: add answerer=<groq model id> (python live_cascade_test.py models lists what your key can use).
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
MAX_N = 200

# Same prompts as the E26 replication, so the numbers are comparable (they tell the model to say "unknown").
SQUAD_SYSTEM = ("Answer the question using ONLY the passage. Reply with the shortest possible answer "
                "(a few words), with no explanation. If the passage does not contain the answer, reply: unknown")
MW_SYSTEM = ("You will read a conversation between a customer and a booking assistant, then answer a question about it. "
             "Reply with only the requested value (a few words, no explanation). "
             "If the conversation does not say, reply: unknown")
JUDGE_SYSTEM = "You grade answers. Reply with exactly one word: YES or NO."
PHRASES = {"area": "area", "food": "type of food", "name": "name", "pricerange": "price range",
           "type": "type of place", "day": "day", "departure": "departure place", "destination": "destination",
           "bookday": "booking day"}


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
    body = {"model": model, "temperature": 0, "max_completion_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"                  # only reasoning models accept this field
    elif "qwen3" in model:
        body["reasoning_effort"] = "none"
    for attempt in range(8):
        r = requests.post(URL, headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}, timeout=90, json=body)
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", 10))
            if wait > 180:
                raise RateLimited(f"Groq asks for a {wait / 60:.0f} minute wait (probably the free daily token limit).")
            time.sleep(wait + 1)
            continue
        if r.status_code in (500, 502, 503, 504):         # Groq overloaded ("over capacity"): back off and retry
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        if r.status_code != 200:
            raise SystemExit(f"Groq error {r.status_code}: {r.text[:400]}")
        out = (r.json()["choices"][0]["message"].get("content") or "").strip()
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        cache[key] = out
        save_cache()
        time.sleep(PAUSE)
        return out
    raise RateLimited("Too many retries (rate limit or Groq overload).")


# ----------------------------------------------------------------- grading
def normalize_squad(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", "", str(s).lower()).split())


def is_unknown(ans):
    na = normalize_squad(ans)
    return na == "" or na.startswith("unknown")


def squad_grade(question, golds, ans):
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


def mw_grade(answer, golds):
    """Same rule as E22/E26: exact match on the annotated value (centre/center accepted)."""
    a = norm(answer)
    for g in golds:
        g = norm(g)
        if g and (f" {g} " in f" {a} " or (len(g) >= 5 and any(w.startswith(g) for w in a.split()))):
            return True
        if g == "centre" and "center" in a.split():
            return True
    return False


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


# ----------------------------------------------------------------- questions
# Each item: dict(qid, req_text, context, preserve_extra, ask(text)->answer string, grade(answer)->bool)
def load_squad_items(n):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation")
    N = len(ds)
    used = set(random.Random(0).sample(range(N), 60)) | set(random.Random(0).sample(range(N), 5))
    pool = [i for i in range(N) if i not in used]
    used |= set(random.Random(1).sample(pool, 100))            # the E26 replication sample
    pool2 = [i for i in range(N) if i not in used]
    first = random.Random(2).sample(pool2, 60)                 # the E27 sample, kept as the first 60
    rest_pool = [i for i in pool2 if i not in set(first)]
    chosen = (first + random.Random(3).sample(rest_pool, MAX_N - 60))[:n]
    items = []
    for i in chosen:
        it = ds[i]
        q, golds = it["question"], it["answers"]["text"]
        items.append(dict(
            qid=str(i), req_text=q, context=it["context"], preserve_extra=[],
            ask=lambda text, q=q: chat(ANSWER_MODEL, SQUAD_SYSTEM, f"Passage:\n{text}\n\nQuestion: {q}"),
            grade=lambda ans, q=q, golds=golds: squad_grade(q, golds, ans)))
    return items


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


def load_multiwoz_items(n):
    from datasets import load_dataset
    used_dids = set()
    for path in ("multiwoz_llm_cache.json", "rep_multiwoz_cache.json"):    # E22 and E26 conversations
        if os.path.exists(path):
            used_dids |= {k.split("|")[0] for k in json.load(open(path, encoding="utf-8"))}
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
    rng = random.Random(2)
    ids = sorted(per_dialogue)
    rng.shuffle(ids)
    need = {"1": MAX_N // 2, "2+": MAX_N // 2}
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
    items = []
    for did, name, golds in chosen[:n]:
        qtext = mw_question(name)
        items.append(dict(
            qid=f"{did}|{name}", req_text=qtext, context=texts[did], preserve_extra=["Customer", "Agent"],
            ask=lambda text, qtext=qtext: chat(ANSWER_MODEL, MW_SYSTEM,
                                               f"Conversation:\n{text}\n\nQuestion: {qtext}\nAnswer:"),
            grade=lambda ans, golds=golds: mw_grade(ans, golds)))
    return items


# ----------------------------------------------------------------- report
def report(rows_, layer, which, backend_name, n_users, total, stopped, strength=0.7, skip=0):
    n = len(rows_)
    lines = [f"Live end-to-end test of the wired retry net ({which}, questions {skip + 1} to {skip + total}, compress_strength {strength}). Backend: {backend_name}; {n_users} simulated users. "
             f"{n} of {total} fresh questions finished" + (" (stopped early by the rate limit)." if stopped else ".")]
    if n == 0:
        lines.append("No question finished.")
        return "\n".join(lines)
    base = [int(r["base_ok"]) for r in rows_]
    plain = [int(r["first_ok"]) for r in rows_]
    final = [int(r["final_ok"]) for r in rows_]
    d_plain = [a - b for a, b in zip(plain, base)]
    d_final = [a - b for a, b in zip(final, base)]
    cost = 100 * sum(r["cost"] for r in rows_) / n
    plain_cost = 100 * sum(r["kept"] for r in rows_) / n
    retried = sum(r["retried"] for r in rows_)
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
        f"(uncompressed: 1.00). The guard fell back (no compression) on {sum(r['fell'] for r in rows_)} questions.",
        f"FCCL's chosen keep: mean {sum(r['rate'] for r in rows_) / n:.2f}, range "
        f"{min(r['rate'] for r in rows_):.2f} to {max(r['rate'] for r in rows_):.2f}; measured words kept (before any retry) "
        f"{plain_cost:.0f}%.",
    ]
    ws = sorted(layer.sat_weight.get(f"u{u}", 1.0) for u in range(n_users))
    lines.append(f"Abstained feedback sent: {sum(r['abstained'] for r in rows_)}; the users' final satisfaction weights "
                 f"range {ws[0]:.2f} to {ws[-1]:.2f} (median {ws[len(ws) // 2]:.2f}).")
    lo, hi = boot_ci(d_final)
    loss = 100 * sum(d_final) / n
    ok = loss >= -3.0 and cost <= 72.0
    lines += ["",
              f"PREDICTION CHECK (E26 prediction, this run): with the retry, accuracy changes by {loss:+.1f} points "
              f"(95% interval [{lo:+.1f}, {hi:+.1f}]) at {cost:.0f}% of the context words. "
              f"Prediction was: loss of at most 3 points at 72% or less.  Result: {'PASS' if ok else 'FAIL'}"
              + ("  (small sample: wide interval)" if n < 100 else ""),
              f"One question is worth {100 / n:.1f} points at n={n}.",
              "Not counted: latency of the extra request, and the cost of the answer tokens."]
    return "\n".join(lines)


def main(argv):
    global ANSWER_MODEL
    args = argv[1:]
    if args == ["models"]:
        if not os.environ.get("GROQ_API_KEY"):
            raise SystemExit("GROQ_API_KEY is not set in this terminal.")
        r = requests.get("https://api.groq.com/openai/v1/models", timeout=30,
                         headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
        for m in sorted(x["id"] for x in r.json().get("data", [])):
            print(m)
        return
    which, n, n_users, use_simple, strength, skip = "squad", None, 5, False, 0.7, 0
    for a in args:
        if a in ("squad", "multiwoz"):
            which = a
        elif a.isdigit():
            n = int(a)
        elif a == "simple":
            use_simple = True
        elif a.startswith("answerer="):
            ANSWER_MODEL = a[9:]
        elif a.startswith("users=") and a[6:].isdigit():
            n_users = max(1, int(a[6:]))
        elif a.startswith("skip=") and a[5:].isdigit():
            skip = int(a[5:])
        elif a.startswith("strength="):
            try:
                strength = float(a[9:])
            except ValueError:
                raise SystemExit("strength= needs a number, for example strength=0.8")
        else:
            raise SystemExit("usage: python live_cascade_test.py [squad|multiwoz] N [users=K] [skip=S] [strength=X] [answerer=ID] [simple]")
    if n is None or n < 1 or skip + n > MAX_N:
        raise SystemExit(f"usage: python live_cascade_test.py [squad|multiwoz] N [users=K] [skip=S] [strength=X] [answerer=ID] [simple]   (skip + N must be at most {MAX_N})")
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set in this terminal. Run setx, then open a NEW terminal.")
    if os.path.exists(CACHE_PATH):
        cache.update(json.load(open(CACHE_PATH, encoding="utf-8")))
    items = (load_squad_items(skip + n) if which == "squad" else load_multiwoz_items(skip + n))[skip:]
    print(f"{len(items)} fresh {which} questions (numbers {skip + 1} to {skip + len(items)}), compress_strength {strength}, "
          f"dealt to {n_users} simulated users.", flush=True)

    layer = FCCL(min_tokens=0, seed=0, compress_strength=strength)   # defaults: min_keep 0.6, keyword context rule
    backend = SimpleCDM() if use_simple else LLMLinguaCDM()
    backend_name = ("SimpleCDM" if use_simple else "LLMLingua-2") + f", answerer {ANSWER_MODEL}"
    slug = re.sub(r"[^a-z0-9]+", "-", ANSWER_MODEL.lower()).strip("-")
    out_rows, stopped = [], False
    try:
        for k, it in enumerate(items, 1):
            uid = f"u{(k - 1) % n_users}"
            base_ans = it["ask"](it["context"])
            plan = layer.plan(Request(uid, it["req_text"], int(len(it["context"].split()) * 1.3)))
            if it["preserve_extra"]:
                plan.preserve = sorted(set(plan.preserve) | set(it["preserve_extra"]))
            res = answer_with_retry(it["ask"], backend, it["context"], plan, prompt=it["req_text"])
            first_ans = it["ask"](res.cdm.text)           # answer on the compressed text alone (cached, free)
            if res.abstained:
                layer.feedback(Feedback(uid, "abstained"))
            out_rows.append(dict(base_ok=it["grade"](base_ans), first_ok=it["grade"](first_ans),
                                 final_ok=it["grade"](res.answer), retried=int(res.retried), cost=res.cost,
                                 kept=res.cdm.kept_words, fell=int(res.cdm.fell_back), rate=plan.rate,
                                 abstained=int(res.abstained)))
            print(f"  {k}/{len(items)} questions done", flush=True)
    except RateLimited as e:
        stopped = True
        print(f"\nSTOPPED: {e} Progress is saved; rerun the same command later to continue.")
    text = report(out_rows, layer, which, backend_name, n_users, len(items), stopped, strength, skip)
    print("\n" + text)
    open(f"live_cascade_{which}_{slug}_s{skip}_results.txt", "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main(sys.argv)