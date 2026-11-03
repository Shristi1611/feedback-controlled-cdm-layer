"""Autopsy of MultiWOZ compression failures: was the answer word deleted, and who states the answer?
No API calls: reads the cached answers and recompresses the 60 conversations locally with LLMLingua-2.

  python autopsy_multiwoz.py
"""
import json
import os
import random
import re
from collections import Counter

from fccl import CompressionPlan, extract_preserve
from cdm import LLMLinguaCDM, apply_plan

CACHE = "multiwoz_llm_cache.json"
PHRASES = {"area": "area", "food": "type of food", "name": "name", "pricerange": "price range",
           "type": "type of place", "day": "day", "departure": "departure place", "destination": "destination",
           "bookday": "booking day"}


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


def has(text, golds):
    t = " " + norm(text) + " "
    return any(norm(g) and f" {norm(g)} " in t for g in golds)


from datasets import load_dataset

ds = load_dataset("pfb30/multi_woz_v22", name="default", revision="refs/convert/parquet", split="validation")
per_dialogue, turn_lists, seen = {}, {}, set()
n = 60
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

cache = json.load(open(CACHE, encoding="utf-8"))
cdm = LLMLinguaCDM()


def key(q, method, rate):
    return f"{q[0]}|{q[1]}|{method}|{rate}"


# who states the answer, and how many words each side speaks
where = {}
agent_words = total_words = 0
for q in chosen:
    turns = turn_lists[q[0]]
    in_c = any(t.startswith("Customer:") and has(t, q[2]) for t in turns)
    in_a = any(t.startswith("Agent:") and has(t, q[2]) for t in turns)
    where[q[0]] = "both" if in_c and in_a else "customer only" if in_c else "agent only"
    for t in turns:
        w = len(t.split())
        total_words += w
        agent_words += w if t.startswith("Agent:") else 0

lines = [f"{len(chosen)} MultiWOZ questions. The agent speaks {100 * agent_words / total_words:.0f}% of the words.",
         "Where the answer is stated: " + ", ".join(f"{k} {v}" for k, v in Counter(where.values()).items()), ""]
for rate in (0.6, 0.45):
    gold_kept = 0
    lost = lost_gone = 0
    by_where = {k: [0, 0] for k in ("customer only", "agent only", "both")}   # [orig correct, lost]
    for q in chosen:
        text = "\n".join(turn_lists[q[0]])
        question = question_for(q[1])
        plan = CompressionPlan(apply=True, rate=rate, preserve=sorted({"Customer", "Agent"} | set(extract_preserve(question))))
        res = apply_plan(cdm, text, plan, prompt=question)
        kept = has(res.text, q[2])
        gold_kept += kept
        orig_ok = cache[key(q, "original", 1.0)]["correct"]
        comp_ok = cache[key(q, "LLMLingua-2", rate)]["correct"]
        if orig_ok:
            by_where[where[q[0]]][0] += 1
            if not comp_ok:
                by_where[where[q[0]]][1] += 1
                lost += 1
                lost_gone += (not kept)
    lines.append(f"LLMLingua-2 keep {rate}: the answer word survives in {gold_kept}/{len(chosen)} compressed conversations.")
    lines.append(f"   of {lost} questions it broke: answer word deleted in {lost_gone}, still present but answered wrong in {lost - lost_gone}.")
    lines.append("   questions broken, by where the answer is stated: " +
                 ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in by_where.items()))
print("\n".join(lines))
with open("autopsy_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))