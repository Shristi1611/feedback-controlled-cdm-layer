"""Look at the real structure of MultiWOZ 2.2 and check that we can build questions from its annotations.
Downloads the public validation split (Apache-2.0).

  python inspect_multiwoz.py
"""
import re
import statistics
from collections import Counter

from datasets import load_dataset

ds = None
for repo in ("pfb30/multi_woz_v22", "multi_woz_v22"):
    for name in ("default", "v2.2", None):
        try:
            kw = dict(path=repo, revision="refs/convert/parquet", split="validation")
            if name:
                kw["name"] = name
            ds = load_dataset(**kw)
            print(f"loaded: {repo}, config {name}")
            break
        except Exception as e:
            print(f"failed ({repo}, {name}): {type(e).__name__}: {str(e)[:200]}")
    if ds is not None:
        break
if ds is None:
    raise SystemExit("Could not load MultiWOZ. Paste me the messages above.")

print("\ncolumns:", ds.column_names, "| dialogues:", len(ds))


def rows(x):
    """Hugging Face stores nested sequences as dict-of-lists; turn that into a list of dicts."""
    if isinstance(x, dict):
        keys = list(x.keys())
        n = len(x[keys[0]]) if keys else 0
        return [{k: x[k][i] for k in keys} for i in range(n)]
    return list(x or [])


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", "", s.lower()).split())


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


first = ds[0]
turns0 = rows(first["turns"])
print("\n--- first dialogue ---")
print("dialogue_id:", first["dialogue_id"], "| services:", first["services"], "| turns:", len(turns0))
print("turn keys:", list(turns0[0].keys()))
for t in turns0[:4]:
    print(f"  speaker={t['speaker']}: {t['utterance'][:120]}")
print("final state:", final_state(turns0, turns0[0]["speaker"]))

n_words, n_turns = [], []
cands, mentions, slot_names = [], Counter(), Counter()
for d in ds:
    turns = rows(d["turns"])
    if not turns:
        continue
    user_code = turns[0]["speaker"]
    text = norm(" ".join(t["utterance"] for t in turns))
    n_words.append(len(" ".join(t["utterance"] for t in turns).split()))
    n_turns.append(len(turns))
    for name, vals in final_state(turns, user_code).items():
        good = [v for v in vals if norm(v) and norm(v) not in ("dontcare", "none")]
        found = [v for v in good if f" {norm(v)} " in f" {text} "]
        if found:
            m = f" {text} ".count(f" {norm(found[0])} ")
            cands.append((d["dialogue_id"], name, vals, m))
            mentions[min(m, 3)] += 1
            slot_names[name] += 1

print(f"\ndialogue length in words: min {min(n_words)}, median {statistics.median(n_words)}, "
      f"mean {statistics.mean(n_words):.0f}, max {max(n_words)}")
print(f"turns per dialogue: median {statistics.median(n_turns)}, max {max(n_turns)}")
print(f"usable question candidates (answer appears in the text): {len(cands)} "
      f"from {len({c[0] for c in cands})} dialogues")
print("how often the answer is mentioned in the dialogue (1, 2, 3+ times):",
      [mentions[1], mentions[2], mentions[3]])
print("most common slots:", slot_names.most_common(10))
print("\nsample candidates (dialogue, slot, accepted answers, mentions):")
for c in cands[:6]:
    print("  ", c)