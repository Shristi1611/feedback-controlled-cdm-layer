"""Build the stimuli for the rating study (E33): for each passage, one summary at each compression level.

Each passage is shown to a participant together with its four summaries (written by the answerer model from the
passage compressed to keep 1.0, 0.8, 0.6 and 0.45 of the words). The participant rates each summary; they are never told
which level made which summary (the form shuffles them).

  python make_study_items.py              25 passages, written to study_items.json
  python make_study_items.py 30 seed=7    30 passages, another seed

Needs GROQ_API_KEY (about 4 requests per passage plus a few spare passages, so about 130 requests) and uses
LLMLingua-2 locally. Resumable through live_cascade_cache.json, like the other scripts.

Selection rule (fixed in advance): take passages of at least 100 words in a seeded random order and drop any whose four summaries
are identical after lower-casing and stripping punctuation (nothing to compare). No other filtering, so the study does not
hand-pick cases where compression looks bad.
"""
import json
import random
import re
import sys

LEVELS = (1.0, 0.8, 0.6, 0.45)
SUMMARY_SYSTEM = "Summarise the passage in two or three sentences. Reply with only the summary."


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", "", s.lower()).split())


def build_items(passages, compress, ask, n_keep, levels=LEVELS):
    """passages: list of (passage_id, text). compress(text, level) -> (compressed_text, kept_fraction).
    ask(text) -> summary. Returns (items, dropped_ids)."""
    items, dropped = [], []
    for pid, text in passages:
        answers = {}
        for lv in levels:
            ctext, kept = compress(text, lv)
            answers[str(lv)] = {"text": ask(ctext).strip(), "kept": round(kept, 2)}
        if len({norm(a["text"]) for a in answers.values()}) == 1:
            dropped.append(pid)                       # four identical summaries: nothing to rate
            continue
        items.append({"id": pid, "passage": text, "answers": answers})
        if len(items) == n_keep:
            break
    return items, dropped


def load_passages(seed, how_many):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation")
    seen, out = set(), []
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    for i in idx:
        c = ds[i]["context"]
        if c in seen or len(c.split()) < 100:
            continue
        seen.add(c)
        out.append((f"p{i}", c))
        if len(out) == how_many:
            break
    return out


def main(argv):
    n, seed = 25, 7
    for a in argv[1:]:
        if a.isdigit():
            n = int(a)
        elif a.startswith("seed=") and a[5:].isdigit():
            seed = int(a[5:])
        else:
            raise SystemExit("usage: python make_study_items.py [N] [seed=S]")
    import json as _json, os
    import live_cascade_test as lct
    from fccl import CompressionPlan
    from cdm import LLMLinguaCDM, apply_plan
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set in this terminal.")
    if os.path.exists(lct.CACHE_PATH):
        lct.cache.update(_json.load(open(lct.CACHE_PATH, encoding="utf-8")))
    backend = LLMLinguaCDM()

    def compress(text, level):
        res = apply_plan(backend, text, CompressionPlan(apply=True, rate=level))
        return res.text, res.kept_words

    def ask(text):
        return lct.chat(lct.ANSWER_MODEL, SUMMARY_SYSTEM, f"Passage:\n{text}", max_tokens=200)

    passages = load_passages(seed, n + 15)           # spare passages in case some are dropped
    items, dropped = build_items(passages, compress, ask, n)
    json.dump({"levels": list(LEVELS), "answerer": lct.ANSWER_MODEL, "items": items}, open("study_items.json", "w", encoding="utf-8"), indent=1)
    print(f"Wrote {len(items)} passages x {len(LEVELS)} summaries to study_items.json (dropped {len(dropped)} with identical summaries).")
    if len(items) < n:
        print(f"Only {len(items)} of {n} passages available; rerun with a larger N or another seed.")


if __name__ == "__main__":
    main(sys.argv)