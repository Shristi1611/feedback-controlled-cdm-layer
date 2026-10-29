"""Real-data test: how much does compression hurt question answering on real passages (SQuAD)?

For each real question: compress its passage (LLMLingua-2 and SimpleCDM, with FCCL's protection rule and the
guard), let a small QA model answer from the compressed passage, and score against the human answer.
Free and local. First run downloads a ~260 MB QA model (the dataset is already cached).

  python real_squad_test.py 20     quick smoke test (a few minutes)
  python real_squad_test.py 150    the real run (roughly 15 to 25 minutes on a laptop CPU)
"""
import math
import random
import re
import string
import sys
from collections import Counter

import torch

from fccl import CompressionPlan, extract_preserve
from cdm import LLMLinguaCDM, SimpleCDM, apply_plan


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def em_score(pred, gold):
    return float(normalize(pred) == normalize(gold))


def best(metric, pred, golds):
    return max(metric(pred, g) for g in golds)


def mean_ci(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))
    return m, 1.96 * sd / math.sqrt(len(xs))


n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

ds = load_dataset("rajpurkar/squad", split="validation")
items = [ds[i] for i in random.Random(0).sample(range(len(ds)), n)]

QA_MODEL = "distilbert-base-cased-distilled-squad"
tok = AutoTokenizer.from_pretrained(QA_MODEL)
model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL).eval()


def ask(question, context, max_answer_tokens=30, top=20):
    """Extractive QA: pick the best answer span in the context (long contexts are split into windows)."""
    if not context.strip():
        return ""
    try:
        enc = tok(question, context, truncation="only_second", max_length=384, stride=128,
                  return_overflowing_tokens=True, return_offsets_mapping=True,
                  padding=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping")
        with torch.no_grad():
            out = model(**enc)
        best_score, best_text = -1e9, ""
        for i in range(enc["input_ids"].shape[0]):
            seq_ids = enc.sequence_ids(i)
            s_logits, e_logits = out.start_logits[i], out.end_logits[i]
            starts = torch.topk(s_logits, top).indices.tolist()
            ends = torch.topk(e_logits, top).indices.tolist()
            for s in starts:
                for e in ends:
                    if seq_ids[s] != 1 or seq_ids[e] != 1 or e < s or e - s + 1 > max_answer_tokens:
                        continue
                    score = float(s_logits[s] + e_logits[e])
                    if score > best_score:
                        best_score = score
                        best_text = context[int(offsets[i][s][0]):int(offsets[i][e][1])]
        return best_text
    except Exception:
        return ""


RATES = (0.8, 0.6, 0.45, 0.3)
backends = {"LLMLingua-2": LLMLinguaCDM(), "SimpleCDM": SimpleCDM()}
rows = {"original": []}
for name in backends:
    for r in RATES:
        rows[(name, r)] = []

for k, it in enumerate(items, 1):
    context, question, golds = it["context"], it["question"], it["answers"]["text"]
    pred = ask(question, context)
    rows["original"].append(dict(f1=best(f1_score, pred, golds), em=best(em_score, pred, golds),
                                 kept=1.0, fell=0.0, survived=1.0))
    for name, cdm in backends.items():
        for r in RATES:
            plan = CompressionPlan(apply=True, rate=r, preserve=extract_preserve(question))
            res = apply_plan(cdm, context, plan, prompt=question)
            pred = ask(question, res.text)
            rows[(name, r)].append(dict(
                f1=best(f1_score, pred, golds), em=best(em_score, pred, golds), kept=res.kept_words,
                fell=float(res.fell_back),
                survived=float(any(g.lower() in res.text.lower() for g in golds))))
    if k % 10 == 0:
        print(f"  {k}/{n} questions done", flush=True)

lines = [f"{n} real SQuAD questions. F1 and EM are percentages with a 95% margin.",
         f"{'method':<13}{'keep':>5}{'words kept %':>14}{'guard fell back %':>19}{'answer survives %':>19}{'F1 %':>14}{'EM %':>14}"]
for key, rs in rows.items():
    name, rate = ("original", 1.0) if key == "original" else key
    f1m, f1c = mean_ci([100 * r["f1"] for r in rs])
    emm, emc = mean_ci([100 * r["em"] for r in rs])
    avg = lambda k: 100 * sum(r[k] for r in rs) / len(rs)
    lines.append(f"{name:<13}{rate:>5}{avg('kept'):>14.0f}{avg('fell'):>19.0f}{avg('survived'):>19.0f}"
                 f"{f1m:>8.1f}+-{f1c:<4.1f}{emm:>8.1f}+-{emc:<4.1f}")
print("\n" + "\n".join(lines))
with open("squad_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))