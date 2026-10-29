# FCCL results log

A running record of every experiment: what was tested, what came out, and what it does and doesn't show.
Add new entries at the bottom of "Experiments" and update the other sections as needed. Log started 2026-10-29.

**Standing caveats (apply to everything below):**
- The learning experiments (E1 to E4) use *simulated* users that I invented. They show the mechanism works, not how real people react.
- The detector and compressor experiments (E5 onward) use a handful of hand-written texts. They are small samples.
- I wrote most of the test texts and all the labels. Nothing here has been checked by an independent person.

---

## Experiments

### E0. Earlier study: answer length (simulation)
- **Question:** can a policy choose answer length per user, saving tokens without more dissatisfaction?
- **Setup:** 5 answer-length levels (40, 100, 200, 380, 600 tokens; default level 3), simulated users with hidden preferences, weak feedback ("show more", "too long", occasional ratings). Policies: pooled, personalized, safe (risk-constrained). 10 paired seeds, token price 0.5.
- **Result (button on):** personalized saved 21.6% of tokens, satisfaction +0.18, dissatisfied turns +0.3 pp; safe saved 14.9%, +0.17, +0.0 pp; pooled saved 19.7%, +0.04, +1.2 pp.
- **Result (weak feedback):** without the show-more button or with sparse feedback, pooled and personalized lost satisfaction or added 2.5 to 7 pp dissatisfaction, while safe stayed within 0.3 pp of the default.
- **Takeaway:** the risk constraint keeps users safe when feedback is thin, at the cost of smaller savings. Its ideas (per-user beliefs, safe exploration) carry into FCCL.
- **Files:** `sim.py`, `policies.py`, `experiments.py`.

### E1. FCCL ablation: do the two weights help? (simulation)
- **Setup:** 100 fake users, 40 turns, 5 seeds. Hidden amount of context each user needs; "bad reaction" = a correction. The context rule holds for 60% of users. Keyword detector.
- **Result:**

| policy | tokens saved | bad reactions, first 10 turns | bad reactions, last 20 turns |
|---|---|---|---|
| no compression | 0.0% | 0.0% | 0.0% |
| fixed keep 0.7 | 30.0% | 10.6% | 10.1% |
| fixed keep 0.5 | 50.0% | 35.6% | 34.6% |
| fixed keep 0.3 | 70.0% | 66.5% | 66.7% |
| FCCL full | 36.7% | 12.2% | 3.2% |
| FCCL, no context weight | 30.8% | 11.3% | 3.9% |
| FCCL, no learning | 41.9% | 18.0% | 18.4% |
| FCCL, neither | 35.0% | 15.4% | 15.8% |

- **Takeaway:** learning protects users (late bad reactions 18.4% without it, 3.2% with it). The context weight mostly buys savings (about 6 points). FCCL is slightly worse than a fixed setting in the first turns (cost of learning).
- **Caveat:** 60% of fake users follow the rule's pattern, so the context weight's benefit is probably overstated (see E2).

### E2. Stress test: what if the context rule is wrong? (simulation)
- **Setup:** three worlds (pattern holds for 60%, no pattern, pattern backwards for 60%), `ctx_strength` 0.0 / 0.25 / 0.5 / 1.0.
- **Result (tokens saved / bad reactions first 10 turns / last 20 turns):**

| world | strength 0.0 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| pattern holds | 30.8 / 11.3 / 3.9 | 33.2 / 10.6 / 3.7 | 34.7 / 10.4 / 3.4 | 36.7 / 12.2 / 3.2 |
| no pattern | 31.1 / 12.0 / 3.5 | 32.2 / 13.0 / 3.8 | 32.8 / 14.9 / 3.7 | 33.3 / 18.6 / 3.8 |
| pattern backwards | 30.2 / 13.2 / 3.6 | 30.0 / 15.1 / 4.1 | 29.8 / 19.3 / 4.0 | 29.1 / 25.8 / 4.5 |

- **Takeaway:** learning corrects a wrong guess (late bad reactions stay 3.2% to 4.5% everywhere), but a wrong guess is paid for early (25.8% vs 13.2% at full trust when the pattern is backwards). The context weight is an unbalanced bet. A low trust level (0.25) hedges.

### E3. Adaptive trust (simulation)
- **Setup:** an epsilon-greedy bandit chooses among trust levels 0, 0.25, 0.5, 1.0; reward = fraction saved minus 1.0 per bad reaction; ties go to the lower level.
- **Result (tokens saved / bad first 10 / bad last 20):** pattern holds 34.0 / 11.3 / 3.6; no pattern 32.0 / 13.7 / 4.0; pattern backwards 30.0 / 14.9 / 3.9. In the backwards world it mostly picks trust 0.0 (70% of requests); when the pattern holds it leans to 0.5 and 1.0.
- **Takeaway:** it never beats the best fixed setting in every world, but it avoids the full-trust disaster and keeps most of the gain, without being told which world it is in.
- **Caveats:** one shared trust level for all users; the penalty of 1.0 is arbitrary.

### E4. Conversation-type detector: keywords vs Laya
- **Setup:** 20 messages I wrote and labelled (7 chatty, 7 logistics, 6 other), many deliberately avoiding the keyword lists. Laya (421M-parameter local decision model, Apache 2.0) asked: "what is the user mainly doing?" with the labels chatty / logistics / other.
- **Result:** keyword rule 8/20, Laya 18/20.
- **Details:** keywords missed venting without emotion words and practical requests without our words, and read "border" as "order". Laya's misses: "lol that movie was wild" (said other) and "Tell me about the history of Rome" (said chatty).
- **Laya's confidences** were low (0.01 to 0.62) and the model warns they are uncalibrated; they did not cleanly separate right from wrong answers, so `LayaContext` does not gate on them (`min_conf=0.0`).
- **Caveat:** tilted against keywords by design; the sample is small; label accuracy only, not whether the policy is right.
- **Files:** `compare_context.py`, `laya_context.py`.

### E5. ScaleDown on a short text (96 tokens)
- **Result:** HTTP 200, the key works. Asked to remove 50%, the returned text was identical to the input; reported 96 to 70 tokens, but the text did not change. Billed `input_tokens` = 96.
- **Takeaway:** very short inputs are passed through. Matches FCCL's `min_tokens=500`.

### E6. ScaleDown on a longer text (511 tokens, 362 words)
- **Result:**

| asked to remove | tokens reported | words (ours) | protected |
|---|---|---|---|
| 30% | 511 to 473 | 362 to 413 | all kept |
| 50% | 511 to 340 | 362 to 297 | all kept |
| 70% | 511 to 203 | 362 to 176 | `X540` lost |

- **Takeaway:** actual savings are lower than requested; at 30% the output repeats overlapping chunks and has more words than the input. Billing is by input size (511 per call).

### E7. ScaleDown calibration (requested vs actual removal)
- **Setup:** the same 511-token text at 10% to 90% removal, run twice; both runs gave identical numbers.
- **Result:**

| asked to remove | tokens kept (ScaleDown's count) | words kept (ours) | protected items lost |
|---|---|---|---|
| 10% | 110% | 135% | none |
| 20% | 107% | 131% | none |
| 30% | 93% | 114% | none |
| 40% | 78% | 97% | `X540` |
| 50% | 67% | 82% | none |
| 60% | 52% | 65% | `X540` |
| 70% | 40% | 49% | `X540` |
| 80% | 26% | 32% | `4821`, `Park Street`, `X540` |
| 90% | 13% | 15% | `4821`, `Park Street`, `X540` |

- **Takeaways:** below about 50% requested removal the output is longer than the input; real savings start at 50%. It works on whole chunks (keeps the opening and the final request) and drops details without warning; the documented API has no way to protect words. Its token counts disagree with the text it returns, so we measure the text ourselves.
- **Decision:** `ScaleDownCDM` skips the API call when the requested removal is below 50% (`min_removal`), saving credit.
- **Caveat:** one text of about 510 tokens. Longer or different text may behave differently.
- **File:** `calibrate_scaledown.py`.

### E8. End-to-end demo with ScaleDown
- **Setup:** Laya + FCCL + ScaleDown on the 511-token context, three cases.
- **Result:** venting message (plan keep 0.79) skipped, below ScaleDown's useful range; logistics message (plan keep 0.44) compressed but **rejected by the guard** because `X540` was lost, so the original text was used and nothing was saved; tiny context not compressed.
- **Takeaway:** the layer behaved as designed (gentler for venting, harder for logistics, nothing for tiny contexts), but with strict protection ScaleDown gave no safe savings on this text.
- **File:** `demo_pipeline.py`.

### E9. LLMLingua-2 on a short text (86 tokens, CPU)
- **Setup:** `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`, `device_map="cpu"`, with `force_tokens` for 4821, Rahul, Park, Street, X540, PM.
- **Result:** keep 0.7 gave 59 of 86 tokens, keep 0.5 gave 42, keep 0.3 gave 24. All protected words survived at every level. Readable at 0.5, telegraphic but factual at 0.3.
- **Takeaway:** honours the rate closely, protects forced words, free, runs on the CPU, no API.
- **Caveats:** deletes individual words (grammar is lost); query-agnostic (ignores the user's question); one short text.

### E10. End-to-end demo with LLMLingua-2 (511-token context)
- **Result:** venting message (plan keep 0.79) kept 77% of words / 75% of tokens, accepted; logistics message (plan keep 0.44) kept 42% of words / 39% of tokens, accepted with all protected IDs surviving; tiny context not compressed.
- **Takeaway:** first run with real savings that match the plan. The first 300 characters of the logistics output kept the stress paragraph while the question was about the order, which suggests query-agnostic selection can keep irrelevant content.

### E11. Fact survival check (weak test)
- **Setup:** 12 facts from the context, each checked by one keyword; keep rates 0.8, 0.6, 0.44, 0.3; LLMLingua-2 and the local SimpleCDM.
- **Result:** LLMLingua-2 kept 12/12 at 0.8, 0.6 and 0.44; at 0.3 the guard rejected it (so that row is just the original). SimpleCDM also kept 12/12 down to 0.44 and 11/12 at 0.3 (missing: support calls on hold).
- **Takeaway:** the test cannot tell good compression from bad, because both methods favour the keywords I picked (names, numbers, long words). It shows only that key tokens survive.
- **File:** `fact_check.py`.

### E12. Guard diagnosis for LLMLingua-2 at keep 0.3
- **Result:** the guard rejected the raw output because it lost `Where`, a capitalised word from the start of the user's message. Raw words kept: 28%.
- **Takeaway:** `extract_preserve` over-protects sentence-starting words; needless fallbacks result (see "Open bugs").

### E13. Blinded quality test
- **Question:** if an LLM answers questions from the compressed text, how much is lost?
- **Setup:** one 362-word context; 8 shuffled samples (S1 to S8, backend names hidden); I (an LLM) answered six fixed questions from each without knowing which was which, then graded against the original (0 to 1 per question, partial credit). Questions: invoice number; delivery address and date; what is in the package and why it matters; what the user already tried with support; the four requests (where the parcel is, new delivery date, refund of the shipping fee, address change); the landlord, rent and sister situation. The guard was bypassed to see raw output.
- **Result (score out of 6; words kept in brackets):**

| compression | LLMLingua-2 | SimpleCDM | ScaleDown |
|---|---|---|---|
| original (362 words) | 6.0 | | |
| keep 0.6 | 6.0 (209) | 6.0 (217) | |
| keep 0.44 | 5.75 (153) | 5.4 (159) | |
| keep 0.3 | 5.1 (103, guard rejected: lost "Where") | 4.6 (109) | |
| removal 0.5 | | | 4.5 (297) |

- **Takeaways:** about 40% fewer words with no detected loss at keep 0.6 for both local methods; details (the personal story, "called twice", "can't afford") are lost first as the rate drops; LLMLingua-2 edged SimpleCDM at each rate but the margin (0.35 to 0.5) is within noise; ScaleDown kept the most words (82%) yet scored lowest, losing whole paragraphs (the landlord story, the phone calls) and repeating others, though it kept all four requests. The guard rejected the best 0.3 sample for dropping one capitalised word ("Where"), confirming the over-protection bug.
- **Hypothesis (untested):** the content lost first is the personal story, which is what a venting user would need kept; this supports giving venting contexts gentler compression.
- **Caveats:** one text; the answerer and judge are the same model and had the ground truth when grading; the questions reward keeping keywords, which favours word-dropping methods; tests facts, not reasoning; differences under about 0.5 are noise.
- **Files:** `make_blind_samples.py` (outputs `blind_samples.txt` and `blind_key.txt`, not committed).

---

## Decisions and why
- "too_long" is not a negative reaction for compression learning: it is about answer length, not about compression.
- FCCL's `rate` is the fraction *kept*; ScaleDown's is the fraction *removed* (converted inside `ScaleDownCDM`).
- The guard falls back to the original text when protected words are lost or when the output is not shorter. It prefers no savings over lost details.
- ID-like tokens in the context (numbers and codes such as 4821 or X540) are protected, not just words from the user's message.
- `ScaleDownCDM.min_removal = 0.5`, from E7 (below that the output was longer).
- Laya runs as `context_fn`, with the keyword rule kept as the default so the simulations stay reproducible.
- LLMLingua-2 was dropped once (out of memory) and retried after a restart; it needs `device_map="cpu"` with a CPU-only PyTorch.

## Open bugs and weaknesses
- `extract_preserve` protects capitalised sentence-starting words (for example "Where", "Ugh"), causing needless fallbacks (E12, E13).
- The guard checks IDs and words, not meaning; a compressed text can keep every protected word and still lose what the user needs.
- The simulations assume the requested rate is honoured exactly; ScaleDown does not do this.
- One shared trust level for all users in adaptive trust.
- Laya's confidence is uncalibrated and unused.
- All compressor results come from one text; they need at least a second and third text of a different kind.

## Environment notes (Windows laptop, about 6 GB RAM in use at peak)
- Python 3.12 in a `.venv` created from Anaconda; `python` and `pip` resolve to the venv first.
- Loading Laya or LLMLingua-2 failed with out-of-memory errors until the machine was restarted and other apps were closed. Laya needs about 1 GB free on top of `torch`.
- Laya is cached at about 3.1 GB (all checkpoints); the English one loaded is about 0.8 GB. LLMLingua-2 (bert-base) is about 0.7 GB.
- The PowerShell profile error in `powershell -command` is harmless (a blocked profile script).
- Hugging Face warns about symlinks and an unauthenticated token; both are harmless.
- ScaleDown API key is stored as the Windows user environment variable `SCALEDOWN_API_KEY` (never in a file in the repo).

## ScaleDown free-token log (estimates; check the dashboard)
Billing is by input size (about 511 tokens per call on the long text). Calls so far: E5 (96), E6 (3 x 511), E7 (2 runs x 9 x 511), E8 (2 x 511 for the first demo, 1 x 511 after `min_removal` was added), E13 (1 x 511). Estimated total about 12,900 of the 1,000,000 free tokens.

## Entry template
```
### E<n>. <short title>
- **Question:**
- **Setup:**
- **Result:**
- **Takeaway:**
- **Caveats:**
- **Files:**
```


---

## Later experiments (append new entries here)

### E14. Blinded quality test on two more texts
- **Question:** do the E13 findings hold on texts of a different kind?
- **Setup:** text B (technical help request, 278 words) and text C (long casual vent, 289 words); LLMLingua-2 and SimpleCDM at keep 0.6, 0.44, 0.3 (raw output, guard bypassed), plus originals; 14 shuffled samples; same blind method as E13 (answerer and grader are the same model, ground truth available when grading). Six questions per text. Text B: setup and versions; symptom and frequency; query time normal vs failing; the three fixes tried and what each did; when failures happen; the four things asked. Text C: flatmate and landlord threat; Meera (what, when, how found out); presentation and laptop; mum and the Diwali dates; the run; the two questions asked.
- **Result (score out of 6; words kept in brackets):**

| keep | method | text A (E13) | text B | text C | mean |
|---|---|---|---|---|---|
| 1.0 | original | 6.0 | 6.0 | 6.0 | 6.0 |
| 0.6 | LLMLingua-2 | 6.0 (209) | 5.65 (183) | 6.0 (161) | 5.9 |
| 0.6 | SimpleCDM | 6.0 (217) | 5.85 (167) | 6.0 (173) | 5.95 |
| 0.44 | LLMLingua-2 | 5.75 (153) | 5.55 (132) | 5.8 (112) | 5.7 |
| 0.44 | SimpleCDM | 5.4 (159) | 5.35 (122) | 5.4 (127) | 5.4 |
| 0.3 | LLMLingua-2 | 5.1 (103) | 4.1 (87) | 4.5 (70) | 4.6 |
| 0.3 | SimpleCDM | 4.6 (109) | 4.2 (83) | 4.7 (87, guard rejected) | 4.5 |

- **Takeaways:** a knee near keep 0.44: about 40% fewer words is near-lossless on all three texts; the cost is small at 0.44 and about 1.5 points at 0.3. Details lost first are relations (what a fix did, who did what to whom, why) and, for SimpleCDM, dates written like "12th". The two local methods are within noise of each other (differences of 0.5 or less, direction flips between texts).
- **Guard results:** all samples passed except SimpleCDM on text C at 0.3, which lost `12th`, `14th`, `29th`, `31st` (its rule protects only pure digits). LLMLingua-2 kept them because they were passed as forced tokens.
- **Actual kept fraction vs requested (LLMLingua-2):** text B 66% / 47% / 31% for requests of 0.6 / 0.44 / 0.3; text C 56% / 39% / 24%. SimpleCDM tracked the request exactly.
- **Implication (tentative):** FCCL's plan (`keep = 1 - 0.7 * context weight * satisfaction weight`, clamped at 0.1) can go far below the knee for a user whose satisfaction weight has grown. A hard minimum keep (around 0.45) is worth testing.
- **Caveats:** three texts; answerer and judge are one model; questions reward keeping keywords, which flatters a simple word-dropper; reading short telegraphic text lets the judge reconstruct meaning from context; the toy matching LLMLingua-2 suggests the test is not discriminating enough to rank compressors.
- **Files:** `make_blind_samples2.py` (outputs `blind_samples2.txt` and `blind_key2.txt`, not committed).

### Fix log
- 2026-10-29: `extract_preserve` no longer protects sentence-initial capitalised words (commit 43e8fcc). Verified: "Where is order 4821 ... Friday" gives `['4821', 'Friday']`, "Ugh I feel so stressed ..." gives `[]`. The simulation output is unchanged. Trade-off: a name that opens a sentence is not protected.
- SimpleCDM weakness found: only pure digits count as IDs, so "12th" and "29th" are dropped at high compression. The guard catches this.