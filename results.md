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
- `FCCL` default `min_keep` is now 0.6 (E18, E21, E22: quality falls sharply below it; E26: the retry net stops working at 0.45). The simulations pin `min_keep` explicitly so their results stay reproducible.
- Retry-on-unknown (`cdm.answer_with_retry`) is the way to run keep 0.6: it replicated on 200 fresh questions (E26). The prompt must tell the model to say "unknown", and the model's abstention is fed back to FCCL as the automatic `"abstained"` feedback (a milder back-off than a user complaint).
- Per-user adaptation: four real-data tests at about 12 turns per user (E19, E20, E24, E25) all favour a strong global default, so FCCL keeps a global default and treats per-user learning as unproven rather than dropped.

## Open bugs and weaknesses
- `extract_preserve` protects capitalised sentence-starting words (for example "Where", "Ugh"), causing needless fallbacks (E12, E13).
- The guard checks IDs and words, not meaning; a compressed text can keep every protected word and still lose what the user needs.
- The simulations assume the requested rate is honoured exactly; ScaleDown does not do this.
- One shared trust level for all users in adaptive trust.
- Laya's confidence is uncalibrated and unused.
- Compression-curve results (E17 to E22) come from one answerer model (gpt-oss-20b) and an LLM judge. The retry net was also checked live on Qwen3.8-27B and ALLaM-7B (E30). Venting-style text is untested.
- The retry net needs a model that says "unknown"; it fails at keep 0.45 and below, and multiple-choice tasks cannot use it.
- Per-user adaptation is untested for long histories (more than about 12 turns), compression-specific preferences, and other populations.

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
- 2026-11-03: `cdm.is_abstention` stripped digits, so bare-number answers counted as "unknown" and triggered needless retries and false "abstained" feedback (found by E27). Digits are now kept; regression case added to `test_cascade_wiring.py`.
- 2026-11-06: `live_cascade_test.py` crashed on a Groq 503 ("over capacity") with Qwen3.8-27B at question 28 of MultiWOZ. It now waits and retries on 500/502/503/504 and stops cleanly (resumable) if the provider stays down. Found by E30; covered by an offline test.


### E15. Hard minimum keep (`min_keep`) in the simulation
- **Question:** how often does FCCL plan below the quality knee found in E13/E14 (about 0.45 keep), and what does a floor cost?
- **Setup:** new `min_keep` setting in `FCCL` (default 0.1 = old behaviour); `floor_sim.py` runs the three simulated worlds, 5 seeds, and counts plans below 0.45. Simulated users, with tolerances I invented.
- **Result (tokens saved % / bad % first 10 turns / bad % last 20 turns / plans below 0.45 %):**

| world | policy | saved | first 10 | last 20 | plans < 0.45 |
|---|---|---|---|---|---|
| pattern holds | strength 1.0, min_keep 0.1 | 36.7 | 12.2 | 3.2 | 25.9% |
| | strength 1.0, min_keep 0.3 | 36.7 | 12.2 | 3.3 | 25.7% |
| | strength 1.0, min_keep 0.45 | 36.0 | 11.3 | 2.6 | 0.0% |
| | strength 1.0, min_keep 0.6 | 31.2 | 7.5 | 2.4 | 0.0% |
| | adaptive trust, min_keep 0.1 | 34.0 | 11.3 | 3.6 | 8.1% |
| | adaptive trust, min_keep 0.45 | 34.5 | 11.1 | 3.2 | 0.0% |
| no pattern | strength 1.0, min_keep 0.1 | 33.3 | 18.6 | 3.8 | 20.1% |
| | strength 1.0, min_keep 0.3 | 33.2 | 18.6 | 3.8 | 19.8% |
| | strength 1.0, min_keep 0.45 | 32.8 | 18.3 | 3.5 | 0.0% |
| | strength 1.0, min_keep 0.6 | 29.3 | 13.4 | 3.2 | 0.0% |
| | adaptive trust, min_keep 0.1 | 32.0 | 13.7 | 4.0 | 3.3% |
| | adaptive trust, min_keep 0.45 | 31.5 | 13.4 | 3.6 | 0.0% |
| pattern backwards | strength 1.0, min_keep 0.1 | 29.1 | 25.8 | 4.5 | 10.3% |
| | strength 1.0, min_keep 0.3 | 29.0 | 25.8 | 4.2 | 10.4% |
| | strength 1.0, min_keep 0.45 | 29.0 | 26.0 | 4.5 | 0.0% |
| | strength 1.0, min_keep 0.6 | 27.5 | 19.0 | 4.3 | 0.0% |
| | adaptive trust, min_keep 0.1 | 30.0 | 14.9 | 3.9 | 1.6% |
| | adaptive trust, min_keep 0.45 | 30.0 | 14.6 | 4.2 | 0.0% |

- **Takeaways:** without a floor, full-trust FCCL plans below the 0.45 knee in 10% to 26% of compressions (adaptive trust: 2% to 8%). A floor of 0.45 costs at most about 0.7 points of savings and slightly lowers bad reactions in two of three worlds. A floor of 0.3 has almost no effect. A floor of 0.6 trades about 5 points of savings for 5 to 7 fewer points of early bad reactions.
- **Decision:** code default stays `min_keep=0.1` so earlier simulation results stay reproducible; use `min_keep=0.45` with real compressors (`demo_pipeline.py` does).
- **Update (2026-11-02):** superseded by E18 to E26. The code default is now 0.6, the simulations pin `min_keep` explicitly, and the 0.45 floor has no measured knee behind it.
- **Caveats:** the simulation shows the cost of the floor, not that 0.45 is right; that number comes from three texts judged by one model (E13, E14). Differences of about 0.5 point are noise.
- **Files:** `fccl.py` (`min_keep`), `floor_sim.py`.


### E16. Simulation using the measured quality curve
- **Question:** once the simulation uses the measured quality-vs-keep curve (instead of invented thresholds), does FCCL beat a fixed keep?
- **Setup:** `real_curve_sim.py`. Quality kept at each keep level = mean blind-test score / 6 (E13, E14; both local methods averaged): 1.0 gives 100%, 0.6 gives 99%, 0.44 gives 92.5%, 0.3 gives 76%, 0.1 gives 52% (extrapolated, a guess). A user's chance of a correction is `K x sensitivity x quality lost`. Measured: the curve. Still invented: the user sensitivities and the dial K (3 = users care moderately, 6 = strongly). 100 users, 40 turns, 5 seeds, three worlds. "Quality kept" is averaged over planned compressions, weighted by context size.
- **Result (tokens saved % / quality kept % / bad % first 10 turns / bad % last 20 turns):**

| K | world | fixed keep 0.6 | FCCL strength 1.0, min_keep 0.45 | FCCL adaptive, min_keep 0.45 |
|---|---|---|---|---|
| 3 | pattern holds | 40.0 / 99.0 / 1.3 / 1.6 | 39.3 / 96.4 / 4.6 / 4.0 | 39.1 / 97.8 / 2.3 / 3.1 |
| 3 | no pattern | 40.0 / 99.0 / 1.5 / 1.9 | 38.4 / 96.6 / 5.9 / 4.3 | 38.6 / 97.8 / 2.7 / 3.5 |
| 3 | pattern backwards | 40.0 / 99.0 / 1.7 / 1.8 | 37.6 / 96.8 / 7.6 / 5.1 | 38.0 / 98.0 / 3.2 / 3.9 |
| 6 | pattern holds | 40.0 / 99.0 / 3.1 / 3.1 | 36.6 / 97.1 / 8.8 / 5.1 | 36.5 / 98.3 / 4.5 / 4.4 |
| 6 | no pattern | 40.0 / 99.0 / 3.3 / 3.6 | 35.2 / 97.5 / 10.9 / 5.4 | 35.6 / 98.5 / 4.6 / 4.6 |
| 6 | pattern backwards | 40.0 / 99.0 / 3.4 / 3.6 | 34.2 / 97.7 / 12.4 / 5.8 | 35.1 / 98.6 / 4.9 / 5.0 |

- Other fixed settings (K=3, pattern holds): keep 0.8 gives 20.0 / 99.5 / 0.8 / 1.0; keep 0.45 gives 55.0 / 92.9 / 11.0 / 10.4; keep 0.3 gives 70.0 / 76.0 / 36.9 / 36.9. At K=6 the bad-reaction rates roughly double.
- **Takeaways:** fixed keep 0.6 (right at the knee) beats every FCCL variant on savings, quality and bad reactions, in all three worlds and at both K. FCCL's formula (keep roughly 0.44 to 0.79) is not anchored to the measured curve: logistics plans land on the steep part, venting plans leave savings unused. Inside the FCCL family: learning still lowers bad reactions over time, the floor and adaptive trust still help, and the context weight adds at most about 1.5 points of savings while costing quality (98.4% to 95.2% at K=3).
- **Caveats:** fixed 0.6 was chosen with hindsight of the curve; every simulated user shares the same loss curve and differs only in sensitivity, so heterogeneity barely shows on the flat part; the curve is from three texts, one judge; real content types and users with different needs are not modelled. The measured data also gave no support for "venting needs more context": on the fact-recall measure the vent text lost less than the technical text at every keep level (this does not test the premise, which is about what users want).
- **Next:** add real heterogeneity (a picky minority, content types with different curves) to see when personalization can beat a fixed keep; then consider re-anchoring FCCL's plan to the curve.
- **File:** `real_curve_sim.py`.


### E17. Real-data test: compression vs question answering (SQuAD, 150 questions)
- **Question:** on real passages with human-marked answers, how much does compression hurt, measured by an objective model instead of my judging?
- **Setup:** `real_squad_test.py`; 150 random SQuAD v1 validation questions (seed 0); passages compressed by LLMLingua-2 (CPU) and SimpleCDM at keep 0.8, 0.6, 0.45, 0.3 with FCCL's protection rule (numbers and mid-sentence capitalised words from the question) and the guard on (the original passage is used when the guard falls back); `distilbert-base-cased-distilled-squad` answers by extracting a span (windows of 384 tokens, stride 128); F1 and exact match against the human answers; 95% margins.
- **Result:**

| method | keep | words kept | guard fell back | answer text survives | F1 | EM | F1 relative to original |
|---|---|---|---|---|---|---|---|
| original | 1.0 | 100% | 0% | 100% | 82.5 +- 5.5 | 74.7 +- 7.0 | 100% |
| LLMLingua-2 | 0.8 | 82% | 0% | 82% | 75.9 +- 5.9 | 60.0 +- 7.9 | 92% |
| LLMLingua-2 | 0.6 | 58% | 1% | 74% | 64.4 +- 6.4 | 42.7 +- 7.9 | 78% |
| LLMLingua-2 | 0.45 | 42% | 1% | 69% | 55.8 +- 6.5 | 33.3 +- 7.6 | 68% |
| LLMLingua-2 | 0.3 | 27% | 1% | 55% | 43.2 +- 6.5 | 24.0 +- 6.9 | 52% |
| SimpleCDM | 0.8 | 80% | 0% | 91% | 80.9 +- 5.4 | 66.7 +- 7.6 | 98% |
| SimpleCDM | 0.6 | 62% | 5% | 77% | 72.9 +- 5.9 | 50.0 +- 8.0 | 88% |
| SimpleCDM | 0.45 | 55% | 18% | 69% | 60.9 +- 6.5 | 38.7 +- 7.8 | 74% |
| SimpleCDM | 0.3 | 45% | 21% | 63% | 49.6 +- 6.9 | 31.3 +- 7.4 | 60% |

- **Takeaways:** quality falls steadily with compression; there is no plateau at 0.6 and no sharp knee at 0.45. This contradicts the hand-judged curve of E13/E14 (99% at 0.6): that test was lenient (a strong reader reconstructing meaning from clipped text, partial credit, keyword-friendly questions), while this one is strict (an extractive model must find the answer in the text word for word, so dropped small words inside an answer count against it). The true cost for an LLM answerer is probably between the two. SimpleCDM is at least as good as LLMLingua-2 at similar words kept (borderline; margins about +-6; no paired test yet), and its guard rejects it often at low rates (18% to 21%, meaning no savings on those questions) while LLMLingua-2 almost never falls back because its protected words are forced to stay.
- **Consequences:** the `min_keep=0.45` floor has no knee to justify it (it is still a harmless safety net, but the number is not supported); the E16 conclusion ("fixed keep 0.6 beats FCCL") rested on the flat curve and is unsupported.
- **Caveats:** Wikipedia passages (about 120 to 200 words) and single factual questions, not conversations; an extractive model, not a generative LLM; margins of +-5 to 8 points; unpaired comparisons between methods; the question's own names and numbers are protected, which helps retrieval of exactly the facts asked about; one run, one seed.
- **Next:** repeat with an LLM answerer and semantic grading on the same real questions, then rebuild the simulation on the measured curve.
- **File:** `real_squad_test.py` (output `squad_results.txt`, not committed).

## E18 - Real SQuAD, generative LLM answerer (2026-10-29)
Setup: 60 random SQuAD validation questions (seed 0). Answerer openai/gpt-oss-20b (Groq), judge openai/gpt-oss-120b,
correct = normalised containment of a gold answer, or judge YES when containment fails. Script: real_squad_llm_test.py.

| method | keep | words kept % | guard fell back % | says unknown % | correct % (95% margin) |
|---|---|---|---|---|---|
| original | 1.0 | 100 | 0 | 0 | 96.7 +-4.5 |
| LLMLingua-2 | 0.8 | 82 | 0 | 0 | 96.7 +-4.5 |
| LLMLingua-2 | 0.6 | 58 | 0 | 5 | 88.3 +-8.1 |
| LLMLingua-2 | 0.45 | 42 | 0 | 7 | 78.3 +-10.4 |
| LLMLingua-2 | 0.3 | 27 | 0 | 23 | 51.7 +-12.6 |
| SimpleCDM | 0.6 | 62 | 5 | 2 | 91.7 +-7.0 |
| SimpleCDM | 0.45 | 55 | 18 | 12 | 76.7 +-10.7 |

Findings:
- Quality falls steadily; no knee at 0.45. Keep 0.8 is free; 0.6 costs about 8 points (borderline significant); 0.45 about 18; 0.3 about 45.
- The hand-judged blind test (E-blind) was too lenient. min_keep=0.45 has no measured knee behind it; E16 conclusion (fixed keep 0.6 beats FCCL) is unsupported because it used that flat curve.
- LLMLingua-2 is not clearly better than SimpleCDM at keep 0.6 (88.3 vs 91.7, within noise).
- Caveats: SQuAD is single-fact lookup, close to a worst case for compression; n=60; one answerer model; judge is an LLM.
Decision: demo min_keep 0.45 -> 0.6. Next: rebuild the simulation on this measured curve.

Paired analysis (paired_analysis.py, same 60 questions, exact sign test + bootstrap):
| method | keep | lost | gained | diff (points) | 95% interval | p |
|---|---|---|---|---|---|---|
| LLMLingua-2 | 0.8 | 1 | 1 | +0.0 | [-5.0, +5.0] | 1.000 |
| LLMLingua-2 | 0.6 | 5 | 0 | -8.3 | [-15.0, -1.7] | 0.062 |
| LLMLingua-2 | 0.45 | 11 | 0 | -18.3 | [-28.3, -8.3] | 0.001 |
| LLMLingua-2 | 0.3 | 27 | 0 | -45.0 | [-58.3, -33.3] | 0.000 |
| SimpleCDM | 0.6 | 5 | 2 | -5.0 | [-13.3, +3.3] | 0.453 |
| SimpleCDM | 0.45 | 12 | 0 | -20.0 | [-30.0, -10.0] | 0.000 |
Reading: 0.8 is free; 0.6 is a likely small loss (borderline); 0.45 and below clearly hurt. Compression never helped.
LLMLingua-2 vs SimpleCDM at 0.6 cannot be separated with n=60.
Decision: default max compression keep 0.8; min_keep 0.6 as the lowest allowed; 0.45 knee rejected.

## E19 - PRISM: does per-user learning of length preference beat a global rule? (2026-10-29)
Setup: PRISM utterances (1389 users, 1-100 ratings of ~4 model responses per turn). Online, per user, using only earlier turns,
predict which response in a turn is rated higher from response length alone (log words). Script: prism_length_test.py.
Result (pairwise accuracy, 50% = chance): global rule 59.2%; user-only 55.8% (-3.4, CI [-4.1,-2.7]);
blends k=1/3/10/30/100: 56.9/58.0/59.1/59.2/59.2. User-only learning curve rises 53.6 -> 57.1% from 2-3 to 11+ turns, never reaches global.
81% of users rate longer answers higher; odd-vs-even-turn stability of a user's preference r = +0.12.
Finding: per-user length adaptation does not beat a global prior at ~12 turns/user; personal preference is weak.
Caveats: length is confounded with model quality (different models per turn); one crude feature; no prompt-type or stated-preference features; about output length, not input compression.
Implication: do not claim per-user adaptation helps; keep a strong global prior; feedback-driven settings will lean toward less compression.

## E20 - PRISM controls: model identity and conversation type (2026-10-29)
Script: prism_control_test.py. Same online per-user test as E19, with controls.
- Model-adjusted score: global 57.6%, user-only 54.7% (-3.0, CI [-3.7,-2.1]); blends k=10/30: -0.4/+0.1. Stability r=+0.11.
- Model-adjusted score and length: global 56.4%, user-only 53.6% (-2.8); stability r=+0.10.
- Global length slope falls from +11.7 to +5.9 after removing model quality: part of "longer is better" was model quality.
- Model identity alone orders 59.2% of pairs correctly, equal to length alone (~59%).
- By conversation type, slope +12.8 (controversy), +10.2 (unguided), +12.5 (values); longer-wins accuracy 60.8/58.3/59.2%. No strong context effect.
Finding: E19 holds after controlling for model quality; no hidden stable personal length preference (r about 0.1); no strong context dependence by conversation type.
Caveats: accuracies not comparable across variants; PRISM conversation types are not chatty/logistics; stated preferences and non-length features untested.

## E21 - Real DREAM dialogues, generative LLM answerer (2026-10-29, partial: 47 of 100 planned)
Setup: DREAM validation, dialogues of 100+ words (921 eligible), 47 questions of a fixed random sample (seed 0; run stopped by Groq free rate limit).
Answerer openai/gpt-oss-20b, multiple choice (3 options, shuffled), exact scoring, no judge. Speaker labels rewritten (Man/Woman) and protected. Script: real_dream_llm_test.py.
| method | keep | words kept % | guard fell back % | correct % (95% margin) | paired diff vs original | p |
|---|---|---|---|---|---|---|
| no dialogue | 0.0 | 0 | 0 | 27.7 +-12.8 | -68.1 | 0.000 |
| original | 1.0 | 100 | 0 | 95.7 +-5.8 | | |
| LLMLingua-2 | 0.8 | 91 | 4 | 93.6 +-7.0 | -2.1 (1 lost, 0 gained) | 1.000 |
| LLMLingua-2 | 0.6 | 68 | 0 | 87.2 +-9.5 | -8.5 (4 lost, 0 gained) | 0.125 |
| LLMLingua-2 | 0.45 | 51 | 0 | 72.3 +-12.8 | -23.4 (11 lost, 0 gained) | 0.001 |
| LLMLingua-2 | 0.3 | 34 | 0 | 68.1 +-13.3 | -27.7 (13 lost, 0 gained) | 0.000 |
| SimpleCDM | 0.6 | 60 | 0 | 91.5 +-8.0 | -4.3 (2 lost, 0 gained) | 0.500 |
| SimpleCDM | 0.45 | 46 | 2 | 76.6 +-12.1 | -19.1 (9 lost, 0 gained) | 0.004 |
Findings: sanity checks pass (no-dialogue 27.7% vs ~33% chance; original 95.7%). Curve matches SQuAD (E18): 0.8 free, 0.6 about -8.5, 0.45 about -23.
Chat-style dialogue did NOT tolerate more compression than SQuAD. Pooling E18 and E21 informally, LLMLingua-2 at keep 0.6: 9 lost, 0 gained (sign test p about 0.004); at 0.8: 2 lost, 1 gained.
Compression never improved an answer. SimpleCDM >= LLMLingua-2 at keep 0.6 on both datasets (within noise).
Caveats: n=47; the 0.3 row is damped by the 3-choice floor; DREAM dialogues are short and dense, not redundant; genuinely filler-heavy or venting text untested; one answerer model.
Decision: keep 0.8 as the no-cost level and 0.6 as the lowest allowed; no change to min_keep recommendation.

## E22 - Real MultiWOZ 2.2 booking conversations, generative LLM answerer (2026-10-30)
Setup: MultiWOZ 2.2 validation (1000 conversations with usable questions). 60 questions, 30 whose answer is stated once and 30 stated 2+ times.
Question templated from human-annotated final slot values (e.g. restaurant area); word answers only (numeric and yes/no excluded; guard protects number-like tokens by design).
Answerer openai/gpt-oss-20b, exact-match scoring (centre/center accepted). Speaker labels protected. Script: real_multiwoz_llm_test.py.
| method | keep | words kept % | guard fell back % | unknown % | correct % (95% margin) | paired change vs original |
|---|---|---|---|---|---|---|
| no dialogue | 0.0 | 0 | 0 | 100 | 0.0 | |
| original | 1.0 | 100 | 0 | 3 | 95.0 +-5.5 | |
| LLMLingua-2 | 0.8 | 86 | 0 | 5 | 91.7 +-7.0 | -3.3 (2 lost, 0 gained, p=0.500) |
| LLMLingua-2 | 0.6 | 63 | 0 | 8 | 86.7 +-8.6 | -8.3 (5/0, p=0.062) |
| LLMLingua-2 | 0.45 | 45 | 0 | 10 | 75.0 +-11.0 | -20.0 (12/0, p<0.001) |
| SimpleCDM | 0.6 | 60 | 0 | 7 | 81.7 +-9.8 | -13.3 (8/0, p=0.008) |
Repetition (LLMLingua-2 change, answer stated once vs 2+ times): 0.8: -3.3 / -3.3; 0.6: -13.3 / -3.3; 0.45: -23.3 / -16.7. SimpleCDM 0.6: -16.7 / -10.0.
Direction consistent in every row, but 30 questions per group: not significant (p around 0.4 at 0.6).
Findings: same curve as SQuAD (E18) and DREAM (E21): keep 0.6 about -8 points on all three; 0.45 about -18 to -23. Compression never helped (0 gained everywhere).
Informal pooling E18+E21+E22 (LLMLingua-2): keep 0.8: 4 lost, 1 gained; 0.6: 14 lost, 0 gained; 0.45: 34 lost, 0 gained.
Filler/repetition did not make logistics-style text safe to compress hard. FCCL's own plan for logistics (context weight 0.8, keep about 0.44) corresponds to about -20 points here, so min_keep=0.6 does real work.
Caveats: n=60, one answerer, templated questions, strict scoring, numeric answers excluded, venting-style text still untested.

## E23 - Failure autopsy and retry-on-abstention cascade (2026-10-30, exploratory, post-hoc)
Scripts: autopsy_multiwoz.py (no API calls), cascade_analysis.py (no API calls; reads squad_llm_cache.json and multiwoz_llm_cache.json).
Autopsy (MultiWOZ, 60 questions, LLMLingua-2): the answer word survives in 59/60 compressed conversations at keep 0.6 and 57/60 at 0.45.
Of 5 questions broken at 0.6: answer deleted in 0, present but answered wrong in 5. Of 12 broken at 0.45: deleted in 2, present but wrong in 10.
Loss is damaged context (wrong binding of value to slot/option), not deleted tokens, so token protection cannot fix most failures.
By who states the answer (broken/total): keep 0.6 customer-only 1/25, agent-only 4/10, both 0/22; keep 0.45 customer-only 6/25, agent-only 3/10, both 3/22. The agent speaks 55% of the words. Small post-hoc subgroups; hint only.
Cascade: if the compressed-context answer is "unknown", retry with the original context. Cost counts context tokens only (compressed call = fraction kept; a retry adds 100%).
Pooled (SQuAD 60 + MultiWOZ 61 questions; 61 = 60 plus one extra from the smoke run), LLMLingua-2, tokens used / accuracy change vs original (lost/gained):
| keep | plain | cascade |
|---|---|---|
| 0.8 | 84% / -1.7 (3/1) | 87% / +0.0 (1/1) |
| 0.6 | 60% / -8.3 (10/0) | 67% / -1.7 (2/0), p = 0.5 |
| 0.45 | 43% / -19.0 (23/0) | 52% / -10.7 (13/0), p < 0.001 |
SimpleCDM keep 0.6: plain 61% / -9.1 (13/2); cascade 65% / -5.0 (8/2).
Per dataset at keep 0.6: SQuAD cascade -3.3 (2/0); MultiWOZ cascade +0.0 (0/0); confident wrong answers (retry cannot catch) SQuAD 2, MultiWOZ 0.
At keep 0.45 the abstention rate barely rises (SQuAD 5% to 7%, MultiWOZ 8% to 10%) but confident wrong answers jump (pooled 2 to 13), so the cascade cannot rescue harsher compression and abstention rate is a poor dial for the level.
Finding: keep 0.6 plus retry-on-abstention gives about 33% context savings at about -1.7 points, vs about 16% savings at -1.7 for plain keep 0.8.
Caveats: discovered after looking at these results (needs replication on fresh questions); context tokens only (latency and the extra request ignored); needs a model that abstains (prompts told it to; DREAM multiple choice cannot); informal pooling; simulated from saved answers (temperature 0), not a live run.
Decision: replicate on fresh, larger samples before claiming it; keep min_keep=0.6.

## E24 - PRISM: does the survey predict a user's length preference? (2026-10-31)
Script: prism_survey_test.py. 1367 users with 6+ rated turns. Target = each user's own length slope (positive = rates longer answers higher).
Reliability: odd/even-turn correlation of the slope +0.12 (full-length reliability about 0.21), so the ceiling for any predictor's correlation with the observed slope is about 0.46.
Out-of-sample (5-fold over users, weighted ridge), correlation with observed slope (lambda 10/100/1000): stated preferences +0.079/+0.081/+0.089 (permutation p 0.040 at lambda 100);
AI usage +0.018/+0.016/+0.003 (p 0.597); free text brief/detail +0.032/+0.031/+0.026 (p 0.493); demographics (reference only) +0.012/+0.018/+0.019 (p 0.612); all except demographics +0.063/+0.064/+0.064 (p 0.065).
Top-vs-bottom 20% gap in observed slope: +1.47 (stated prefs), +0.72, +1.61, -0.62, +2.02, against a typical slope of about +11.7.
Pairwise accuracy vs the global rule: +0.0 [+0.0, +0.0] for every group (degenerate: predicted slopes stay positive for nearly everyone, so the ordering never changes).
Free text: 185 users mention brevity (mean slope +12.81 vs +10.91 for others); 57 mention detail (+13.21 vs +11.08). Stated wish for brevity does not show up as a lower slope (difference not tested).
Finding: nothing in the survey predicts length preference. The p=0.04 on stated preferences is likely chance (5 groups tested) and explains under 1% of variance.
Together with E19/E20: on PRISM, for answer length, a strong global default beats per-user learning (about 12 turns each) and survey-based guesses.
Caveats: length only; about 12 turns per user (about 21% of slope variance is stable trait at that length, so longer histories could make per-user learning worthwhile, untested); about output length, not compression.

## E25 - PRISM multi-feature test: do people differ on things other than length? (2026-10-31)
Script: prism_multi_test.py. 1396 users (1390 in the online test). Seven response features: length (log words), bullet list, disclaimer/refusal, hedging density, asks a question, bold/headings, word complexity.
Part 1 (global weight in score points per SD of the within-turn feature / share of users leaning positive / odd-vs-even stability r):
length +5.60 / 81% / +0.15; bullet list +0.92 / 70% / +0.06; disclaimer/refusal -1.35 / 47% / +0.06; hedging +0.51 / 61% / +0.01;
asks a question -1.05 / 44% / +0.05; bold/headings -0.57 / 47% / +0.05; word complexity +2.08 / 77% / +0.04.
Part 2 (online, pairwise accuracy; global 60.3%): user-only k=1/3/10: 55.5/55.7/55.9 (-4.8/-4.6/-4.4, all intervals below 0);
blend k=3/10/30/100/300: 57.2/58.2/59.5/60.4/60.4 (-3.1/-2.1/-0.8/+0.1/+0.1; k=100 interval [-0.2, +0.4]).
Learning curve: user-only rises from about 52% (2-3 turns) to about 57% (11+); global about 60% throughout.
Finding: no feature shows a stable personal lean (stability 0.01-0.06 except length 0.15); per-user learning does not beat the global rule with seven features; blending only recovers the global level.
With E19, E20 and E24: four real-data tests of per-user adaptation at about 12 turns per user, all negative.
Caveats: about 12 turns per user (longer histories untested); rare binary features give few examples per user, so low stability is partly measurement noise; PRISM is about answer preferences, not compression; lexical features only.

## E26 - Replication of the retry-on-unknown cascade on fresh questions (2026-11-02)
Script: replicate_cascade.py. 100 fresh SQuAD + 100 fresh MultiWOZ questions, none used earlier. Answerer gpt-oss-20b, judge gpt-oss-120b. Pre-registered: cascade at keep 0.6 loses <= 3 points at <= 72% of tokens (pooled).
Pooled (change vs original, lost/gained, tokens %): LLMLingua-2 0.8 plain +0.5 (8/9) 84%, cascade +2.0 (5/9) 89%; 0.6 plain -4.5 (21/12) 60%, cascade -1.0 (14/12) 67% p=0.845; 0.45 plain -16.0 (41/9) 43%, cascade -8.5 (26/9) 55% p=0.006.
Result: PASS. Cascade at 0.6: -1.0 points (95% interval [-6.0, +4.0]) at 67% of tokens. Retried 7% of questions.
Finding: retry-on-abstention replicates. It recovers most of the plain-0.6 loss. It still fails at 0.45 (26 confident wrong answers).
Caveats: wide interval (a real loss of 4 to 5 points is not ruled out); one answerer model; needs a model that abstains; judge is an LLM.
Decision: wired into the code on 2026-11-02: `cdm.answer_with_retry` (retry on "unknown"), the automatic `"abstained"` feedback in `fccl.py`, and the `min_keep` default raised to 0.6. Offline wiring test: `test_cascade_wiring.py` (no live model run yet).

## E27 - Live end-to-end test of the wired retry net (2026-11-03)
Script: live_cascade_test.py. Real path on 60 fresh SQuAD questions (seed 2, excluding E18 and E26 samples): FCCL.plan (min_keep 0.6, keyword context rule, one user) -> LLMLingua-2 -> cdm.answer_with_retry -> live Groq answers (gpt-oss-20b, judge gpt-oss-120b) -> automatic "abstained" feedback. Same prompt as E26. Compared with the uncompressed answer on the same questions.
First run found a BUG in cdm.is_abstention: it stripped digits, so a bare-number answer ("1998", "42%") counted as "unknown". 17% of questions were retried (E26: about 4%), 10 false "abstained" signals pushed the satisfaction weight 1.00 -> 0.60, FCCL raised its keep to as high as 0.87, and cost reached 91% of the words (accuracy -3.3). Fixed (digits kept) and covered by a regression case in test_cascade_wiring.py; the offline fake-model test could not have found it.
Rerun after the fix (accuracy / change vs original, lost/gained / context words used):
- uncompressed 98.3%
- compressed, no retry 90.0%, -8.3 (5/0), p=0.062, 66% of words
- compressed + retry (real) 95.0%, -3.3 (2/0), p=0.500, 71% of words
Retried 3 of 60 (5%); 1.05 requests per question; guard fell back 0 times. FCCL's keep: mean 0.67, range 0.60 to 0.82. Satisfaction weight 1.00 -> 0.86 after 3 abstentions.
Prediction check (E26 prediction): -3.3 points (95% interval [-8.3, +0.0]) at 71% of the words: FAIL on the strict reading (limit was -3.0 at 72%). One question is worth 1.7 points at n=60, so the verdict hinges on one question.
Finding: the wired code works end to end. Cost matches (71% vs 67% in E26; higher because FCCL plans about 0.65, and abstention feedback raised keep slightly). The retry fixed 3 of the 5 compressed-only losses; the 2 remaining are confident wrong answers, as E23 and E26 predicted. Accuracy loss is consistent with E26 within noise but is not stronger evidence: informally pooled with E26's SQuAD cascade (9 lost, 8 gained) the retry net is 11 lost, 8 gained over 160 questions (-1.9 points).
Caveats: n=60 (wide interval); SQuAD only, one answerer model; all 60 independent questions were sent through one simulated user, so abstentions accumulate on one weight (a real user has one conversation); the per-abstention back-off (0.95, or 0.9 right after an experiment) is a number I chose and was not tuned; latency of the extra request not counted; plain keep -8.3 here vs -4.0 in E26 shows how much a 60-question sample moves.

## E28 - Live end-to-end test, 100 SQuAD + 100 MultiWOZ questions, five simulated users (2026-11-03)
Script: live_cascade_test.py (squad 100, multiwoz 100). Same real path as E27 (FCCL.plan -> LLMLingua-2 -> answer_with_retry -> live Groq answers -> "abstained" feedback), is_abstention fixed. Questions dealt to 5 simulated users in turn (own satisfaction weight each). SQuAD: first 60 = the E27 sample, 40 new; fresh relative to E18 and E26. MultiWOZ: conversations not used in E22 or E26. Same prompts as E26.
SQuAD, 100 questions (accuracy / change vs uncompressed, lost/gained, p / context words used): uncompressed 98.0%; compressed, no retry 91.0%, -7.0 (7/0), p=0.016, 65%; compressed + retry 95.0%, -3.0 (3/0), p=0.250, 69%. Retried 4%; keep mean 0.66 (0.60 to 0.79); weights 0.90 to 1.00. Interval [-7.0, +0.0].
MultiWOZ, 100 questions: uncompressed 91.0%; compressed, no retry 87.0%, -4.0 (9/5), p=0.424, 75%; compressed + retry 92.0%, +1.0 (4/5), p=1.000, 83%. Retried 8%; keep mean 0.70 (0.60 to 0.82); weights 0.86 to 1.00 (median 0.90). Interval [-5.0, +7.0].
Combined (informal, 200 questions): no retry about -5.5 points (16 lost, 5 gained) at about 70% of the words; with retry about -1.0 point (7 lost, 5 gained) at about 76% of the words; 1.06 requests per question.
Prediction check (E26 prediction was pooled: loss of at most 3 points at 72% of the words or less): accuracy part PASSES (about -1.0), cost part FAILS (about 76%). SQuAD alone: -3.0 at 69% (pass, exactly at the limit). MultiWOZ alone: +1.0 at 83% (fails the cost limit; in E26 MultiWOZ alone was 73%).
Finding: the retry net holds up on accuracy in the live pipeline on both datasets (it recovered 4 of 7 SQuAD and 5 of 9 MultiWOZ compressed-only losses). The saving is smaller than E26's fixed-0.6 numbers: FCCL plans about 0.65 for neutral questions (not 0.6), abstention feedback pushes keep up for that user (0.66 to 0.70 mean), LLMLingua-2 keeps a little more than asked on MultiWOZ, and 8% retries add 8 points. Realistic summary: about 24% fewer words (pooled) at about -1 point; about 31% fewer on SQuAD at -3.0; about 17% fewer on MultiWOZ at +1.0.
Caveats: one answerer model; wide intervals ([-7.0, +0.0] and [-5.0, +7.0]); the 60 E27 questions overlap with the SQuAD sample here; simulated users share one stream of independent questions; the 0.95 per-abstention back-off and the neutral plan of 0.65 are numbers I chose and were not tuned; baseline accuracy differs by dataset (98% vs 91%), so lost/gained is not comparable across them; pooled figures are informal averages; latency of the extra request not counted. Lowering the neutral plan to about 0.6 would raise the saving but must be tested on fresh questions (not these).

## E29 - Live test with compress_strength 0.8 (neutral plan keep 0.6), fresh questions (2026-11-05)
Script: live_cascade_test.py squad 100 skip=100 strength=0.8, and multiwoz 100 skip=100 strength=0.8. Questions 101 to 200 of the fixed samples (never used by E27 or E28), five simulated users, same prompts and models as E28. FCCL `compress_strength` 0.8 plans keep 0.6 for a neutral question (default 0.7 plans 0.65). Pre-registered before any run: pooled over the two sets, loss of at most 3 points at 72% of the words or less.
SQuAD, 100 questions (accuracy / change vs uncompressed, lost/gained, p / context words used): uncompressed 96.0%; compressed, no retry 87.0%, -9.0 (12/3), p=0.035, 59%; compressed + retry 91.0%, -5.0 (8/3), p=0.227, 64%. Retried 5%; mean planned keep 0.61 (0.60 to 0.77); weights 0.90 to 1.00. Interval [-12.0, +1.0].
MultiWOZ, 100 questions (the first attempt stopped at 77 of 100 at the Groq limit and was resumed from the cache; the table below is the finished run): uncompressed 89.0%; compressed, no retry 83.0%, -6.0 (11/5), p=0.210, 67%; compressed + retry 88.0%, -1.0 (6/5), p=1.000, 73%. Retried 6%; mean planned keep 0.64 (0.60 to 0.78). Interval [-8.0, +6.0].
Combined (informal, 200 questions): no retry about -7.5 points (23 lost, 8 gained) at about 63% of the words; with retry about -3.0 points (14 lost, 8 gained) at about 69% of the words; 1.06 requests per question.
Prediction check (pooled): -3.0 points at about 69% of the words: PASS, exactly at the accuracy limit (one question is worth 0.5 points pooled). Per dataset: SQuAD -5.0 (accuracy limit exceeded); MultiWOZ -1.0 at 73% (cost limit exceeded by one point).
Finding: planning 0.6 instead of 0.65 moved the system along the trade-off as the plain columns predicted: about 31% fewer words at about -3 points (E29) against about 24% fewer at about -1 point (E28). The retry roughly halves the plain loss at both settings (-5.5 to -1.0 and -7.5 to -3.0). Neither setting is free.
Caveats: the E28 and E29 rows differ by amounts inside sampling noise (intervals of 12 to 14 points width per dataset; uncompressed accuracy differs between the SQuAD samples, 98% vs 96%), so this does not show that 0.6 is worse than 0.65; one answerer model; simulated users; pooled figures are informal averages; latency not counted.

## E30 - Live test with two other answerer models, default plan (2026-11-06)
Script: live_cascade_test.py with answerer=qwen/qwen3.8-27b and answerer=allam-2-7b, squad 100 and multiwoz 100 each (the same questions 1 to 100 as E28, compress_strength 0.7, five simulated users, LLMLingua-2, same prompts, same judge gpt-oss-120b). Models were chosen from what the Groq key listed (`python live_cascade_test.py models`): no Llama chat model was available; the others were speech or safety models. gpt-oss-20b rows are E28.
Per run (accuracy / change vs uncompressed, lost/gained / context words used; "plain" = no retry):
- Qwen3.8-27B, SQuAD: uncompressed 91.0%; plain 85.0%, -6.0 (8/2), p=0.109, 66%; with retry 89.0%, -2.0 (4/2), p=0.688, 75%. Retried 9%; weights 0.86 to 1.00. Interval [-7.0, +3.0].
- Qwen3.8-27B, MultiWOZ: uncompressed 97.0%; plain 94.0%, -3.0 (4/1), p=0.375, 73%; with retry 95.0%, -2.0 (3/1), p=0.625, 74%. Retried 1%. Interval [-6.0, +2.0]. (The first attempt stopped at question 28 on a Groq 503 "over capacity" error; resumed from the cache after the script was changed to back off and retry on 5xx errors.)
- ALLaM-7B, SQuAD: uncompressed 93.0%; plain 90.0%, -3.0 (5/2), p=0.453, 65%; with retry 91.0%, -2.0 (4/2), p=0.688, 67%. Retried 2%. Interval [-7.0, +3.0].
- ALLaM-7B, MultiWOZ: uncompressed 91.0%; plain 89.0%, -2.0 (5/3), p=0.727, 74%; with retry 90.0%, -1.0 (4/3), p=1.000, 76%. Retried 2%. Interval [-6.0, +4.0].
Per answerer, SQuAD and MultiWOZ together (informal, 200 questions each): gpt-oss-20b plain -5.5 (16 lost, 5 gained) at 70% of the words, with retry -1.0 (7/5) at 76%, retried 6%; Qwen3.8-27B plain -4.5 (12/3) at 70%, with retry -2.0 (7/3) at 75%, retried 5%; ALLaM-7B plain -2.5 (10/5) at 70%, with retry -1.5 (8/5) at 72%, retried 2%.
All three answerers, six runs (informal): plain 38 lost, 13 gained, about -4.2 points at about 70% of the words; with retry 22 lost, 13 gained, about -1.5 points at about 74% of the words. The retry reduced the loss in all six runs and made none worse. The 600 answers are only 200 distinct questions asked of three models, so they are not independent and no pooled p-value is given.
Prediction check (E26 prediction, per run): the accuracy part (at most 3 points lost) is met in all six runs (gpt-oss SQuAD sits exactly at -3.0); the cost part (72% of the words or less) is met in two of six (gpt-oss SQuAD 69%, ALLaM SQuAD 67%) and missed in four (gpt-oss MultiWOZ 83%, Qwen 75% and 74%, ALLaM MultiWOZ 76%). The script prints FAIL on the strict reading in those four.
Finding: the retry net is not specific to gpt-oss-20b. On two models from other families it brought a plain loss of 2 to 6 points down to 1 to 2 points, at about 72% to 75% of the words. How much it can recover depends on how often the model abstains: Qwen said "unknown" on 9% of SQuAD questions (and recovered 4 of 8 losses), ALLaM on 2% (it recovered 1 of 5 SQuAD losses and 1 of 5 MultiWOZ losses; the rest are confident wrong answers the net cannot see). The smaller model (ALLaM) lost the least from compression in these samples, which is not what one would guess and is within noise.
Caveats: all three answerers are served by one provider on its free tier at temperature 0 with the same prompts; no closed or large dense model; the three runs share the same 200 questions, so errors are correlated; samples of 100 give intervals about 8 to 10 points wide and every interval includes zero; the judge (gpt-oss-120b) is from the same family as one of the answerers and grades SQuAD borderline cases only (MultiWOZ is exact match); the harder plan (strength 0.8) was not run on these models; the latency of the extra request is not counted; simulated users never complain.


## E31 - Latency of the compression layer and the retry (2026-11-09)
Script: latency_test.py (not cached; times only the successful request, not rate-limit waits or the pause between questions), SQuAD, gpt-oss-20b, LLMLingua-2 on a laptop CPU, default plan, one question at a time. Two runs: questions 1 to 30, and questions 31 to 90 (the second run also splits the compressed path into compression and request).
Questions 31 to 90 (n=60), median / 90th percentile:
- uncompressed, one request: 0.67 s / 0.86 s
- compressed, no retry: 1.46 s / 1.65 s (compression 0.69 s median, slowest 2.09 s; the request on the compressed text 0.79 s)
- FCCL pipeline with retry: 1.46 s / 1.83 s, slowest 2.89 s
- retried 3 of 60 (5%); a retried question took 0.66 s longer (median) than the no-retry path; averaged over all questions the retry adds 0.03 s (2% of the no-retry median).
Questions 1 to 30 (n=30): uncompressed 0.82 s, compressed 1.66 s (median), 0 retries, slowest 6.55 s (compression time was not split out in this run, so the cause of the slow case is not known).
Finding: on this setup the retry is cheap in time (about one extra full request on the 5% of questions where it fires), but the compression step is not free: it adds about 0.7 s per question locally, and the request on the compressed text was not faster than the full one (0.79 s against 0.67 s; passages are short, so a fixed per-request time dominates). So FCCL saves tokens and cost here, not time. Time savings would need long inputs or a slower model, which was not measured.
Caveats: free-tier latency varies with load and the two numbers come from different moments; SQuAD only (MultiWOZ was not timed); one answerer; short passages (about 150 words); compression time depends on the CPU and was measured with LLMLingua-2 on CPU only; the 5% retry rate is from 60 questions, so it is imprecise (3 events).


## E32 - Simulation: does per-user learning help when users differ in tolerance? (2026-11-13)
Script: personalisation_sim.py. SIMULATION ONLY: it tests whether the mechanism can work, not whether real people behave this way. Simulated users each have a tolerance t (unhappy if the plan keeps less than t of the text), drawn from a normal distribution with mean 0.55, 0.65 or 0.75 and spread 0, 0.1 or 0.2. Unhappy users complain with probability 0.5; happy users thumbs-up with 0.1 and falsely complain with 0.03. Utility per request = (1 - keep) - 1.0 * unhappy. Policies: oracle (knows each tolerance), global (one fixed keep level, best chosen with hindsight), fixed (FCCL plan with learning off), learn (FCCL normal). Same users and random numbers for all policies; paired bootstrap over users. 300 users x 40 requests (min_keep 0.6 and 0.3) and 200 users x 200 requests (min_keep 0.6).
Pre-registered predictions (written before the first run, which used only mean 0.75): P1 no gain over the best global level at spread 0: held (learning was below the hindsight-tuned global level in every spread-0 cell). P2 learning beats non-learning at spread 0.15 / 40 requests / min_keep 0.3: held in the neighbouring cells tested (spread 0.1 and 0.2, interval excludes 0; spread 0.15 itself was not run). P3 small gain under the default min_keep 0.6 (under 0.02): WRONG; gains over non-learning were +0.08 to +0.45 in 17 of 18 cells (the exceptions: +0.001 and +0.006 where the default already fits). P4 no win over the best global level at 40 requests: held in most cells.
Results, learn minus fixed: positive with an interval excluding zero in 26 of 27 cells (spread 0 and a default that already fits: +0.001, interval includes zero). Learn minus best global level, min_keep 0.6: at 40 requests learning is better in 1 of 9 cells, tied in 2, worse in 6; at 200 requests better in 3 (spread 0.1 to 0.2 with mean 0.55 to 0.65: +0.020 to +0.100), tied in 3, worse in 3 (all three spread-0 cells, where one global level is exactly right).
Finding: learning is clearly better than FCCL without learning in this setup (it backs off after complaints, moving toward keeping more text). It beats a well-chosen single global level only when users differ clearly (spread 0.1 to 0.2), the default plan is too aggressive for them, and each user has about 200 requests; at 40 requests per user it usually does not, because each user gives only about 4 complaints. The sweep over mean tolerance was added after seeing the first run, so it is exploratory.
Caveats: simulated users; the tolerance, complaint rates and cost of an unhappy answer are numbers I chose; the global baseline is tuned with hindsight on the same users, which flatters it; with the default min_keep 0.6 the plan can only personalise toward keeping more text, never compress more than 40%; the result depends on where tolerance sits relative to FCCL's default plan (built into the setup). Implication for a real study: about 40 rated answers per person is too few to show a gain over one good global setting; plan for about 100 or more per person.