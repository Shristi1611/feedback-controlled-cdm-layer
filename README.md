# FCCL: Feedback-Controlled CDM Layer

A feedback-driven layer between users and CDMs that decides when and how hard to compress, without hurting answer quality.

**Status:** working prototype with a measured safe range. The guarded compression pipeline works, and its quality cost has been measured on real data (real SQuAD questions, real DREAM dialogues and real MultiWOZ booking conversations, a real generative LLM, real compressors). The two ideas that made the project distinctive, learning per user and weighting by conversation context, have **not** been supported by real data so far: a real-data test of per-user length adaptation (PRISM) found it did not beat a single global rule. The learning results in the simulation sections below use *invented* users, so they show that the mechanism works, not that it helps real people. The full experiment log is in [`results.md`](results.md).

## What real data says (read this first)

| claim | evidence | status |
|---|---|---|
| Guarded compression at a mild rate is safe | keep 0.8 shows no detectable loss on SQuAD (E18) and DREAM dialogues (E21), and at most about 3 points on MultiWOZ (E22) | supported, on the three kinds of task tested |
| Compression below about 0.6 is cheap | keep 0.6 costs about 8 points, 0.45 about 18 to 23, on three different tasks (SQuAD, DREAM dialogues, MultiWOZ booking conversations); there is **no knee at 0.45** and filler or repetition did not change that (E17, E18, E21, E22) | rejected |
| The guard is worth having | at nominal keep 0.45 it caught 18% of cases the naive compressor would have damaged (E18) | supported |
| Learning each user's preference beats a global rule | PRISM, 1,389 users: global rule 59.2% vs per-user learning 55.8% pairwise accuracy; held after controlling for the model (E19, E20) | not supported (answer length only) |
| The context weight helps | one real check by conversation type found no strong effect; the logistics plan (keep about 0.44) cost about 20 points on real booking conversations; venting-style text untested on real data (E20, E22) | not supported, not yet fairly tested |
| Laya labels conversation types better than keywords | 18/20 vs 8/20 on 20 messages I wrote and labelled | weak evidence |

Recommended settings today: keep rate **0.8** costs little or nothing; **0.6** is the lowest I would allow (`min_keep=0.6`), and it costs a few points; keep the guard on.

## The idea

A CDM (constrained decoding model, such as ScaleDown's Compress) shrinks context before it reaches the LLM. Compress too hard for the wrong user or the wrong conversation and answers get worse. FCCL sits in the middle, reads the user and the conversation, tells the CDM how to compress, and learns from the user's reactions.

```
user message + history
        |
   [ FCCL layer ]  <--- feedback (user reactions)
   reads user + conversation,
   builds a compression plan
        |
   plan: apply?, rate, preserve
        v
   [ CDM backend ] compresses the context
        |
   [ guard ]  checks the result, falls back to the original if it is unsafe
        |
      [ LLM ]  answers
        |
   answer --> user --> reaction goes back to FCCL
```

## How it works

Each decision combines two weights and a safety limit on experimenting:

- **Context weight:** how compressible this kind of conversation is. Venting or chatting keeps more of the context (0.3). Logistics compresses more (0.8). Anything else is neutral (0.5). The detector is pluggable: a simple keyword rule by default, or Laya (a small local decision model) through `FCCL(context_fn=LayaContext())`. How much the layer trusts this guess is controlled by `ctx_strength` (0 = ignore it, 1 = trust it fully), and with `adaptive_trust=True` the layer learns the right level itself.
- **Satisfaction weight:** one number per user, updated from reactions to the answer ("show more", correction, regenerate, thumbs). Unhappy users get compressed less. Reactions to the *answer* count, not general mood, so a user venting is not treated as dissatisfied.
- **Careful experimenting:** the layer sometimes compresses a little harder for users who seem happy. It never experiments during a user's first few requests, never for unhappy users, and never below a floor. It remembers which plans were experiments, so a bad reaction can be traced back to the experiment and punished harder.
- **Adaptive trust:** the layer keeps a score for each trust level (0, 0.25, 0.5, 1.0): text saved, minus a penalty for each bad reaction. It mostly uses the best-scoring level, tries a random one 15% of the time, and ties go to the lower (safer) level.
- **Hard minimum (`min_keep`):** no plan ever keeps less than this fraction of the text. The default is 0.1 (the old behaviour, so earlier simulation results stay reproducible). Real quality tests (below) show little or no loss at 0.8, a likely small loss at 0.6 and a clear loss at 0.45 and below, so use `min_keep=0.6` with a real compressor.

The plan it produces has three fields so far: `apply` (compress at all?), `rate` (fraction of the text to keep) and `preserve` (words that must survive: numbers, and capitalised words that are not at the start of a sentence).

### The CDM side (`cdm.py`)

Backends share one interface: `LLMLinguaCDM` (local, CPU), `ScaleDownCDM` (the real API) and `SimpleCDM` (a free word-dropper, used as a stand-in). `apply_plan()` runs a backend and then a **guard**:

- protected words (from the user's message) and ID-like tokens in the context (numbers, codes such as order or part numbers) must survive, or the original text is used instead;
- if the output is not actually shorter than the original, the original is used;
- it reports the *measured* kept fraction, because the requested rate is only approximately honoured.

`ScaleDownCDM` skips the API call when the requested removal is below 50% (`min_removal`), because calibration showed it makes text longer there. `LLMLinguaCDM` passes the protected words to LLMLingua-2 as forced tokens.

## Real-data tests

These are the experiments that did not rely on invented users. Full tables and caveats are in `results.md`.

### Compression quality on real SQuAD questions, generative LLM (E18)

60 random SQuAD validation questions. `openai/gpt-oss-20b` (Groq) answers using only the passage, and `openai/gpt-oss-120b` judges when a simple containment check fails. LLMLingua-2 and SimpleCDM compress the passage; the question's protected words are kept.

| method | keep | words kept | guard fell back | says unknown | correct (95% margin) |
|---|---|---|---|---|---|
| original | 1.0 | 100% | 0% | 0% | 96.7% (±4.5) |
| LLMLingua-2 | 0.8 | 82% | 0% | 0% | 96.7% (±4.5) |
| LLMLingua-2 | 0.6 | 58% | 0% | 5% | 88.3% (±8.1) |
| LLMLingua-2 | 0.45 | 42% | 0% | 7% | 78.3% (±10.4) |
| LLMLingua-2 | 0.3 | 27% | 0% | 23% | 51.7% (±12.6) |
| SimpleCDM | 0.6 | 62% | 5% | 2% | 91.7% (±7.0) |
| SimpleCDM | 0.45 | 55% | 18% | 12% | 76.7% (±10.7) |

Paired against the original on the same questions (`paired_analysis.py`, exact sign test and bootstrap):

| method | keep | lost | gained | difference | 95% interval | p |
|---|---|---|---|---|---|---|
| LLMLingua-2 | 0.8 | 1 | 1 | +0.0 | [-5.0, +5.0] | 1.000 |
| LLMLingua-2 | 0.6 | 5 | 0 | -8.3 | [-15.0, -1.7] | 0.062 |
| LLMLingua-2 | 0.45 | 11 | 0 | -18.3 | [-28.3, -8.3] | 0.001 |
| LLMLingua-2 | 0.3 | 27 | 0 | -45.0 | [-58.3, -33.3] | 0.000 |
| SimpleCDM | 0.6 | 5 | 2 | -5.0 | [-13.3, +3.3] | 0.453 |
| SimpleCDM | 0.45 | 12 | 0 | -20.0 | [-30.0, -10.0] | 0.000 |

- Keep 0.8 is free. 0.6 probably costs a few points (borderline). 0.45 and below clearly cost. Compression never improved an answer.
- **There is no knee at 0.45.** The hand-judged blind tests below were too lenient.
- LLMLingua-2 and SimpleCDM cannot be separated at keep 0.6 with 60 questions.
- At nominal 0.45, SimpleCDM only kept 55% of words because the guard fell back to the original on 18% of questions, so that row is not comparable.
- SQuAD is single-fact lookup, close to a worst case for compression.

Run it with `python real_squad_llm_test.py 60` (needs `GROQ_API_KEY`; resumable) and then `python paired_analysis.py`.

### Compression quality on real SQuAD, strict extractive reader (E17)

150 questions, DistilBERT extractive QA. F1 was about 64 to 73 at keep 0.6 against 82.5 for the original text, with a steady decline as compression increases. This is the strict end; the generative LLM above is more forgiving, and the truth for a given reader sits between the two. Run it with `python real_squad_test.py`.

### Compression quality on real dialogues, DREAM (E21, partial)

47 multiple-choice questions about conversations of 100+ words (the run was stopped early by a free rate limit). Same answerer as above, exact scoring (the model replies with a letter, choices shuffled), speaker labels rewritten as "Man:" and "Woman:" and protected.

| method | keep | words kept | guard fell back | correct (95% margin) | paired change vs original | p |
|---|---|---|---|---|---|---|
| no dialogue | 0.0 | 0% | 0% | 27.7% (±12.8) | -68.1 | 0.000 |
| original | 1.0 | 100% | 0% | 95.7% (±5.8) | | |
| LLMLingua-2 | 0.8 | 91% | 4% | 93.6% (±7.0) | -2.1 (1 lost, 0 gained) | 1.000 |
| LLMLingua-2 | 0.6 | 68% | 0% | 87.2% (±9.5) | -8.5 (4 lost, 0 gained) | 0.125 |
| LLMLingua-2 | 0.45 | 51% | 0% | 72.3% (±12.8) | -23.4 (11 lost, 0 gained) | 0.001 |
| LLMLingua-2 | 0.3 | 34% | 0% | 68.1% (±13.3) | -27.7 (13 lost, 0 gained) | 0.000 |
| SimpleCDM | 0.6 | 60% | 0% | 91.5% (±8.0) | -4.3 (2 lost, 0 gained) | 0.500 |
| SimpleCDM | 0.45 | 46% | 2% | 76.6% (±12.1) | -19.1 (9 lost, 0 gained) | 0.004 |

- The test is sane: with no dialogue the model scores 27.7% (chance is about 33%), and 95.7% on the original.
- Chat-style dialogue did **not** tolerate more compression than SQuAD. The curve is almost the same (0.8 free, 0.6 about -8.5, 0.45 about -23). The 0.3 row is damped by the 3-choice floor.
- SimpleCDM matched or beat LLMLingua-2 at keep 0.6, within noise.
- Caveat: DREAM dialogues are short and dense.

Run it with `python real_dream_llm_test.py 100` (needs `GROQ_API_KEY`; resumable; stops cleanly at the free rate limit and reports on the questions finished).

### Compression quality on real booking conversations, MultiWOZ (E22)

60 questions about MultiWOZ 2.2 booking conversations (median 198 words, with natural filler and repeated details). Each question is templated from human-annotated details, such as the restaurant's area; 30 have answers stated once and 30 stated 2+ times. Numeric answers are excluded because the guard protects number-like tokens by design.

| method | keep | words kept | correct | paired change vs original |
|---|---|---|---|---|
| original | 1.0 | 100% | 95.0% | |
| LLMLingua-2 | 0.8 | 86% | 91.7% | -3.3 (2 lost, 0 gained) |
| LLMLingua-2 | 0.6 | 63% | 86.7% | -8.3 (5 lost, 0 gained, p = 0.062) |
| LLMLingua-2 | 0.45 | 45% | 75.0% | -20.0 (12 lost, 0 gained, p < 0.001) |
| SimpleCDM | 0.6 | 60% | 81.7% | -13.3 (8 lost, 0 gained, p = 0.008) |

- The curve matches SQuAD and DREAM almost exactly (keep 0.6: -8.3, -8.5, -8.3). Filler and repetition did not make the text safe to compress hard.
- Answers stated 2+ times lost somewhat less than answers stated once in every row (for example -3.3 vs -13.3 at keep 0.6), but with 30 questions per group this is not significant.
- Informal pooling of the three real tests (LLMLingua-2, same answerer): keep 0.8 lost 4 and gained 1; keep 0.6 lost 14 and gained 0; keep 0.45 lost 34 and gained 0.
- FCCL's own plan for a logistics message is about keep 0.44 (the demo shows this). That level cost about 20 points here, so `min_keep=0.6` is doing real work.
- Caveats: one answerer model, templated questions, strict exact-match scoring, numeric answers excluded.

Run it with `python real_multiwoz_llm_test.py 60` (needs `GROQ_API_KEY`; resumable; stops cleanly at the free rate limit and reports on the questions finished).

### Does learning each user's preference help? PRISM (E19, E20)

PRISM has 1,389 users who rated several model responses to the same prompt on a 1 to 100 scale (median 48 rated responses per user). For each user, in order, we predicted which response in a turn they rated higher from response length alone, using only that user's earlier turns. Accuracy is the share of within-turn response pairs ordered correctly (50% is chance).

| method | accuracy | vs global rule |
|---|---|---|
| global rule (people prefer longer answers) | 59.2% | |
| user-only learning | 55.8% | -3.4 points, 95% interval [-4.1, -2.7] |
| user blended toward the global rule, k=3 / 10 / 30 | 58.0% / 59.1% / 59.2% | -1.2 / -0.1 / +0.0 |

- The user-only learner improves with history (53.6% after 2 to 3 turns, 57.1% after 11 or more) but never reaches the global rule.
- 81% of users rate longer answers higher. A user's length preference is only weakly stable across their own turns (correlation about +0.12).
- **Controls (E20).** Removing each model's average quality, and then also its usual length, did not change the conclusion: user-only learning stayed 2.8 to 3.0 points below global, and stability stayed at about +0.10. The global length slope halved (+11.7 to +5.9), so part of "longer is better" was model quality. Model identity alone predicts ratings as well as length does (59.2%).
- **By conversation type** the length preference is about the same (slopes +10.2 to +12.8; "longer wins" accuracy 58.3% to 60.8%).
- Caveats: this is one feature (length), a simple learner, about 12 turns per user, and about *output* length, not input compression. PRISM's conversation types are not FCCL's chatty and logistics contexts. Survey-stated preferences were not used.

Run it with `python inspect_prism.py`, `python prism_length_test.py` and `python prism_control_test.py`.

## Results so far (simulated users)

The simulations below use **invented** users, hidden tolerances and reactions that I wrote, and they assume the requested rate is honoured exactly. Read them as checks of how the mechanism behaves, not as evidence that it helps real people. The real-data tests above are the evidence.

100 fake users, 40 turns each, 5 seeds. Each fake user has a hidden amount of context they need kept. "Bad reaction" means the user corrected the answer. These simulations use the keyword detector, and they assume the requested rate is honoured exactly (which is not true for ScaleDown and only roughly true for LLMLingua-2).

### First test: do the weights help?

The context rule holds for 60% of the fake users. This table came from an earlier version of `fccl_sim.py`; the same settings are `ctx_strength=1.0` ("full"), `ctx_strength=0.0` or `use_context=False` ("no context"), and `learn=False` ("no learning").

| policy | tokens saved | bad reactions, first 10 turns | bad reactions, last 20 turns |
|---|---|---|---|
| no compression | 0.0% | 0.0% | 0.0% |
| fixed keep 0.7 | 30.0% | 10.6% | 10.1% |
| fixed keep 0.5 | 50.0% | 35.6% | 34.6% |
| fixed keep 0.3 | 70.0% | 66.5% | 66.7% |
| **FCCL full** | 36.7% | 12.2% | **3.2%** |
| FCCL, no context weight | 30.8% | 11.3% | 3.9% |
| FCCL, no learning | 41.9% | 18.0% | 18.4% |
| FCCL, neither | 35.0% | 15.4% | 15.8% |

- Fixed settings trade savings for bad reactions in a straight line. FCCL breaks that trade-off *in the simulation*.
- **Learning protects users** (in the simulation). Turning it off takes late bad reactions from 3.2% to 18.4%. The PRISM test above did not find a comparable benefit from learning on real people, so treat this as untested on humans.
- **The context weight mostly buys savings** (about 6 extra points with learning on), and on its own it is only a guess that needs correcting.
- FCCL is slightly worse than a fixed setting in the first few turns, the cost of learning each user from scratch.

### Stress test: what if the context rule is wrong?

Three simulated worlds: the pattern holds for 60% of users, there is no pattern, and the pattern is backwards for 60%.

| world | strength | tokens saved | bad reactions, first 10 turns | bad reactions, last 20 turns |
|---|---|---|---|---|
| pattern holds | 0.0 | 30.8% | 11.3% | 3.9% |
| | 0.25 | 33.2% | 10.6% | 3.7% |
| | 0.5 | 34.7% | 10.4% | 3.4% |
| | 1.0 | 36.7% | 12.2% | 3.2% |
| no pattern | 0.0 | 31.1% | 12.0% | 3.5% |
| | 0.25 | 32.2% | 13.0% | 3.8% |
| | 0.5 | 32.8% | 14.9% | 3.7% |
| | 1.0 | 33.3% | 18.6% | 3.8% |
| pattern backwards | 0.0 | 30.2% | 13.2% | 3.6% |
| | 0.25 | 30.0% | 15.1% | 4.1% |
| | 0.5 | 29.8% | 19.3% | 4.0% |
| | 1.0 | 29.1% | 25.8% | 4.5% |

- Learning corrects a wrong guess: late bad reactions stay between 3.2% and 4.5% in every world.
- A wrong guess is paid for early: at full trust in the backwards world, first-10-turn bad reactions double (25.8% vs 13.2%).
- The context weight is an unbalanced bet: about +6 points of savings when right, double the early bad reactions when wrong.
- A low trust level (0.25) hedges: +2.4 points of savings when the rule holds, at most about +2 points of early bad reactions when it does not.

### Learning the trust level

| world | policy | tokens saved | bad reactions, first 10 turns | bad reactions, last 20 turns |
|---|---|---|---|---|
| pattern holds | strength 0.0 | 30.8% | 11.3% | 3.9% |
| | strength 0.25 | 33.2% | 10.6% | 3.7% |
| | strength 1.0 | 36.7% | 12.2% | 3.2% |
| | **adaptive trust** | 34.0% | 11.3% | 3.6% |
| no pattern | strength 0.0 | 31.1% | 12.0% | 3.5% |
| | strength 0.25 | 32.2% | 13.0% | 3.8% |
| | strength 1.0 | 33.3% | 18.6% | 3.8% |
| | **adaptive trust** | 32.0% | 13.7% | 4.0% |
| pattern backwards | strength 0.0 | 30.2% | 13.2% | 3.6% |
| | strength 0.25 | 30.0% | 15.1% | 4.1% |
| | strength 1.0 | 29.1% | 25.8% | 4.5% |
| | **adaptive trust** | 30.0% | 14.9% | 3.9% |

- It never beats the best fixed trust level in every world, but it avoids the full-trust disaster when the rule is wrong and keeps most of the gain when the rule holds, without being told which world it is in.
- It learns which world it is in: with the pattern backwards it mostly uses trust 0.0 (70% of requests); with the pattern holding it leans to 0.5 and 1.0.

Run the current comparison with `python fccl_sim.py 5`.

### A hard minimum keep (`min_keep`)

Without a floor, full-trust FCCL plans below 0.45 (where the real tests measured about an 18 to 23 point accuracy loss) in 26%, 20% and 10% of compressions in the three worlds (adaptive trust: 8%, 3% and 2%).

| world | policy | tokens saved | bad % first 10 | bad % last 20 | plans below 0.45 |
|---|---|---|---|---|---|
| pattern holds | strength 1.0, no floor | 36.7% | 12.2% | 3.2% | 25.9% |
| | strength 1.0, min_keep 0.45 | 36.0% | 11.3% | 2.6% | 0.0% |
| | strength 1.0, min_keep 0.6 | 31.2% | 7.5% | 2.4% | 0.0% |
| no pattern | strength 1.0, no floor | 33.3% | 18.6% | 3.8% | 20.1% |
| | strength 1.0, min_keep 0.45 | 32.8% | 18.3% | 3.5% | 0.0% |
| | strength 1.0, min_keep 0.6 | 29.3% | 13.4% | 3.2% | 0.0% |
| pattern backwards | strength 1.0, no floor | 29.1% | 25.8% | 4.5% | 10.3% |
| | strength 1.0, min_keep 0.45 | 29.0% | 26.0% | 4.5% | 0.0% |
| | strength 1.0, min_keep 0.6 | 27.5% | 19.0% | 4.3% | 0.0% |

- A floor of 0.45 costs at most about 0.7 points of savings and slightly lowers bad reactions in two of three worlds. A floor of 0.3 does almost nothing (not shown). A floor of 0.6 trades about 5 points of savings for 5 to 7 fewer points of early bad reactions.
- The simulation shows the *cost* of the floor in savings. The real SQuAD, DREAM and MultiWOZ tests show quality loss already at 0.6 and a clear loss at 0.45, so the 0.6 rows are the relevant ones, and the original "knee at 0.45" reasoning does not hold.

Run it with `python floor_sim.py 5`.

## Conversation-type detector: keywords vs Laya (20 hand-labelled messages)

Laya is a small (421M parameter), open-source, non-autoregressive decision model from ConvAI Innovations. Given a message and a question, it picks a label. We ask whether the user is mainly venting (`chatty`), handling practical tasks (`logistics`) or neither (`other`), and map the label to a context weight.

| detector | correct on 20 messages |
|---|---|
| keyword rule | 8 / 20 |
| Laya | 18 / 20 |

- The keyword rule only fires on its listed words, so it missed venting without emotion words ("I can't believe my boss did that again, I'm done with this place") and practical requests without our words ("remind me to pick up the parcel"). It also read "border" as "order".
- Laya got those right, and also a mixed message ("I'd order a pizza but I'm too sad to even move"). Its two misses were "lol that movie was wild" (called `other`) and "Tell me about the history of Rome" (called `chatty`).
- **This comparison is tilted:** I wrote the messages to avoid the keywords, and I wrote the labels. Treat 8/20 as a worst case for keywords, not an expected score, and treat the sample size (20) as too small for real conclusions.
- Laya's confidence values are uncalibrated (its own warning says so) and don't cleanly separate right from wrong answers, so `LayaContext` does not gate on them by default (`min_conf=0.0`).
- This tests label accuracy only. It does not show that keeping more context for venting is the right policy for real users.

Run it with `python compare_context.py`.

## Real compressors

Three compressors were tried on hand-written texts. Details and every table are in [`results.md`](results.md).

| | ScaleDown (API) | LLMLingua-2 (local, CPU) | SimpleCDM (local toy) |
|---|---|---|---|
| cost | prepaid credit; free allowance on signup (1M tokens at the time of testing) | free | free |
| follows the requested rate | no: below about 50% requested removal the output is *longer* than the input | within a few points | exactly |
| protecting words | no option in the documented API | forced tokens work | only pure digits (drops "12th") |
| style of output | whole chunks, repeated passages | individual words dropped | individual words dropped |
| uses the user's question | yes | no | no |

**ScaleDown calibration** (one 511-token text, removal 10% to 90%, `calibrate_scaledown.py`):

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

It drops specific details without warning, its token counts don't always match the text it returns (so we measure the text ourselves), and it is billed by input size (about 510 tokens per call on this text).

**Blind quality tests on three texts** (support message A, technical help request B, casual vent C). Each text was compressed to three sizes; an LLM (me) answered six questions per sample without knowing which method or rate produced it, and graded the answers against the original. Scores are out of 6, with the original at 6.0 on every text:

| keep | method | text A | text B | text C | mean |
|---|---|---|---|---|---|
| 0.6 | LLMLingua-2 | 6.0 | 5.65 | 6.0 | 5.9 |
| 0.6 | SimpleCDM | 6.0 | 5.85 | 6.0 | 5.95 |
| 0.44 | LLMLingua-2 | 5.75 | 5.55 | 5.8 | 5.7 |
| 0.44 | SimpleCDM | 5.4 | 5.35 | 5.4 | 5.4 |
| 0.3 | LLMLingua-2 | 5.1 | 4.1 | 4.5 | 4.6 |
| 0.3 | SimpleCDM | 4.6 | 4.2 | 4.7 | 4.5 |
| ScaleDown, 50% removal (text A only) | | 4.5 | | | |

> **These blind tests were too lenient and their main conclusion did not survive.** They suggested a knee near keep 0.44 with near-lossless compression above it. The later real tests (SQuAD, DREAM and MultiWOZ, above) found a steady decline with a clear loss at 0.45, about 8 points at 0.6, and no free level below 0.8. The likely reasons: the judge could reconstruct meaning from short text, had the ground truth, and the questions rewarded keeping keywords. Keep the findings that still hold: the two local methods are within noise of each other, ScaleDown fails by dropping whole passages, and what differs between methods is mainly how they *fail*.

- The two local methods are within noise of each other. A toy word-dropper matching LLMLingua-2 suggests the test is not discriminating enough to rank compressors. The later real tests mostly agree: SimpleCDM matched or beat LLMLingua-2 at keep 0.6 on SQuAD and DREAM, within noise. On MultiWOZ it was worse (-13.3 vs -8.3), though the two were not tested directly against each other.
- ScaleDown kept the most words (82%) and still scored lowest, losing whole paragraphs and repeating others, though it kept the four requests.
- The guard and the protected tokens matter more than which compressor is used.

End to end (`demo_pipeline.py`, Laya + FCCL + compressor, same 511-token context; the demo uses `min_keep=0.6`, so the exact plans below come from the earlier 0.45 setting and may differ slightly):

| case | plan (keep) | LLMLingua-2 | ScaleDown |
|---|---|---|---|
| venting message | 0.79 | kept 77% of words, accepted | skipped: below its useful range |
| logistics message | 0.44 | kept 42% of words, accepted, all IDs survived | compressed, then rejected by the guard (`X540` lost): no savings |
| tiny context | not compressed | unchanged | unchanged |

## Limits

- The learning results in the simulation sections use simulated users. I wrote their hidden tolerances and reactions, and the simulations assume the requested rate is honoured exactly. The first real test of per-user learning (PRISM, answer length) did not support it.
- In the first test, 60% of the fake users follow the pattern the context rule assumes, so the context weight's benefit there is probably overstated. The stress test checks what happens when it is wrong.
- Adaptive trust uses one trust level shared by all users, while real populations are mixed. The penalty for a bad reaction (1.0) is a number I chose, not a measured one.
- The simulations use the keyword detector. Laya is wired in as an option but has not been run end to end inside the simulation or on real conversations.
- The detector comparison is 20 messages that I wrote and labelled.
- **The real quality curves come from three datasets, with one answerer model (`gpt-oss-20b`).** SQuAD is single-fact lookup (60 questions, close to a worst case for compression); DREAM is dialogue inference (47 questions, short and dense); MultiWOZ is booking conversations with natural filler and repetition (60 questions, templated from human annotations, numeric answers excluded, strict scoring). All three give nearly the same curve. Venting-style text is untested and might tolerate more compression. The 0.6 loss is borderline significant within each dataset and clear when they are pooled informally.
- **The hand-judged blind tests were too lenient** (one model as both answerer and judge, with ground truth). Their "0.45 knee" is rejected and should not be relied on.
- **PRISM tests output length, not input compression,** with one feature, a simple learner and about 12 turns per user. A negative result there does not rule out other kinds of per-user adaptation, other features, longer histories or survey-based starting guesses (untested).
- The guard checks protected words and length, not meaning: a compressed text can keep every protected word and still lose what the user needs.
- LLMLingua-2 is query-agnostic (it ignores what the user asked), so it can keep irrelevant content and drop relevant content. Compression that uses the user's question has not been tested yet.
- Laya needs about 1 GB of free RAM on top of `torch`, and LLMLingua-2 about the same. Both failed with out-of-memory errors on a laptop until other apps were closed and the machine was restarted.
- "No reaction" after a harder experiment is treated as weak evidence of satisfaction. That is an assumption.
- Simulation results are averages over 5 seeds. Small differences between rows are not meaningful.

## Earlier study: answer length (verbosity)

`sim.py`, `policies.py` and `experiments.py` are an earlier study of choosing answer length per user. Headline result on simulated users: a personalized policy saved about 22% of tokens and raised satisfaction, and a version with a risk constraint kept dissatisfaction within 0.3 points of the default even when feedback was weak, at the cost of smaller savings. Its ideas (per-user beliefs, safe exploration) feed into FCCL. Note that the later PRISM test on real people found that a global answer-length rule beat per-user learning at about 12 turns per user.

## Files

| file | what it is |
|---|---|
| `fccl.py` | the layer: data types (`Request`, `CompressionPlan`, `Feedback`) and the `FCCL` class |
| `laya_context.py` | `LayaContext`: uses Laya as the conversation-type detector (pass it as `context_fn`) |
| `cdm.py` | CDM backends (`LLMLinguaCDM`, `ScaleDownCDM`, `SimpleCDM`) and `apply_plan` with the safety guard |
| `fccl_sim.py` | tests FCCL against simulated users in three worlds |
| `floor_sim.py` | the cost of a hard minimum keep (`min_keep`) in the simulation |
| `real_curve_sim.py` | simulation using the hand-judged quality curve (that curve is now known to be too optimistic) |
| `compare_context.py` | keyword rule vs Laya on 20 hand-labelled messages |
| `calibrate_scaledown.py`, `test_scaledown.py`, `test_scaledown2.py` | ScaleDown checks and calibration |
| `demo_pipeline.py` | Laya + FCCL + a compressor end to end on three cases |
| `fact_check.py` | which facts survive compression (a weak, keyword-based check) |
| `make_blind_samples.py`, `make_blind_samples2.py` | build shuffled, unlabelled compressed samples for the blind quality tests |
| `test_llmlingua.py` | first LLMLingua-2 test |
| `real_squad_test.py` | real SQuAD with an extractive reader (strict) |
| `real_squad_llm_test.py` | real SQuAD with a generative LLM answerer and LLM judge (Groq, resumable) |
| `paired_analysis.py` | paired significance analysis of the SQuAD LLM results |
| `inspect_dream.py`, `real_dream_llm_test.py` | DREAM dialogues: structure check, and the multiple-choice compression test (Groq, resumable) |
| `inspect_multiwoz.py`, `real_multiwoz_llm_test.py` | MultiWOZ booking conversations: structure check, and the compression test with questions templated from human annotations (Groq, resumable) |
| `inspect_prism.py`, `prism_length_test.py`, `prism_control_test.py` | PRISM: structure, per-user length adaptation, model and conversation-type controls |
| `sim.py`, `policies.py`, `experiments.py` | the earlier answer-length study |
| `results.md` | the full log of every experiment, with caveats |

## Run it

```
pip install numpy matplotlib
python fccl_sim.py 5          # FCCL vs fixed settings in three worlds (keyword detector)
python floor_sim.py 5         # what a hard minimum keep costs
python experiments.py 3       # earlier answer-length study

pip install laya              # needs Python 3.10+ and roughly 1 GB free RAM
python laya_context.py        # Laya -> context weights on 3 sample messages
python compare_context.py     # keyword rule vs Laya

pip install llmlingua         # local compressor; run on the CPU (device_map="cpu")
python cdm.py                 # the free local compressor on a sample text
python demo_pipeline.py local   # whole pipeline with the free local compressor
python demo_pipeline.py lingua  # whole pipeline with LLMLingua-2 (about 1-2 GB free RAM)
```

Real-data tests (need `pip install datasets huggingface_hub requests numpy`; the SQuAD, DREAM and MultiWOZ LLM tests need a free Groq key):

```
setx GROQ_API_KEY "your-key"          # Windows; open a new terminal afterwards
python real_squad_llm_test.py 60      # about 30 minutes; saves progress, safe to rerun
python paired_analysis.py             # significance of the drops, no API calls
python inspect_dream.py               # look at the DREAM data
python real_dream_llm_test.py 100     # dialogues; stops cleanly at the free rate limit, rerun to continue
python inspect_multiwoz.py            # look at the MultiWOZ data
python real_multiwoz_llm_test.py 60   # booking conversations; resumable
python inspect_prism.py               # look at the PRISM data
python prism_length_test.py           # per-user length adaptation vs a global rule
python prism_control_test.py          # same, with model and conversation-type controls
```

To use the real ScaleDown API, get a key from ScaleDown and keep it in an environment variable, never in a file in the repo:

```
setx SCALEDOWN_API_KEY "your-key"     # Windows; open a new terminal afterwards
python demo_pipeline.py               # uses about 500 free tokens per compressed case
python calibrate_scaledown.py         # 9 calls, about 4,600 free tokens
```

Use the pieces together:

```python
from fccl import FCCL
from laya_context import LayaContext

layer = FCCL(context_fn=LayaContext(), min_keep=0.6)
```

## Roadmap

- [x] Data types and first decision logic (context weight, satisfaction weight)
- [x] Careful experimenting with safety guards
- [x] Learn how much to trust the context rule instead of using a fixed `ctx_strength`
- [x] Replace the keyword detector with a real classifier (Laya, optional `context_fn`)
- [x] Connect real compressors (ScaleDown, LLMLingua-2) behind a common interface, with a safety guard
- [x] Stop protecting sentence-initial capitalised words (they caused needless fallbacks)
- [x] Hard minimum keep (`min_keep`) with its cost measured in the simulation
- [x] Measure compression quality on real questions with a real LLM (SQuAD, E17 and E18): no knee, 0.8 free, 0.6 about -8 points
- [x] First real-data test of per-user adaptation (PRISM, E19 and E20): not supported for answer length
- [x] Test on dialogue context (DREAM, E21, partial): same curve as SQuAD
- [x] Test filler-heavy booking conversations (MultiWOZ, E22): same curve as SQuAD and DREAM
- [ ] Question-aware compression: keep the parts of the context most relevant to the user's prompt, and compare with the question-blind curve on the same questions
- [ ] Test venting-style text, and use more answerer models
- [ ] Raise the default `min_keep` in `fccl.py` from 0.1 to a measured-safe value (0.6)
- [ ] Rerun the simulation on the measured quality curve (users would still be invented, so label it a sim)
- [ ] Test whether survey-stated preferences (PRISM) give a useful starting guess for a new user, and try features other than length
- [ ] Test the context weight on real chatty vs logistics conversations
- [ ] Make the plan aware of each backend's useful range (a choice between "none" and "at least X")
- [ ] Decide what to do when the guard rejects a result (retry at a gentler rate, or another backend)
- [ ] Test the detector on more messages with labels I didn't write
- [ ] Small human study of satisfaction with compressed contexts
- [ ] Decide, from the above, whether to keep per-user learning or reduce FCCL to a guarded compression layer with a strong default
- [ ] Add `where` (which parts of the context may be compressed) and `keep_original` (reversible compression) to the plan

## Changelog

- Data types and the first decision logic (context weight, satisfaction weight)
- Careful experimenting with safety guards
- Simulation test against fixed settings
- Ablation: learning protects users, the context weight buys savings (in the simulation)
- Stress test with three worlds and a `ctx_strength` dial for trusting the context rule
- Adaptive trust: FCCL learns how much to trust the context rule
- Laya as an optional conversation-type detector (`context_fn`); keyword vs Laya comparison on 20 hand-labelled messages (8/20 vs 18/20)
- CDM backends (ScaleDown API, local stand-in) with a guard that falls back to the original text; ScaleDown calibrated on one text (useful from about 50% removal, drops details); end-to-end demo
- LLMLingua-2 backend running on the CPU; blind quality tests on three texts (later found too lenient)
- `extract_preserve` no longer protects sentence-initial words; `min_keep` floor added and its cost measured in the simulation
- Experiment log moved to `results.md`
- Real SQuAD tests: extractive reader (E17) and generative LLM answerer with judge (E18), with a paired analysis. Quality falls steadily; no knee at 0.45; keep 0.8 is free, 0.6 about -8 points
- PRISM tests (E19, E20): per-user length adaptation did not beat a global rule, even after controlling for model quality and conversation type
- README rewritten to match the real-data evidence; demo and usage snippet now use `min_keep=0.6`
- Real DREAM dialogue test (E21, partial): the compression curve matches SQuAD; dense dialogues did not tolerate more compression
- Real MultiWOZ booking-conversation test (E22): same curve as SQuAD and DREAM; filler and repetition did not make text safe to compress hard