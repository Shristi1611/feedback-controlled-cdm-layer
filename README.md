# FCCL: Feedback-Controlled CDM Layer

A feedback-driven layer between users and CDMs that decides when and how hard to compress, without hurting answer quality.

**Status:** early prototype. The layer's learning is tested on *simulated* users only, so those numbers show that the mechanism works, not how real people would react. The conversation-type detector (Laya) and two real compressors (ScaleDown, LLMLingua-2) have been tried on three hand-written texts. Nothing has been tested on real users. The full experiment log is in [`results.md`](results.md).

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
- **Hard minimum (`min_keep`):** no plan ever keeps less than this fraction of the text. The default is 0.1 (the old behaviour, so earlier simulation results stay reproducible). Blind quality tests suggest quality drops below about 0.45, so use `min_keep=0.45` with a real compressor.

The plan it produces has three fields so far: `apply` (compress at all?), `rate` (fraction of the text to keep) and `preserve` (words that must survive: numbers, and capitalised words that are not at the start of a sentence).

### The CDM side (`cdm.py`)

Backends share one interface: `LLMLinguaCDM` (local, CPU), `ScaleDownCDM` (the real API) and `SimpleCDM` (a free word-dropper, used as a stand-in). `apply_plan()` runs a backend and then a **guard**:

- protected words (from the user's message) and ID-like tokens in the context (numbers, codes such as order or part numbers) must survive, or the original text is used instead;
- if the output is not actually shorter than the original, the original is used;
- it reports the *measured* kept fraction, because the requested rate is only approximately honoured.

`ScaleDownCDM` skips the API call when the requested removal is below 50% (`min_removal`), because calibration showed it makes text longer there. `LLMLinguaCDM` passes the protected words to LLMLingua-2 as forced tokens.

## Results so far

### Learning, on simulated users

100 fake users, 40 turns each, 5 seeds. Each fake user has a hidden amount of context they need kept. "Bad reaction" means the user corrected the answer. These simulations use the keyword detector, and they assume the requested rate is honoured exactly (which is not true for ScaleDown and only roughly true for LLMLingua-2).

#### First test: do the weights help?

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

- Fixed settings trade savings for bad reactions in a straight line. FCCL breaks that trade-off.
- **Learning protects users.** Turning it off takes late bad reactions from 3.2% to 18.4%.
- **The context weight mostly buys savings** (about 6 extra points with learning on), and on its own it is only a guess that needs correcting.
- FCCL is slightly worse than a fixed setting in the first few turns, the cost of learning each user from scratch.

#### Stress test: what if the context rule is wrong?

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

#### Learning the trust level

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

#### A hard minimum keep (`min_keep`)

Without a floor, full-trust FCCL plans below the 0.45 quality knee (see the blind tests below) in 26%, 20% and 10% of compressions in the three worlds (adaptive trust: 8%, 3% and 2%).

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
- The simulation shows the *cost* of the floor. Whether 0.45 is the right number rests on the blind tests below.

Run it with `python floor_sim.py 5`.

### Conversation-type detector: keywords vs Laya (20 hand-labelled messages)

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

### Real compressors

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

- **There is a knee near keep 0.44.** Cutting about 40% of the words is close to lossless on all three texts; the cost is small at 0.44 and about 1.5 points at 0.3. What is lost first are relations (what a fix did, who did what to whom) and dates.
- **The two local methods are within noise of each other.** A toy word-dropper matching LLMLingua-2 suggests the test is not discriminating enough to rank compressors (the questions reward keeping keywords, and the judge can reconstruct meaning from short text).
- ScaleDown kept the most words (82%) and still scored lowest, losing whole paragraphs and repeating others, though it kept the four requests.
- What differs between methods is how they *fail*: the guard and the protected tokens matter more than which compressor is used.

End to end (`demo_pipeline.py`, Laya + FCCL + compressor, same 511-token context):

| case | plan (keep) | LLMLingua-2 | ScaleDown |
|---|---|---|---|
| venting message | 0.79 | kept 77% of words, accepted | skipped: below its useful range |
| logistics message | 0.44 | kept 42% of words, accepted, all IDs survived | compressed, then rejected by the guard (`X540` lost): no savings |
| tiny context | not compressed | unchanged | unchanged |

## Limits

- The learning results use simulated users. I wrote their hidden tolerances and reactions, and the simulations assume the requested rate is honoured exactly.
- In the first test, 60% of the fake users follow the pattern the context rule assumes, so the context weight's benefit there is probably overstated. The stress test checks what happens when it is wrong.
- Adaptive trust uses one trust level shared by all users, while real populations are mixed. The penalty for a bad reaction (1.0) is a number I chose, not a measured one.
- The simulations use the keyword detector. Laya is wired in as an option but has not been run end to end inside the simulation or on real conversations.
- The detector comparison is 20 messages that I wrote and labelled.
- **The compressor findings come from three texts of about 300 to 360 words.** I wrote them. Longer contexts, other languages and other styles may behave differently.
- **The quality test has one model as both answerer and judge,** and the judge had the ground truth when grading. Differences under about 0.5 points are noise. It tests whether facts survive, not reasoning.
- The 0.45 quality knee behind `min_keep=0.45` comes from those tests, so it is a starting point, not a validated threshold.
- The guard checks protected words and length, not meaning: a compressed text can keep every protected word and still lose what the user needs.
- LLMLingua-2 is query-agnostic (it ignores what the user asked), so it can keep irrelevant content and drop relevant content.
- Laya needs about 1 GB of free RAM on top of `torch`, and LLMLingua-2 about the same. Both failed with out-of-memory errors on a laptop until other apps were closed and the machine was restarted.
- "No reaction" after a harder experiment is treated as weak evidence of satisfaction. That is an assumption.
- Results are averages over 5 seeds of a simulation. Small differences between rows are not meaningful.

## Earlier study: answer length (verbosity)

`sim.py`, `policies.py` and `experiments.py` are an earlier study of choosing answer length per user. Headline result on simulated users: a personalized policy saved about 22% of tokens and raised satisfaction, and a version with a risk constraint kept dissatisfaction within 0.3 points of the default even when feedback was weak, at the cost of smaller savings. Its ideas (per-user beliefs, safe exploration) feed into FCCL.

## Files

| file | what it is |
|---|---|
| `fccl.py` | the layer: data types (`Request`, `CompressionPlan`, `Feedback`) and the `FCCL` class |
| `laya_context.py` | `LayaContext`: uses Laya as the conversation-type detector (pass it as `context_fn`) |
| `cdm.py` | CDM backends (`LLMLinguaCDM`, `ScaleDownCDM`, `SimpleCDM`) and `apply_plan` with the safety guard |
| `fccl_sim.py` | tests FCCL against simulated users in three worlds |
| `floor_sim.py` | the cost of a hard minimum keep (`min_keep`) in the simulation |
| `compare_context.py` | keyword rule vs Laya on 20 hand-labelled messages |
| `calibrate_scaledown.py` | requested vs actual savings from ScaleDown on one text |
| `demo_pipeline.py` | Laya + FCCL + a compressor end to end on three cases |
| `fact_check.py` | which facts survive compression (a weak, keyword-based check) |
| `make_blind_samples.py`, `make_blind_samples2.py` | build shuffled, unlabelled compressed samples for the blind quality tests |
| `test_llmlingua.py` | first LLMLingua-2 test |
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

layer = FCCL(context_fn=LayaContext(), min_keep=0.45)
```

## Roadmap

- [x] Data types and first decision logic (context weight, satisfaction weight)
- [x] Careful experimenting with safety guards
- [x] Learn how much to trust the context rule instead of using a fixed `ctx_strength`
- [x] Replace the keyword detector with a real classifier (Laya, optional `context_fn`)
- [x] Connect real compressors (ScaleDown, LLMLingua-2) behind a common interface, with a safety guard
- [x] Stop protecting sentence-initial capitalised words (they caused needless fallbacks)
- [x] Hard minimum keep (`min_keep`) based on the quality knee, with its cost measured in the simulation
- [ ] Make the simulation use the real compressors' measured quality instead of invented tolerances
- [ ] Repeat the quality test with a harder question set and a judge that did not write the texts; use more and longer texts, and other languages
- [ ] Make the plan aware of each backend's useful range (a choice between "none" and "at least X")
- [ ] Decide what to do when the guard rejects a result (retry at a gentler rate, or another backend)
- [ ] Test the detector on more messages with labels I didn't write, and run the simulation with it
- [ ] Per-user trust level instead of one shared level
- [ ] Add `where` (which parts of the context may be compressed) and `keep_original` (reversible compression) to the plan
- [ ] Test on real preference data, then a small human study
- [ ] Merge the answer-length policies into the layer

## Changelog

- Data types and the first decision logic (context weight, satisfaction weight)
- Careful experimenting with safety guards
- Simulation test against fixed settings
- Ablation: learning protects users, the context weight buys savings
- Stress test with three worlds and a `ctx_strength` dial for trusting the context rule
- Adaptive trust: FCCL learns how much to trust the context rule
- Laya as an optional conversation-type detector (`context_fn`); keyword vs Laya comparison on 20 hand-labelled messages (8/20 vs 18/20)
- CDM backends (ScaleDown API, local stand-in) with a guard that falls back to the original text; ScaleDown calibrated on one text (useful from about 50% removal, drops details); end-to-end demo
- LLMLingua-2 backend running on the CPU; blind quality tests on three texts (knee near keep 0.44; local methods within noise of each other)
- `extract_preserve` no longer protects sentence-initial words; `min_keep` floor added and its cost measured in the simulation
- Experiment log moved to `results.md`