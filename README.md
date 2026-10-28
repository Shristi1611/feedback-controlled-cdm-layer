# FCCL: Feedback-Controlled CDM Layer

A feedback-driven layer between users and CDMs that decides when and how hard to compress, without hurting answer quality.

**Status:** early prototype. Everything so far is tested on *simulated* users only, so the numbers show that the mechanism works, not how real people would react.

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
      [ CDM ]  compresses the context
        |
      [ LLM ]  answers
        |
   answer --> user --> reaction goes back to FCCL
```

## How it works

Each decision combines two weights and a safety limit on experimenting:

- **Context weight:** how compressible this kind of conversation is. Venting or chatting keeps more of the context. Logistics compresses more. This is a starting guess (currently simple keyword rules, to be replaced by a classifier). How much the layer trusts this guess is controlled by `ctx_strength` (0 = ignore it, 1 = trust it fully), and with `adaptive_trust=True` the layer learns the right level itself.
- **Satisfaction weight:** one number per user, updated from reactions to the answer ("show more", correction, regenerate, thumbs). Unhappy users get compressed less. Reactions to the *answer* count, not general mood, so a user venting is not treated as dissatisfied.
- **Careful experimenting:** the layer sometimes compresses a little harder for users who seem happy. It never experiments during a user's first few requests, never for unhappy users, and never below a floor. It remembers which plans were experiments, so a bad reaction can be traced back to the experiment and punished harder.
- **Adaptive trust:** the layer keeps a score for each trust level (0, 0.25, 0.5, 1.0): text saved, minus a penalty for each bad reaction. It mostly uses the best-scoring level, tries a random one 15% of the time, and ties go to the lower (safer) level.

The plan it produces has three fields so far: `apply` (compress at all?), `rate` (fraction of the text to keep) and `preserve` (words that must survive).

## Results so far (simulated users)

100 fake users, 40 turns each, 5 seeds. Each fake user has a hidden amount of context they need kept. "Bad reaction" means the user corrected the answer.

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

- Fixed settings trade savings for bad reactions in a straight line. FCCL breaks that trade-off.
- **Learning protects users.** Turning it off takes late bad reactions from 3.2% to 18.4%.
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

## Limits

- All users are simulated. I wrote their hidden tolerances and reactions.
- In the first test, 60% of the fake users follow the pattern the context rule assumes, so the context weight's benefit there is probably overstated. The stress test checks what happens when it is wrong.
- Adaptive trust uses one trust level shared by all users, while real populations are mixed. The penalty for a bad reaction (1.0) is a number I chose, not a measured one.
- The conversation-type detector is a keyword list, not a real classifier.
- No real CDM is connected yet. `rate` is our own convention (fraction kept) and still has to be mapped to the CDM's own setting.
- "No reaction" after a harder experiment is treated as weak evidence of satisfaction. That is an assumption.
- Results are averages over 5 seeds of a simulation. Small differences between rows are not meaningful.

## Earlier study: answer length (verbosity)

`sim.py`, `policies.py` and `experiments.py` are an earlier study of choosing answer length per user. Headline result on simulated users: a personalized policy saved about 22% of tokens and raised satisfaction, and a version with a risk constraint kept dissatisfaction within 0.3 points of the default even when feedback was weak, at the cost of smaller savings. Its ideas (per-user beliefs, safe exploration) feed into FCCL.

## Files

| file | what it is |
|---|---|
| `fccl.py` | the layer: data types (`Request`, `CompressionPlan`, `Feedback`) and the `FCCL` class |
| `fccl_sim.py` | tests FCCL against simulated users in three worlds |
| `sim.py`, `policies.py`, `experiments.py` | the earlier answer-length study |

## Run it

```
pip install numpy matplotlib
python fccl_sim.py 5        # FCCL vs fixed settings in three worlds
python experiments.py 3     # earlier answer-length study
```

## Roadmap

- [x] Data types and first decision logic (context weight, satisfaction weight)
- [x] Careful experimenting with safety guards
- [x] Learn how much to trust the context rule instead of using a fixed `ctx_strength`
- [ ] Replace the keyword detector with a real classifier
- [ ] Check ScaleDown API access (a free tier is unconfirmed), connect a real CDM and map `rate` to its settings
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