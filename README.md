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

Two weights drive each decision:

- **Context weight:** how compressible this kind of conversation is. Venting or chatting keeps more of the context. Logistics compresses more. This is the starting guess (currently simple keyword rules, to be replaced by a classifier).
- **Satisfaction weight:** one number per user, updated from reactions to the answer ("show more", correction, regenerate, thumbs). Unhappy users get compressed less. Reactions to the *answer* count, not general mood, so a user venting is not treated as dissatisfied.

The layer also experiments carefully: it sometimes compresses a little harder for users who seem happy, never during a user's first few requests, never for unhappy users, and never below a floor. It remembers which plans were experiments so a bad reaction can be traced back to the experiment.

The plan it produces has three fields so far: `apply` (compress at all?), `rate` (fraction of the text to keep) and `preserve` (words that must survive).

## Results so far (simulated users)

100 fake users, 40 turns each, 5 seeds. "Bad reaction" means the user corrected the answer. Run with `python fccl_sim.py 5`.

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

What it shows:
- Fixed settings trade savings for bad reactions in a straight line. FCCL breaks that trade-off.
- **Learning protects users.** Turning it off takes late bad reactions from 3.2% to 18.4%.
- **The context weight mostly buys savings** (about 6 extra points with learning on), and on its own it is only a guess that needs correcting.
- FCCL is slightly worse than a fixed setting in the first few turns, the cost of learning each user from scratch.

## Limits

- All users are simulated. I wrote their hidden tolerances and reactions.
- 60% of the fake users follow the pattern the context rule assumes, so the context weight's benefit is probably overstated.
- The conversation-type detector is a keyword list, not a real classifier.
- No real CDM is connected yet. `rate` is our own convention (fraction kept) and still has to be mapped to the CDM's own setting.
- "No reaction" after a harder experiment is treated as weak evidence of satisfaction. That is an assumption.

## Earlier study: answer length (verbosity)

`sim.py`, `policies.py` and `experiments.py` are an earlier study of choosing answer length per user. Headline result on simulated users: a personalized policy saved about 22% of tokens and raised satisfaction, and a version with a risk constraint kept dissatisfaction within 0.3 points of the default even when feedback was weak, at the cost of smaller savings. Its ideas (per-user beliefs, safe exploration) feed into FCCL.

## Files

| file | what it is |
|---|---|
| `fccl.py` | the layer: data types (`Request`, `CompressionPlan`, `Feedback`) and the `FCCL` class |
| `fccl_sim.py` | tests FCCL against simulated users and runs the ablation |
| `sim.py`, `policies.py`, `experiments.py` | the earlier answer-length study |

## Run it

```
pip install numpy matplotlib
python fccl_sim.py 5        # FCCL vs fixed settings, with ablations
python experiments.py 3     # earlier answer-length study
```

## Roadmap

- [ ] Replace the keyword detector with a real classifier
- [ ] Connect a real CDM and map `rate` to its settings
- [ ] Add `where` (which parts of the context may be compressed) and `keep_original` (reversible compression) to the plan
- [ ] Test on real preference data, then a small human study
- [ ] Merge the answer-length policies into the layer

## Changelog

- Data types and the first decision logic (context weight, satisfaction weight)
- Careful experimenting with safety guards
- Simulation test against fixed settings
- Ablation: learning protects users, the context weight buys savings