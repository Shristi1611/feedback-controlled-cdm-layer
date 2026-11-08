"""Quickstart: the whole FCCL loop with no API key, no model download and no internet.

    pip install -e .          (once, from the repo root)
    python examples/quickstart.py

A fake "model" stands in for your LLM. It answers only if the fact it needs is still in the text, and says
"unknown" if the compressor dropped it. That is enough to show the loop:
plan -> compress (with the guard) -> ask -> retry with the original on "unknown" -> feedback.
"""
from fccl import FCCL, Request, Feedback
from cdm import SimpleCDM, answer_with_retry, ABSTAIN_INSTRUCTION

CONTEXT = (
    "Notes from the move: the landlord said the spare key is under the green flowerpot by the gate, "
    "the plumber comes on Thursday morning to fix the kitchen tap, and the electricity meter number is 70418. "
    "Please remember that the bins go out on Tuesday evenings and that parking is allowed only on the left side. "
    "The neighbours were friendly and mentioned a nice bakery two streets away that opens early on weekends."
)


def make_fake_model(needed_words, answer):
    """Stands in for your LLM call. In real use, `ask` sends the text plus ABSTAIN_INSTRUCTION to your model."""
    def ask(text):
        return answer if all(w in text.lower() for w in needed_words) else "unknown"
    return ask


# min_tokens=0 so this short demo text is compressed; the default skips contexts under 500 tokens.
layer = FCCL(min_tokens=0, seed=0)
compressor = SimpleCDM()          # free local stand-in; LLMLingua-2 (cdm.LLMLinguaCDM) is the real one
user = "demo-user"

questions = [
    ("Where is the spare key?",       ["flowerpot"],         "Under the green flowerpot."),
    ("When does the plumber come?",   ["thursday", "plumber"], "Thursday morning."),
    ("What is the meter number?",     ["70418"],             "70418."),
    # This fake model needs the exact wording, which the compressor shortens, so it says "unknown" and FCCL retries.
    ("Which side can I park on?",     ["allowed only on the left side"], "The left side only."),
]

print("The instruction your prompt must include so the model can say 'unknown':")
print("  ", ABSTAIN_INSTRUCTION, "\n")

for question, needed, answer in questions:
    plan = layer.plan(Request(user, question, context_tokens=len(CONTEXT.split())))
    result = answer_with_retry(make_fake_model(needed, answer), compressor, CONTEXT, plan, prompt=question)

    print(f"Q: {question}")
    print(f"   plan: keep {plan.rate:.0%} of the text   sent: {result.cdm.kept_words:.0%} of the words"
          f"   guard fell back: {result.cdm.fell_back}")
    print(f"   compressed text: {result.cdm.text}")
    print(f"   answer: {result.answer}   retried with the original: {result.retried}   cost: {result.cost:.2f}x")

    if result.abstained:                                  # the automatic feedback signal
        layer.feedback(Feedback(user, "abstained"))
        print(f"   -> the model said 'unknown'; FCCL lowered this user's weight to {layer.sat_weight[user]:.2f}")
    print()

print("Cost is a multiple of the full text's words: below 1.00x is a saving, above 1.00x means a retry was needed.")