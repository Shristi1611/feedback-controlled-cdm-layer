"""Offline check of the cascade wiring (no API calls): a fake model answers 'unknown' unless the key words survive."""
from fccl import FCCL, Request, Feedback, CompressionPlan
from cdm import SimpleCDM, answer_with_retry, is_abstention

assert is_abstention("Unknown.") and is_abstention("") and is_abstention("  unknown ") and not is_abstention("Friday")
for numeric in ("1998", "42%", "1,200", "$3.5 million", "1990s"):
    assert not is_abstention(numeric), numeric      # regression: bare numbers were treated as "unknown"
text = ("Hi, I wanted to check on the order I placed last week. The invoice number is 4821 and it was supposed "
        "to be delivered to the address on Park Street by Friday. I haven't received any tracking update, and the "
        "meeting with Rahul about the delivery schedule is tomorrow at 4 PM, so I really need to know where it is.")
calls = []
def ask(t):
    calls.append(len(t.split()))
    return "Friday" if "Friday" in t else "unknown"

fc = FCCL(min_tokens=0, seed=0)
assert fc.min_keep == 0.6
neutral = "What city is the capital of France?"
assert FCCL(min_tokens=0).plan(Request("n", neutral, 500)).rate == 0.65          # default strength 0.7
assert FCCL(min_tokens=0, compress_strength=0.8).plan(Request("n", neutral, 500)).rate == 0.6   # neutral plans 0.6
p = fc.plan(Request("u1", text, 500))
print("plan:", p)
for rate in (0.9, 0.6, 0.3):
    calls.clear()
    plan = CompressionPlan(True, rate, ["4821", "Rahul", "PM"])
    r = answer_with_retry(ask, SimpleCDM(), text, plan)
    print(rate, "->", r.answer, "retried" if r.retried else "no retry", f"cost {r.cost:.2f}", "calls", len(calls))
    if r.abstained:
        before = fc.sat_weight.get("u1", 1.0)
        fc.feedback(Feedback("u1", "abstained"))
        print("   weight", before, "->", round(fc.sat_weight["u1"], 3))
# the retry path: force an abstention
calls.clear()
r = answer_with_retry(lambda t: "unknown" if len(t.split()) < 50 else "Friday", SimpleCDM(), text,
                      CompressionPlan(True, 0.3, ["4821"]))
assert r.retried and r.answer == "Friday" and r.cost > 1.0 and r.abstained, r
print("forced retry OK, cost", round(r.cost, 2))
print("ALL OK")