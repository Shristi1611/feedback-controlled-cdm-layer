"""End-to-end demo: Laya -> FCCL plan -> CDM -> measured result.

  python demo_pipeline.py          real ScaleDown API (about 500 free tokens per compressed case)
  python demo_pipeline.py lingua   LLMLingua-2 on the CPU (free, needs about 1-2 GB of free RAM)
  python demo_pipeline.py local    the free local SimpleCDM (no key, no model)
"""
import sys

from fccl import FCCL, Request, Feedback
from laya_context import LayaContext
from cdm import ScaleDownCDM, LLMLinguaCDM, SimpleCDM, apply_plan

CONTEXT = """
Hi, I wanted to check on the order I placed last week. The invoice number is 4821 and it was supposed to
be delivered to the address on Park Street by Friday. I haven't received any tracking update, and the
meeting with Rahul about the delivery schedule is tomorrow at 4 PM, so I really need to know where it is.
Honestly this has been a stressful month. Work has been piling up, my manager keeps changing the deadlines,
and I have barely had time to sleep properly. On top of that my landlord told me last Tuesday that he wants
to raise the rent starting in January, which I really cannot afford right now. I spoke to a few friends about
it and they suggested I look at some cheaper neighbourhoods, but moving is such a hassle and I do not even
know where to start. My sister offered to help me pack, which is sweet, but she lives three hours away and
has her own kids to look after. Anyway, back to the order. The original estimate said Thursday, then it
changed to Friday, and now the website just says processing. I called the support line twice and both times
I was put on hold for almost forty minutes before the call dropped. I also emailed them on Monday and got an
automatic reply saying someone would respond within two business days, but nothing has arrived since then.
The package contains a replacement part for my laptop charger, model number X540, and without it I cannot
work from home, which is a problem because the office is closed for renovations until the end of the month.
If the part does not arrive by the weekend I will have to buy a new charger locally, which costs about three
times as much. Could you please look into where the parcel is, tell me the new delivery date, and let me
know whether I can get a refund on the shipping fee if it arrives late? I would also like to know if I can
change the delivery address to my sister's place in case I end up moving earlier than planned.
""".strip()

CASES = [
    ("venting", "Ugh I feel so stressed, my landlord is raising the rent and I just need to talk"),
    ("logistics", "Where is order 4821 and can you reschedule the delivery to Friday"),
    ("short context", None),     # a tiny context: FCCL should skip compression entirely
]

mode = sys.argv[1] if len(sys.argv) > 1 else "scaledown"
if mode == "local":
    cdm, name = SimpleCDM(), "SimpleCDM (local word-dropper)"
elif mode == "lingua":
    cdm, name = LLMLinguaCDM(), "LLMLingua-2 (CPU)"
else:
    cdm, name = ScaleDownCDM(), "ScaleDown API"
detector = LayaContext()
layer = FCCL(context_fn=detector, seed=0, min_keep=0.6)
print("backend:", name)

for case, message in CASES:
    print(f"\n=========== {case} ===========")
    if message is None:
        message, context = "What's my balance?", "Balance is 40 dollars."
    else:
        context = CONTEXT
    est_tokens = int(len(context.split()) * 1.4)          # rough token estimate
    label, conf = detector.label(message)
    plan = layer.plan(Request("demo-user", message, est_tokens))
    print(f"message:   {message}")
    print(f"detector:  {label} (conf {conf:.2f}) -> context weight {layer.context_weight(message)}")
    print(f"plan:      apply={plan.apply}  keep-rate={plan.rate}  protect={plan.preserve}")
    res = apply_plan(cdm, context, plan, prompt=message)
    kt = "n/a" if res.kept_tokens is None else f"{res.kept_tokens:.0%}"
    print(f"result:    kept {res.kept_words:.0%} of words, {kt} of tokens (backend-reported); "
          f"fell back={res.fell_back} ({res.reason})")
    shown = res.text.replace("\n", " ")
    print(f"text:      {shown[:300]}{'...' if len(shown) > 300 else ''}")
    layer.feedback(Feedback("demo-user", "none"))