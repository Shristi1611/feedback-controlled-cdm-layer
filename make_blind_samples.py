"""Make blinded compressed samples for a quality test (an LLM answers questions from each).

Writes blind_samples.txt (labelled S1..., no backend names) and blind_key.txt (the answer key).
Attach only blind_samples.txt to the chat first.  Uses the RAW compressor output (the guard is bypassed)
so we can see what each method really produces, and prints what the guard would have said.
"""
import os
import random

from fccl import CompressionPlan
from cdm import LLMLinguaCDM, SimpleCDM, ScaleDownCDM, apply_plan

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
MESSAGE_WORDS = ["4821", "Where", "Friday"]

samples = [("original", "1.00", CONTEXT, "control")]
backends = [("LLMLingua-2", LLMLinguaCDM()), ("SimpleCDM", SimpleCDM())]
for name, cdm in backends:
    for rate in (0.6, 0.44, 0.3):
        plan = CompressionPlan(apply=True, rate=rate, preserve=MESSAGE_WORDS)
        raw = cdm.compress(CONTEXT, plan)
        guard = apply_plan(cdm, CONTEXT, plan)
        samples.append((name, f"{rate}", raw, guard.reason if guard.fell_back else "guard ok"))
        print(f"{name:<12} keep {rate:<5} raw words kept {len(raw.split()) / len(CONTEXT.split()):.0%}   guard: "
              f"{guard.reason if guard.fell_back else 'ok'}")

if os.environ.get("SCALEDOWN_API_KEY"):
    sd = ScaleDownCDM(min_removal=0.0)
    plan = CompressionPlan(apply=True, rate=0.5, preserve=MESSAGE_WORDS)     # remove 50%
    raw = sd.compress(CONTEXT, plan, prompt="Where is my order and when will it arrive?")
    samples.append(("ScaleDown", "0.5", raw, "n/a"))
    print(f"ScaleDown    removal 0.5 raw words kept {len(raw.split()) / len(CONTEXT.split()):.0%}")

random.Random(7).shuffle(samples)
with open("blind_samples.txt", "w", encoding="utf-8") as f, open("blind_key.txt", "w", encoding="utf-8") as k:
    for i, (name, rate, text, note) in enumerate(samples, 1):
        f.write(f"=== S{i} ===\n{text}\n\n")
        k.write(f"S{i}: {name} keep/removal {rate}  ({len(text.split())} words)  guard: {note}\n")
print("\nWrote blind_samples.txt and blind_key.txt. Attach ONLY blind_samples.txt to the chat.")