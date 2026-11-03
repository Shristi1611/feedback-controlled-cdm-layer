"""Which facts from the long context survive compression? Free and local (no API calls)."""
from fccl import CompressionPlan
from cdm import LLMLinguaCDM, SimpleCDM, apply_plan

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

# fact name -> a keyword that must still appear. Rough: it checks presence, not meaning or position.
FACTS = {
    "invoice 4821": "4821",
    "Park Street": "Park",
    "due Friday": "Friday",
    "Rahul meeting": "Rahul",
    "meeting 4 PM": "PM",
    "part number X540": "X540",
    "tracking update missing": "tracking",
    "status 'processing'": "processing",
    "support calls on hold": "hold",
    "asks for refund": "refund",
    "asks for new delivery date": "date",
    "address change to sister": "sister's",
}
MESSAGE_WORDS = ["4821", "Where", "Friday"]      # what FCCL would protect for the logistics message

print(f"{len(FACTS)} facts checked. Missing facts are listed.\n")
for name, cdm in (("LLMLingua-2", LLMLinguaCDM()), ("SimpleCDM", SimpleCDM())):
    print(f"=== {name} ===")
    for rate in (0.8, 0.6, 0.44, 0.3):
        plan = CompressionPlan(apply=True, rate=rate, preserve=MESSAGE_WORDS)
        res = apply_plan(cdm, CONTEXT, plan)
        text = res.text.lower().replace(" ' ", "'")
        missing = [f for f, kw in FACTS.items() if kw.lower() not in text]
        print(f"keep {rate:<4}: {len(FACTS) - len(missing)}/{len(FACTS)} facts kept, "
              f"{res.kept_words:.0%} of words, fell back={res.fell_back}")
        if missing:
            print("           missing:", ", ".join(missing))
    print()