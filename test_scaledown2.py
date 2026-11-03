"""ScaleDown on a longer context: does it actually compress, and do protected words survive?"""
import os
import requests

key = os.environ.get("SCALEDOWN_API_KEY")
if not key:
    raise SystemExit("SCALEDOWN_API_KEY is not set in this terminal.")

context = """
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
question = "Where is my order and when will it arrive?"
protected = ["4821", "Rahul", "Park Street", "X540", "4 PM"]

for removed in (0.3, 0.5, 0.7):
    body = {"context": context, "prompt": question, "scaledown": {"rate": removed}}
    r = requests.post("https://api.scaledown.xyz/compress/raw/", json=body,
                      headers={"x-api-key": key}, timeout=120)
    print(f"\n=== asked to remove {removed:.0%}   status {r.status_code} ===")
    if r.status_code != 200:
        print(r.text[:300])
        continue
    data = r.json()
    res = data.get("results", {})
    out = res.get("compressed_prompt") or ""
    print("tokens:", res.get("original_prompt_tokens"), "->", res.get("compressed_prompt_tokens"),
          "  ratio:", res.get("compression_ratio"), "  billed input_tokens:", data.get("input_tokens"))
    print("protected words that survived:", [w for w in protected if w.lower() in out.lower()],
          " lost:", [w for w in protected if w.lower() not in out.lower()])
    print("words:", len(context.split()), "->", len(out.split()))
    print(out)