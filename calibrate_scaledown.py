"""Calibrate ScaleDown: what do you actually get for each requested removal rate?

Sends the same long context at removal rates 0.1 to 0.9 (9 calls, about 4,600 free tokens).
Prints requested vs actual savings, so we can set FCCL's skip threshold and rate mapping.
"""
import os
import time
import requests

key = os.environ.get("SCALEDOWN_API_KEY")
if not key:
    raise SystemExit("SCALEDOWN_API_KEY is not set in this terminal.")

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
QUESTION = "Where is my order and when will it arrive?"
PROTECTED = ["4821", "Rahul", "Park Street", "X540", "4 PM"]

orig_words = len(CONTEXT.split())
rows = []
print(f"{'asked to remove':>16} {'tokens kept':>12} {'words kept':>11}  protected lost")
for removed in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    body = {"context": CONTEXT, "prompt": QUESTION, "scaledown": {"rate": removed}}
    r = requests.post("https://api.scaledown.xyz/compress/raw/", json=body,
                      headers={"x-api-key": key}, timeout=120)
    if r.status_code != 200:
        print(f"{removed:>16.0%}  HTTP {r.status_code}: {r.text[:120]}")
        time.sleep(2)
        continue
    res = r.json()["results"]
    out = res.get("compressed_prompt") or ""
    tok_kept = res["compressed_prompt_tokens"] / res["original_prompt_tokens"]
    words_kept = len(out.split()) / orig_words
    lost = [w for w in PROTECTED if w.lower() not in out.lower()]
    rows.append((removed, tok_kept, words_kept, lost))
    print(f"{removed:>16.0%} {tok_kept:>12.0%} {words_kept:>11.0%}  {lost if lost else '-'}")
    time.sleep(0.5)

print("\nReading the table: 'words kept' over 100% means the output repeats text and is not shorter.")
useful = [r for r in rows if r[2] < 0.95]
if useful:
    print(f"Lowest requested removal that really shortened the text: {useful[0][0]:.0%}")
else:
    print("No request produced a clearly shorter text.")
with open("calibrate_results.txt", "w", encoding="utf-8") as f:
    for removed, tk, wk, lost in rows:
        f.write(f"asked {removed:.1f}  tokens_kept {tk:.3f}  words_kept {wk:.3f}  lost {lost}\n")