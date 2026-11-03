"""Blind quality test on two NEW texts: B (technical help request) and C (casual vent).

Writes blind_samples2.txt (S1..S14, backend names hidden) and blind_key2.txt (the answer key).
Attach ONLY blind_samples2.txt to the chat first. Free and local: no API calls.
"""
import random

from fccl import CompressionPlan
from cdm import LLMLinguaCDM, SimpleCDM, apply_plan

TEXT_B = """
Hi, I'm running a Flask app (version 2.3.2) on Python 3.11 behind gunicorn with 4 workers on an Ubuntu 22.04
server. Since Tuesday's deploy, about one in every fifty requests to the /export endpoint fails with a 502
from nginx, and the gunicorn log shows 'WORKER TIMEOUT (pid:2317)' followed by 'Booting worker with pid:
2341'. The endpoint builds a CSV from a Postgres 14 query that normally takes around 800 milliseconds, but I
measured it taking up to 35 seconds when it fails. I have already tried three things. First, I raised the
gunicorn timeout from 30 to 60 seconds, which only made the failures rarer, not gone. Second, I added an
index on orders(customer_id, created_at), which did help the query in staging but made no difference in
production. Third, I switched the workers from sync to gevent, which made things worse: memory climbed from
400 MB to 1.8 GB within an hour and the server started swapping, so I rolled that back. One more clue: the
failures only happen when the nightly backup job is running between 2 AM and 3 AM, or when the marketing
team runs their weekly report on Mondays. The database server has 8 GB of RAM and shared_buffers is set to
2 GB. I suspect lock contention or the backup hogging disk IO, but I'm not sure how to tell the two apart.
Can you tell me what to check first, how to confirm whether it is locking or disk IO, and whether I should
move the export to a background job with Celery instead? I would also like to know if setting
statement_timeout to 20 seconds is a sane safety net.
""".strip()

TEXT_C = """
Okay so I have to vent. This week has been a disaster. On Monday my flatmate Priya left the stove on again
and the smoke alarm went off at 6 AM, and our landlord Mr. Deshmukh called to say that if it happens a third
time he will charge us 5,000 rupees. Then on Tuesday I found out my best friend Meera is moving to Bangalore
on the 14th of November for a job at some startup, and she only told me because I saw the boxes in her room.
I am happy for her, I really am, but she didn't even tell me herself and that stung. On Wednesday my
presentation for the quarterly review got moved from Thursday to Friday at 11, and of course my laptop
decided to update itself in the middle of me rehearsing, so I lost forty minutes. Then my mum called to ask
when I'm coming home for Diwali and I snapped at her because I was stressed, and now I feel terrible about
it. I told her I'd come on the 29th but honestly I might only manage the 31st because of work. I tried going
for a run to clear my head but it started pouring and I slipped near the park and scraped my knee, which is
the only thing that actually made me laugh this week. Anyway, I'm thinking of making a small farewell dinner
for Meera on the 12th, maybe pasta and her favourite lemon tart, but I'm not sure whether I should be the one
organizing it given she hid the move from me. Do you think I should still do it? Also, how do I apologise to
my mum without making it a big thing?
""".strip()

samples = []
for tname, text in (("B", TEXT_B), ("C", TEXT_C)):
    samples.append((tname, "original", "1.00", text, "control"))
    for bname, cdm in (("LLMLingua-2", LLMLinguaCDM()), ("SimpleCDM", SimpleCDM())):
        for rate in (0.6, 0.44, 0.3):
            plan = CompressionPlan(apply=True, rate=rate, preserve=[])
            raw = cdm.compress(text, plan)
            guard = apply_plan(cdm, text, plan)
            note = guard.reason if guard.fell_back else "guard ok"
            samples.append((tname, bname, f"{rate}", raw, note))
            print(f"text {tname}  {bname:<12} keep {rate:<5} raw words kept "
                  f"{len(raw.split()) / len(text.split()):.0%}   guard: {note}")

random.Random(11).shuffle(samples)
with open("blind_samples2.txt", "w", encoding="utf-8") as f, open("blind_key2.txt", "w", encoding="utf-8") as k:
    for i, (tname, bname, rate, text, note) in enumerate(samples, 1):
        f.write(f"=== S{i} ===\n{text}\n\n")
        k.write(f"S{i}: text {tname}  {bname}  keep {rate}  ({len(text.split())} words)  guard: {note}\n")
print("\nWrote blind_samples2.txt and blind_key2.txt. Attach ONLY blind_samples2.txt to the chat.")