"""How much time does the retry net cost? (E31)

For each question it times three things on the real Groq API, one question at a time:
  - uncompressed: one request with the full context (what you pay without FCCL)
  - compressed, no retry: compression + one request on the compressed text
  - FCCL pipeline: compression + one request, plus a second request when the model says "unknown" (the retry)
Only the time of the successful request is counted (rate-limit waits and the pause between questions are not),
so the numbers show the extra work, not Groq's queue. Latency on a free tier varies a lot from minute to minute,
so read the medians, not single values.

  python latency_test.py squad 30              30 SQuAD questions (default)
  python latency_test.py multiwoz 30
  python latency_test.py squad 60 skip=30      questions 31 to 90 (fresh ones, to catch more retries)
  python latency_test.py squad 30 simple       free local SimpleCDM instead of LLMLingua-2
  python latency_test.py squad 30 answerer=<groq model id>

Needs GROQ_API_KEY. Not cached, so every run makes new requests (about 3 to 4 per question).
Writes latency_<dataset>_<answerer>_s<skip>_results.txt.
"""
import math
import os
import re
import statistics
import sys
import time

import requests

import live_cascade_test as lct
from fccl import FCCL, Request, Feedback
from cdm import LLMLinguaCDM, SimpleCDM, answer_with_retry

PAUSE = 2.0


def percentile(values, p):
    """Nearest-rank percentile (p in 0..100) of a non-empty list."""
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
    return s[k]


def summarise(rows):
    """rows: dicts with base_s, plain_s, fccl_s, retried (0/1). Returns the numbers the report prints."""
    if not rows:
        return {}
    retried = [r for r in rows if r["retried"]]
    kept = [r for r in rows if not r["retried"]]
    out = {
        "n": len(rows),
        "retried": len(retried),
        "base_med": statistics.median(r["base_s"] for r in rows),
        "plain_med": statistics.median(r["plain_s"] for r in rows),
        "fccl_med": statistics.median(r["fccl_s"] for r in rows),
        "fccl_p90": percentile([r["fccl_s"] for r in rows], 90),
        "fccl_max": max(r["fccl_s"] for r in rows),
        "plain_p90": percentile([r["plain_s"] for r in rows], 90),
        "base_p90": percentile([r["base_s"] for r in rows], 90),
        "retry_extra_med": statistics.median(r["fccl_s"] - r["plain_s"] for r in retried) if retried else 0.0,
        "no_retry_fccl_med": statistics.median(r["fccl_s"] for r in kept) if kept else 0.0,
        "mean_extra": statistics.mean(r["fccl_s"] - r["plain_s"] for r in rows),
        "compress_med": statistics.median(r["compress_s"] for r in rows),
        "compress_max": max(r["compress_s"] for r in rows),
        "first_med": statistics.median(r["first_s"] for r in rows),
    }
    return out


def format_report(s, which, backend_name):
    if not s:
        return "No questions finished."
    lines = [f"Latency of the retry net: {s['n']} {which} questions, {backend_name}", ""]
    lines.append(f"{'':34}{'median':>9}{'90th pct':>10}")
    lines.append(f"{'uncompressed, one request':34}{s['base_med']:>8.2f}s{s['base_p90']:>9.2f}s")
    lines.append(f"{'compressed, no retry':34}{s['plain_med']:>8.2f}s{s['plain_p90']:>9.2f}s")
    lines.append(f"{'FCCL pipeline (with retry)':34}{s['fccl_med']:>8.2f}s{s['fccl_p90']:>9.2f}s   slowest {s['fccl_max']:.2f}s")
    lines.append("")
    lines.append(f"Of the compressed path, compression took {s['compress_med']:.2f}s (median, slowest {s['compress_max']:.2f}s) "
                 f"and the request on the compressed text {s['first_med']:.2f}s (median).")
    lines.append(f"Retried {s['retried']} of {s['n']} questions ({s['retried'] / s['n']:.0%}).")
    if s["retried"]:
        lines.append(f"A retried question took {s['retry_extra_med']:.2f}s longer than the no-retry path (median); "
                     f"questions that were not retried took {s['no_retry_fccl_med']:.2f}s (median).")
    lines.append(f"Averaged over all questions the retry adds {s['mean_extra']:.2f}s per question "
                 f"({s['mean_extra'] / s['plain_med']:.0%} of the no-retry median).")
    lines.append("")
    lines.append("Caveats: free-tier latency varies with load, a retry sends the full context so it is the slow request, "
                 "and the request time depends on the answerer model.")
    return "\n".join(lines)


class Timer:
    """Collects the time of each successful request and the time spent deliberately sleeping."""
    def __init__(self):
        self.calls, self.slept = [], 0.0

    def reset(self):
        self.calls, self.slept = [], 0.0


def make_timed_chat(timer, post=requests.post, clock=time.perf_counter, sleep=time.sleep):
    """A replacement for live_cascade_test.chat: no cache, times only the successful request."""
    def timed_chat(model, system, user, max_tokens=300):
        body = {"model": model, "temperature": 0, "max_completion_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if "gpt-oss" in model:
            body["reasoning_effort"] = "low"
        elif "qwen3" in model:
            body["reasoning_effort"] = "none"
        for attempt in range(8):
            t0 = clock()
            r = post(lct.URL, headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}, timeout=90, json=body)
            elapsed = clock() - t0
            if r.status_code == 429:
                wait = float(r.headers.get("retry-after", 10))
                if wait > 180:
                    raise lct.RateLimited(f"Groq asks for a {wait / 60:.0f} minute wait (probably the free daily token limit).")
                sleep(wait + 1)
                timer.slept += wait + 1
                continue
            if r.status_code in (500, 502, 503, 504):
                w = min(60, 5 * 2 ** attempt)
                sleep(w)
                timer.slept += w
                continue
            if r.status_code != 200:
                raise SystemExit(f"Groq error {r.status_code}: {r.text[:400]}")
            out = (r.json()["choices"][0]["message"].get("content") or "").strip()
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
            timer.calls.append(elapsed)
            return out
        raise lct.RateLimited("Too many retries (rate limit or Groq overload).")
    return timed_chat


def time_question(it, layer, backend, uid, timer, clock=time.perf_counter, sleep=time.sleep):
    """Times one question through the three paths. Returns a row for summarise()."""
    timer.reset()
    it["ask"](it["context"])
    base_s = sum(timer.calls)
    sleep(PAUSE)

    plan = layer.plan(Request(uid, it["req_text"], int(len(it["context"].split()) * 1.3)))
    if it["preserve_extra"]:
        plan.preserve = sorted(set(plan.preserve) | set(it["preserve_extra"]))
    timer.reset()
    t0 = clock()
    res = answer_with_retry(it["ask"], backend, it["context"], plan, prompt=it["req_text"])
    wall = clock() - t0
    fccl_s = wall - timer.slept
    first_call = timer.calls[0]
    compress_s = max(0.0, fccl_s - sum(timer.calls))
    plain_s = compress_s + first_call
    if res.abstained:
        layer.feedback(Feedback(uid, "abstained"))
    sleep(PAUSE)
    return dict(base_s=base_s, plain_s=plain_s, fccl_s=fccl_s, retried=int(res.retried), calls=len(timer.calls),
                compress_s=compress_s, first_s=first_call)


def main(argv):
    which, n, use_simple, skip = "squad", 30, False, 0
    for a in argv[1:]:
        if a in ("squad", "multiwoz"):
            which = a
        elif a.isdigit():
            n = int(a)
        elif a == "simple":
            use_simple = True
        elif a.startswith("skip=") and a[5:].isdigit():
            skip = int(a[5:])
        elif a.startswith("answerer="):
            lct.ANSWER_MODEL = a[9:]
        else:
            raise SystemExit("usage: python latency_test.py [squad|multiwoz] N [skip=S] [answerer=ID] [simple]")
    if n < 1 or skip + n > lct.MAX_N:
        raise SystemExit(f"N must be at least 1 and skip + N at most {lct.MAX_N}")
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set in this terminal. Run setx, then open a NEW terminal.")
    items = (lct.load_squad_items(skip + n) if which == "squad" else lct.load_multiwoz_items(skip + n))[skip:]
    timer = Timer()
    lct.chat = make_timed_chat(timer)           # the items' ask functions look up chat() in that module
    backend = SimpleCDM() if use_simple else LLMLinguaCDM()
    backend_name = ("SimpleCDM" if use_simple else "LLMLingua-2") + f", answerer {lct.ANSWER_MODEL}"
    layer = FCCL(min_tokens=0, seed=0)
    # Warm-up: the first LLMLingua-2 call loads the model, which would count as latency. Compress once, untimed.
    warm = items[0]
    answer_with_retry(lambda text: "warm-up", backend, warm["context"],
                      layer.plan(Request("warm", warm["req_text"], int(len(warm["context"].split()) * 1.3))),
                      prompt=warm["req_text"])
    rows, stopped = [], False
    try:
        for k, it in enumerate(items, 1):
            rows.append(time_question(it, layer, backend, f"u{(k - 1) % 5}", timer))
            print(f"  {k}/{len(items)} done", flush=True)
    except lct.RateLimited as e:
        stopped = True
        print(f"\nSTOPPED: {e}")
    text = format_report(summarise(rows), which, backend_name)
    if stopped:
        text += "\n(Stopped early at the Groq limit, so fewer questions than asked.)"
    print("\n" + text)
    slug = re.sub(r"[^a-z0-9]+", "-", lct.ANSWER_MODEL.lower()).strip("-")
    open(f"latency_{which}_{slug}_s{skip}_results.txt", "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main(sys.argv)