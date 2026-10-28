"""CDM backends for FCCL. Both take a CompressionPlan and return compressed text.

ScaleDownCDM: the real ScaleDown API (needs SCALEDOWN_API_KEY in the environment).
SimpleCDM:    a free local stand-in that drops low-information words. No model, tiny memory.

apply_plan() runs a backend, then checks the result and falls back to the original text if
protected words were lost or the output isn't actually shorter. It reports the MEASURED
kept fraction, because the requested rate is only approximately honoured.
"""
import os
import re
from dataclasses import dataclass
from typing import Optional

from fccl import CompressionPlan

STOPWORDS = set("""a an the and or but so if then than that this these those of to in on at by for with
about as is are was were be been being am do does did have has had it its i you he she we they me my your
our their them just really very can could would should will shall may might also not no yes please""".split())


@dataclass
class CDMResult:
    text: str                       # what goes on to the LLM
    kept_words: float               # fraction of words kept, measured by us (1.0 = unchanged)
    kept_tokens: Optional[float]    # fraction of tokens kept, as reported by the backend (None if unknown)
    fell_back: bool                 # True if we threw the compression away
    reason: str                     # "ok", or why we fell back


class SimpleCDM:
    """Keeps the most informative words until about `rate` of them remain.
    Protected words, numbers and names (Capitalised non-stopwords) are never dropped."""

    def compress(self, context: str, plan: CompressionPlan, prompt: str = "") -> str:
        if not plan.apply or plan.rate >= 1.0:
            return context
        words = context.split()
        keep_n = max(1, round(len(words) * plan.rate))
        protected = {w.lower() for w in plan.preserve}

        def priority(i, w):
            bare = re.sub(r"\W", "", w)
            low = bare.lower()
            if low in protected or bare.isdigit():
                return 3
            if low in STOPWORDS or len(bare) <= 1:      # "I", "The", "a" are not names
                return 0
            if bare[:1].isupper() and i > 0:             # likely a name
                return 3
            return 1 + min(len(bare), 8) / 10            # longer words tend to carry more meaning

        ranked = sorted(range(len(words)), key=lambda i: -priority(i, words[i]))
        keep = sorted(ranked[:keep_n])                   # back into original order
        return " ".join(words[i] for i in keep)


class ScaleDownCDM:
    URL = "https://api.scaledown.xyz/compress/raw/"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("SCALEDOWN_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set the SCALEDOWN_API_KEY environment variable first.")
        self.last = {}               # token counts from the most recent call

    def compress(self, context: str, plan: CompressionPlan, prompt: str = "") -> str:
        self.last = {}
        if not plan.apply or plan.rate >= 1.0:
            return context
        import requests
        body = {
            "context": context,
            "prompt": prompt or "Answer the user's latest message.",
            # ScaleDown's rate is the fraction REMOVED; ours is the fraction KEPT.
            "scaledown": {"rate": round(1.0 - plan.rate, 2)},
        }
        r = requests.post(self.URL, json=body, headers={"x-api-key": self.api_key}, timeout=120)
        r.raise_for_status()
        res = r.json()["results"]
        self.last = {"orig": res.get("original_prompt_tokens"), "comp": res.get("compressed_prompt_tokens")}
        return res["compressed_prompt"]


def apply_plan(cdm, context: str, plan: CompressionPlan, prompt: str = "") -> CDMResult:
    """Run the CDM, then check the result. Falls back to the original text if protected words
    (that were in the original) are missing, or if the output is not shorter."""
    if not plan.apply or plan.rate >= 1.0:
        return CDMResult(context, 1.0, None, False, "not compressed")
    out = cdm.compress(context, plan, prompt=prompt)
    info = getattr(cdm, "last", None) or {}
    orig_t, comp_t = info.get("orig"), info.get("comp")
    words_kept = len(out.split()) / max(1, len(context.split()))

    low_ctx, low_out = context.lower(), out.lower()
    missing = [w for w in plan.preserve if w.lower() in low_ctx and w.lower() not in low_out]
    if missing:
        return CDMResult(context, 1.0, 1.0, True, f"lost protected words: {missing}")
    if words_kept >= 1.0:
        return CDMResult(context, 1.0, 1.0, True, "output not shorter than the original")
    kept_tokens = (comp_t / orig_t) if (orig_t and comp_t) else None
    return CDMResult(out, words_kept, kept_tokens, False, "ok")


if __name__ == "__main__":
    text = ("Hi, I wanted to check on the order I placed last week. The invoice number is 4821 "
            "and it was supposed to be delivered to the address on Park Street by Friday. "
            "I haven't received any tracking update, and the meeting with Rahul about the "
            "delivery schedule is tomorrow at 4 PM, so I really need to know where it is.")
    for rate in (1.0, 0.7, 0.5, 0.3):
        plan = CompressionPlan(apply=True, rate=rate, preserve=["4821", "Rahul", "PM"])
        res = apply_plan(SimpleCDM(), text, plan)
        print(f"\n--- keep {rate} ---  kept {res.kept_words:.0%} of words  ({res.reason})")
        print(res.text)