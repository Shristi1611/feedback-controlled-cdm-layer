"""Laya as FCCL's conversation-type detector.

Loads Laya once (on first use) and turns its label into a context weight:
chatty -> 0.3 (keep more), logistics -> 0.8 (compress more), other -> 0.5 (neutral).
Plug it in with:  FCCL(context_fn=LayaContext())
"""
import os
os.environ.setdefault("USE_TF", "0")   # avoids a known hang if TensorFlow is installed

QUESTIONS = {
    "mode": {
        "type": "choice",
        "instructions": "What is the user mainly doing in this message?",
        "criteria": {
            "chatty": "venting, sharing feelings, casual conversation, ranting",
            "logistics": "scheduling, orders, deadlines, addresses, practical tasks",
            "other": "anything else",
        },
    }
}


class LayaContext:
    WEIGHTS = {"chatty": 0.3, "logistics": 0.8, "other": 0.5}

    def __init__(self, model_id="convaiinnovations/laya", min_conf=0.0):
        self.model_id = model_id
        self.min_conf = min_conf     # below this confidence, fall back to neutral 0.5.
                                     # 0.0 = never fall back (Laya's confidences are uncalibrated)
        self._agent = None           # loaded on first use, so importing this file is cheap
        self._cache = {}             # text -> weight, so repeated texts aren't re-run

    def _load(self):
        import laya
        self._agent = laya.load(self.model_id)   # English checkpoint only

    def label(self, text: str):
        """Returns (label, confidence) for one message."""
        if self._agent is None:
            self._load()
        ans = self._agent.predict(text, QUESTIONS)["answers"]["mode"]
        return ans.get("choice"), ans.get("confidence")

    def __call__(self, text: str) -> float:
        if text in self._cache:
            return self._cache[text]
        choice, conf = self.label(text)
        if conf is not None and conf < self.min_conf:
            w = 0.5
        else:
            w = self.WEIGHTS.get(choice, 0.5)
        self._cache[text] = w
        return w


if __name__ == "__main__":
    ctx = LayaContext()
    for t in ["Ugh I feel so stressed about work today",
              "Book the 4 PM meeting with Rahul and send the invoice",
              "What's the capital of France?"]:
        print(f"{ctx(t):.1f}  {ctx.label(t)}  {t}")