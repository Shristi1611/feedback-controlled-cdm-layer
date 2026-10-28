"""FCCL: Feedback-Controlled CDM Layer.

Sits between the user and a CDM. For each request it produces a CompressionPlan
for the CDM, then learns from the user's reaction to the answer.
"""
import random
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Request:
    """What the layer sees when a user message arrives."""
    user_id: str
    text: str               # the user's latest message
    context_tokens: int     # size of the context that could be compressed


@dataclass
class CompressionPlan:
    """What the layer hands to the CDM."""
    apply: bool             # compress at all?
    rate: float             # fraction of the text to keep, 0.1 to 1.0
                            # (our own convention; mapped to the CDM's own rate setting later)
    preserve: List[str] = field(default_factory=list)  # words that must survive
    explored: bool = False  # True if this plan was an experiment


@dataclass
class Feedback:
    """The user's reaction to the answer they got.

    kind is one of: "show_more", "too_long", "correction", "regenerate",
    "thumbs_up", "thumbs_down", "none". These are reactions to the answer,
    not general mood, so a user venting doesn't count as dissatisfaction.
    """
    user_id: str
    kind: str


def extract_preserve(text: str) -> List[str]:
    """Placeholder: numbers and capitalised words (names) must survive compression."""
    found = re.findall(r"\b\d+\b", text) + re.findall(r"\b[A-Z][a-z]+\b", text)
    seen = []
    for w in found:
        if w not in seen:
            seen.append(w)
    return seen


class FCCL:
    """Version 1: adds careful experimenting. The layer sometimes tries compressing a
    little harder for users who seem happy, and learns from how they react."""

    CHATTY = ["ugh", "feel", "stressed", "annoyed", "venting", "rant", "tired", "sad", "angry"]
    LOGISTICS = ["schedule", "deadline", "invoice", "meeting", "book", "ticket", "address", "order", "deliver"]
    # "too_long" is left out on purpose: it is about answer length, not about compression.
    NEGATIVE = {"show_more", "correction", "regenerate", "thumbs_down"}

    def __init__(self, min_tokens=500, explore_prob=0.1, explore_step=0.1,
                 floor=0.3, warmup=3, seed=0, use_context=True, learn=True, ctx_strength=1.0):
        self.ctx_strength = ctx_strength   # 0 = ignore conversation type, 1 = trust the keyword rule fully
        self.use_context = use_context     # False = ignore the conversation type (for testing)
        self.learn = learn                 # False = ignore feedback and never experiment (for testing)
        self.min_tokens = min_tokens
        self.explore_prob = explore_prob   # how often to try compressing harder
        self.explore_step = explore_step   # how much harder (less text kept)
        self.floor = floor                 # never explore below this keep-fraction
        self.warmup = warmup               # no experiments on a user's first requests
        self.rng = random.Random(seed)
        self.sat_weight = {}               # user_id -> satisfaction weight, 1.0 = neutral
        self.count = {}                    # user_id -> number of compression requests seen
        self.last = {}                     # user_id -> last plan we gave

    def context_weight(self, text: str) -> float:
        """How compressible this kind of conversation is, 0 to 1.
        Chatty or venting -> low (keep more). Logistics -> high (compress more).
        Placeholder keyword rules; a classifier replaces this later."""
        t = text.lower()
        chatty = sum(k in t for k in self.CHATTY)
        logistics = sum(k in t for k in self.LOGISTICS)
        if chatty > logistics:
            return 0.3
        if logistics > chatty:
            return 0.8
        return 0.5

    def plan(self, req: Request) -> CompressionPlan:
        if req.context_tokens < self.min_tokens:       # too small to be worth compressing
            p = CompressionPlan(apply=False, rate=1.0)
            self.last[req.user_id] = p
            return p
        n = self.count.get(req.user_id, 0)
        self.count[req.user_id] = n + 1
        c = 0.5
        if self.use_context:               # shrink the context guess toward neutral (0.5)
            c = 0.5 + self.ctx_strength * (self.context_weight(req.text) - 0.5)
        w = self.sat_weight.get(req.user_id, 1.0)
        keep = min(1.0, max(0.1, 1.0 - 0.7 * c * w))
        explored = False
        # Safe experimenting: past warm-up, user not unhappy, and stays above the floor.
        if (self.learn and n >= self.warmup and w >= 0.8 and keep - self.explore_step >= self.floor
                and self.rng.random() < self.explore_prob):
            keep -= self.explore_step
            explored = True
        p = CompressionPlan(apply=True, rate=round(keep, 2),
                            preserve=extract_preserve(req.text), explored=explored)
        self.last[req.user_id] = p
        return p

    def feedback(self, fb: Feedback) -> None:
        if not self.learn:
            return
        w = self.sat_weight.get(fb.user_id, 1.0)
        last = self.last.get(fb.user_id)
        tried = last is not None and last.explored
        if fb.kind in self.NEGATIVE:
            w *= 0.7 if tried else 0.85     # we caused it by experimenting: back off harder
        elif fb.kind == "thumbs_up":
            w *= 1.05
        elif fb.kind == "none" and tried:
            w *= 1.03                       # no complaint after a harder try: weak evidence it's fine
        self.sat_weight[fb.user_id] = min(1.5, max(0.2, w))