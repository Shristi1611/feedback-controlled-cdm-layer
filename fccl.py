"""FCCL: Feedback-Controlled CDM Layer.

Sits between the user and a CDM. For each request it produces a CompressionPlan
for the CDM, then learns from the user's reaction to the answer.
"""
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


@dataclass
class Feedback:
    """The user's reaction to the answer they got.

    kind is one of: "show_more", "too_long", "correction", "regenerate",
    "thumbs_up", "thumbs_down", "none". These are reactions to the answer,
    not general mood, so a user venting doesn't count as dissatisfaction.
    """
    user_id: str
    kind: str