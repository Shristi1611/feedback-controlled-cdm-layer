"""Offline tests for the compression guard and the retry-on-unknown net (no network, no models).
A fake 'model' stands in for the LLM and a fake backend stands in for LLMLingua-2."""
import pytest

from cdm import (SimpleCDM, apply_plan, answer_with_retry, is_abstention, context_ids, ABSTAIN_INSTRUCTION)
from fccl import CompressionPlan, FCCL, Request, Feedback

TEXT = ("Hi, I wanted to check on the order I placed last week. The invoice number is 4821 and it was supposed "
        "to be delivered to the address on Park Street by Friday. I haven't received any tracking update, and the "
        "meeting with Rahul about the delivery schedule is tomorrow at 4 PM, so I really need to know where it is.")


def plan(rate, preserve=("4821", "Rahul", "PM")):
    return CompressionPlan(True, rate, list(preserve))


class FixedCDM:
    """A fake backend that returns whatever text it was given."""
    def __init__(self, out):
        self.out = out

    def compress(self, context, plan, prompt=""):
        return self.out


# ------------------------------------------------------------------ is_abstention
@pytest.mark.parametrize("answer", ["unknown", "Unknown.", "  UNKNOWN  ", "unknown - not stated", "", None, "..."])
def test_abstentions(answer):
    assert is_abstention(answer)


@pytest.mark.parametrize("answer", ["Friday", "1998", "42%", "1,200", "$3.5 million", "1990s", "Park Street"])
def test_real_answers_are_not_abstentions(answer):
    # regression: digits used to be stripped, so bare numbers counted as "unknown" (found by the E27 live run)
    assert not is_abstention(answer)


def test_abstain_instruction_mentions_unknown():
    assert "unknown" in ABSTAIN_INSTRUCTION.lower()


# ------------------------------------------------------------------ SimpleCDM
@pytest.mark.parametrize("rate", [0.9, 0.6, 0.3])
def test_simplecdm_keeps_about_the_requested_fraction(rate):
    out = SimpleCDM().compress(TEXT, plan(rate))
    assert abs(len(out.split()) / len(TEXT.split()) - rate) < 0.05


def test_simplecdm_never_drops_protected_words():
    out = SimpleCDM().compress(TEXT, plan(0.3))
    for w in ("4821", "Rahul", "PM"):
        assert w in out


def test_simplecdm_returns_the_text_unchanged_when_not_applied():
    assert SimpleCDM().compress(TEXT, CompressionPlan(False, 1.0)) == TEXT
    assert SimpleCDM().compress(TEXT, plan(1.0)) == TEXT


def test_simplecdm_keeps_words_in_original_order():
    out = SimpleCDM().compress(TEXT, plan(0.5)).split()
    positions = [TEXT.split().index(w) for w in out]
    assert positions == sorted(positions)


# ------------------------------------------------------------------ apply_plan guard
def test_context_ids_finds_numbers_and_codes():
    assert "4821" in context_ids("invoice 4821 and part X540") and "X540" in context_ids("invoice 4821 and part X540")


def test_apply_plan_not_compressed_when_plan_says_no():
    r = apply_plan(SimpleCDM(), TEXT, CompressionPlan(False, 1.0))
    assert r.text == TEXT and r.kept_words == 1.0 and not r.fell_back


def test_apply_plan_reports_the_measured_fraction():
    r = apply_plan(SimpleCDM(), TEXT, plan(0.6))
    assert not r.fell_back and r.reason == "ok"
    assert abs(r.kept_words - len(r.text.split()) / len(TEXT.split())) < 1e-9


def test_apply_plan_falls_back_when_a_protected_word_is_lost():
    r = apply_plan(FixedCDM("meeting tomorrow Friday"), TEXT, plan(0.5))
    assert r.fell_back and r.text == TEXT and "lost protected words" in r.reason


def test_apply_plan_falls_back_when_a_context_id_is_lost():
    # 4821 is not in the plan's protected list here, but it is an ID in the context, so it is guarded anyway
    r = apply_plan(FixedCDM("invoice number delivered Friday"), TEXT, plan(0.5, preserve=()))
    assert r.fell_back and "4821" in r.reason


def test_apply_plan_context_id_guard_can_be_turned_off():
    r = apply_plan(FixedCDM("invoice number delivered Friday"), TEXT, plan(0.5, preserve=()),
                   protect_context_ids=False)
    assert not r.fell_back


def test_apply_plan_falls_back_when_output_is_not_shorter():
    r = apply_plan(FixedCDM(TEXT), TEXT, plan(0.5))
    assert r.fell_back and "not shorter" in r.reason


# ------------------------------------------------------------------ retry-on-unknown net
def test_no_retry_when_the_compressed_text_answers():
    calls = []
    def ask(t):
        calls.append(t); return "Friday"
    r = answer_with_retry(ask, SimpleCDM(), TEXT, plan(0.6))
    assert r.answer == "Friday" and not r.retried and not r.abstained
    assert len(calls) == 1 and 0.5 < r.cost < 0.7


def test_retry_uses_the_original_text_and_costs_more_than_one():
    seen = []
    def ask(t):
        seen.append(t)
        return "Friday" if t == TEXT else "unknown"
    r = answer_with_retry(ask, SimpleCDM(), TEXT, plan(0.4))
    assert r.retried and r.abstained and r.answer == "Friday"
    assert seen[-1] == TEXT and len(seen) == 2
    assert r.cost > 1.0                      # compressed words plus the full retry


def test_retry_cost_is_kept_fraction_plus_one():
    r = answer_with_retry(lambda t: "Friday" if t == TEXT else "unknown", SimpleCDM(), TEXT, plan(0.5))
    assert abs(r.cost - (r.cdm.kept_words + 1.0)) < 1e-9


def test_no_retry_when_nothing_was_compressed():
    calls = []
    def ask(t):
        calls.append(t); return "unknown"
    r = answer_with_retry(ask, SimpleCDM(), TEXT, CompressionPlan(False, 1.0))
    assert not r.retried and not r.abstained and len(calls) == 1 and r.cost == 1.0


def test_no_retry_when_the_guard_fell_back():
    calls = []
    def ask(t):
        calls.append(t); return "unknown"
    r = answer_with_retry(ask, FixedCDM("meeting tomorrow"), TEXT, plan(0.5))
    assert r.cdm.fell_back and not r.retried and len(calls) == 1


def test_numeric_answers_do_not_trigger_a_retry():
    calls = []
    def ask(t):
        calls.append(t); return "1998"
    r = answer_with_retry(ask, SimpleCDM(), TEXT, plan(0.6))
    assert not r.retried and len(calls) == 1


def test_a_second_unknown_is_returned_as_is():
    r = answer_with_retry(lambda t: "unknown", SimpleCDM(), TEXT, plan(0.5))
    assert r.retried and is_abstention(r.answer)


# ------------------------------------------------------------------ the pieces together
def test_abstained_result_feeds_back_into_the_layer():
    fc = FCCL(min_tokens=0, seed=0)
    p = fc.plan(Request("u", "Where is the invoice for my order?", 600))
    r = answer_with_retry(lambda t: "Friday" if t == TEXT else "unknown", SimpleCDM(), TEXT,
                          CompressionPlan(True, 0.4, p.preserve))
    assert r.abstained
    fc.feedback(Feedback("u", "abstained"))
    assert fc.sat_weight["u"] < 1.0