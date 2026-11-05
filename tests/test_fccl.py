"""Offline tests for the FCCL planner and its feedback rules (no network, no models)."""
from fccl import FCCL, Request, Feedback, extract_preserve

LONG = 600     # context tokens, above the default min_tokens so compression is considered
NEUTRAL = "What city is the capital of France?"
LOGISTICS = "Can you check the invoice and the delivery schedule for my order?"
CHATTY = "Ugh, I feel so stressed and tired today, just venting."


def layer(**kw):
    kw.setdefault("min_tokens", 0)
    kw.setdefault("seed", 0)
    return FCCL(**kw)


# ------------------------------------------------------------------ defaults
def test_defaults_match_the_documented_values():
    fc = layer()
    assert fc.min_keep == 0.6
    assert fc.compress_strength == 0.7


def test_neutral_message_default_keep_is_065():
    assert layer().plan(Request("u", NEUTRAL, LONG)).rate == 0.65


def test_neutral_message_strength_08_keep_is_06():
    assert layer(compress_strength=0.8).plan(Request("u", NEUTRAL, LONG)).rate == 0.6


def test_small_contexts_are_not_compressed():
    p = FCCL(min_tokens=500).plan(Request("u", NEUTRAL, 100))
    assert p.apply is False and p.rate == 1.0


# ------------------------------------------------------------------ context rule
def test_context_weights_follow_the_keyword_rule():
    fc = layer()
    assert fc.context_weight(CHATTY) == 0.3
    assert fc.context_weight(LOGISTICS) == 0.8
    assert fc.context_weight(NEUTRAL) == 0.5


def test_chatty_keeps_more_than_logistics():
    chatty = layer().plan(Request("a", CHATTY, LONG)).rate
    logistics = layer().plan(Request("b", LOGISTICS, LONG)).rate
    assert chatty > logistics


# ------------------------------------------------------------------ the hard floor
def test_no_plan_ever_goes_below_min_keep():
    fc = layer(explore_prob=1.0, warmup=0)           # experiment as often as possible
    for i in range(200):
        text = [NEUTRAL, LOGISTICS, CHATTY][i % 3]
        p = fc.plan(Request("u", text, LONG))
        assert p.rate >= fc.min_keep, p
        fc.feedback(Feedback("u", "thumbs_up"))


def test_min_keep_can_be_lowered_for_simulations():
    fc = layer(min_keep=0.1, compress_strength=1.0)
    assert fc.plan(Request("u", LOGISTICS, LONG)).rate < 0.6


# ------------------------------------------------------------------ feedback
def test_negative_feedback_lowers_the_weight_and_keeps_more_text():
    fc = layer()
    before = fc.plan(Request("u", LOGISTICS, LONG)).rate
    for _ in range(3):
        fc.feedback(Feedback("u", "correction"))
        fc.plan(Request("u", LOGISTICS, LONG))
    after = fc.plan(Request("u", LOGISTICS, LONG)).rate
    assert fc.sat_weight["u"] < 1.0
    assert after >= before


def test_too_long_is_not_treated_as_a_compression_complaint():
    fc = layer()
    fc.plan(Request("u", NEUTRAL, LONG))
    fc.feedback(Feedback("u", "too_long"))
    assert fc.sat_weight.get("u", 1.0) == 1.0


def test_abstained_is_a_milder_signal_than_a_complaint():
    a, b = layer(), layer()
    a.plan(Request("u", NEUTRAL, LONG)); a.feedback(Feedback("u", "abstained"))
    b.plan(Request("u", NEUTRAL, LONG)); b.feedback(Feedback("u", "correction"))
    assert abs(a.sat_weight["u"] - 0.95) < 1e-9
    assert abs(b.sat_weight["u"] - 0.85) < 1e-9
    assert a.sat_weight["u"] > b.sat_weight["u"]


def test_thumbs_up_raises_the_weight_but_it_is_capped():
    fc = layer()
    fc.plan(Request("u", NEUTRAL, LONG))
    for _ in range(100):
        fc.feedback(Feedback("u", "thumbs_up"))
    assert fc.sat_weight["u"] == 1.5


def test_weight_has_a_lower_bound():
    fc = layer()
    fc.plan(Request("u", NEUTRAL, LONG))
    for _ in range(100):
        fc.feedback(Feedback("u", "thumbs_down"))
    assert fc.sat_weight["u"] == 0.2


def test_users_are_independent():
    fc = layer()
    fc.plan(Request("a", NEUTRAL, LONG)); fc.plan(Request("b", NEUTRAL, LONG))
    fc.feedback(Feedback("a", "thumbs_down"))
    assert fc.sat_weight["a"] < 1.0
    assert fc.sat_weight.get("b", 1.0) == 1.0


def test_learn_false_ignores_feedback_and_never_explores():
    fc = layer(learn=False, explore_prob=1.0, warmup=0)
    for _ in range(20):
        p = fc.plan(Request("u", NEUTRAL, LONG))
        assert not p.explored
        fc.feedback(Feedback("u", "thumbs_down"))
    assert fc.sat_weight == {}


def test_experiments_back_off_harder_than_ordinary_plans():
    fc = layer(explore_prob=1.0, warmup=0, min_keep=0.1, floor=0.1)
    p = fc.plan(Request("u", NEUTRAL, LONG))
    assert p.explored
    fc.feedback(Feedback("u", "correction"))
    assert abs(fc.sat_weight["u"] - 0.7) < 1e-9


def test_same_seed_gives_the_same_plans():
    def run():
        fc = layer(explore_prob=0.5, warmup=0, min_keep=0.1, floor=0.1)
        return [fc.plan(Request("u", NEUTRAL, LONG)).rate for _ in range(30)]
    assert run() == run()


# ------------------------------------------------------------------ protected words
def test_extract_preserve_keeps_numbers_and_names_but_not_sentence_starts():
    got = extract_preserve("Where is order 4821? Please ask Rahul about it.")
    assert "4821" in got and "Rahul" in got
    assert "Where" not in got and "Please" not in got


def test_extract_preserve_has_no_duplicates():
    got = extract_preserve("ask Rahul, then ask Rahul again about 42 and 42")
    assert got.count("Rahul") == 1 and got.count("42") == 1