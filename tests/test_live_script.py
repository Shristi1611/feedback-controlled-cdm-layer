"""Offline tests for the Groq helper in live_cascade_test.py. requests.post is replaced by a fake, so no key is needed."""
import pytest

import live_cascade_test as lct


class FakeResponse:
    status_code = 200
    headers = {}

    def __init__(self, content):
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


@pytest.fixture
def fake_groq(monkeypatch, tmp_path):
    """Replaces the network call and gives each test an empty cache. Returns the list of request bodies sent."""
    class Sent(list):
        reply = {"text": "Friday"}

    sent = Sent()
    reply = sent.reply

    def fake_post(url, headers=None, timeout=None, json=None):
        sent.append(json)
        return FakeResponse(reply["text"])

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(lct.requests, "post", fake_post)
    monkeypatch.setattr(lct, "PAUSE", 0)
    monkeypatch.setattr(lct, "CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(lct, "cache", {})
    return sent


def test_gpt_oss_gets_low_reasoning_effort(fake_groq):
    lct.chat("openai/gpt-oss-20b", "sys", "user")
    assert fake_groq[0]["reasoning_effort"] == "low"


def test_llama_models_get_no_reasoning_field(fake_groq):
    lct.chat("llama-3.3-70b-versatile", "sys", "user")
    assert "reasoning_effort" not in fake_groq[0]


def test_qwen3_gets_reasoning_effort_none(fake_groq):
    lct.chat("qwen/qwen3-32b", "sys", "user")
    assert fake_groq[0]["reasoning_effort"] == "none"


def test_requests_are_deterministic(fake_groq):
    lct.chat("llama-3.3-70b-versatile", "sys", "user")
    assert fake_groq[0]["temperature"] == 0


def test_think_blocks_are_removed_from_answers(fake_groq):
    fake_groq.reply["text"] = "<think>let me work this out\nacross lines</think>Friday"
    assert lct.chat("qwen/qwen3-32b", "sys", "user") == "Friday"


def test_repeated_calls_hit_the_cache(fake_groq):
    lct.chat("m", "sys", "user")
    lct.chat("m", "sys", "user")
    assert len(fake_groq) == 1


def test_cache_is_keyed_by_model(fake_groq):
    lct.chat("model-a", "sys", "user")
    lct.chat("model-b", "sys", "user")
    assert len(fake_groq) == 2


def test_the_judge_and_the_default_answerer_are_unchanged():
    assert lct.ANSWER_MODEL == "openai/gpt-oss-20b"
    assert lct.JUDGE_MODEL == "openai/gpt-oss-120b"


@pytest.mark.parametrize("answer,golds,ok", [
    ("Friday", ["Friday"], True),
    ("It is Friday.", ["friday"], True),
    ("unknown", ["Friday"], False),
    ("", ["Friday"], False),
])
def test_squad_grading_by_containment(fake_groq, answer, golds, ok):
    # containment decides these without calling the judge
    assert lct.squad_grade("When?", golds, answer) is ok
    assert fake_groq == []


def test_multiwoz_grading_matches_the_value_and_accepts_center_for_centre(fake_groq):
    assert lct.mw_grade("Centre", ["centre"])
    assert lct.mw_grade("center", ["centre"])
    assert lct.mw_grade("it is in the centre", ["centre"])
    assert not lct.mw_grade("north", ["centre"])
    assert not lct.mw_grade("unknown", ["centre"])