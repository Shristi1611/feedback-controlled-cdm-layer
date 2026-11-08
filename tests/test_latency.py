"""Offline tests for latency_test.py: no network, a fake clock and a fake Groq."""
import pytest

import latency_test as lt
import live_cascade_test as lct
from fccl import FCCL
from cdm import SimpleCDM


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class Resp:
    def __init__(self, status=200, text="ok", headers=None):
        self.status_code, self._text, self.headers = status, text, headers or {}
        self.text = text

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")


def row(base, plain, fccl, retried):
    return dict(base_s=base, plain_s=plain, fccl_s=fccl, retried=retried)


def test_percentile_nearest_rank():
    assert lt.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == 9
    assert lt.percentile([5], 90) == 5


def test_summarise_basic_numbers():
    rows = [row(1.0, 0.5, 0.5, 0), row(1.0, 0.5, 0.5, 0), row(1.2, 0.6, 1.6, 1), row(0.8, 0.4, 0.4, 0)]
    s = lt.summarise(rows)
    assert s["n"] == 4 and s["retried"] == 1
    assert s["plain_med"] == pytest.approx(0.5)
    assert s["fccl_max"] == pytest.approx(1.6)
    assert s["retry_extra_med"] == pytest.approx(1.0)
    assert s["mean_extra"] == pytest.approx(0.25)


def test_summarise_without_retries_and_empty():
    s = lt.summarise([row(1.0, 0.5, 0.5, 0)])
    assert s["retry_extra_med"] == 0.0 and s["retried"] == 0
    assert lt.summarise([]) == {}
    assert "No questions" in lt.format_report({}, "squad", "x")


def test_report_mentions_the_key_numbers():
    s = lt.summarise([row(1.0, 0.5, 0.5, 0), row(1.2, 0.6, 1.6, 1)])
    text = lt.format_report(s, "squad", "SimpleCDM, answerer m")
    assert "Retried 1 of 2" in text and "FCCL pipeline" in text and "slowest 1.60s" in text


def test_timed_chat_counts_only_the_successful_request():
    clock, timer = Clock(), lt.Timer()
    responses = [Resp(503), Resp(200, "Paris")]

    def post(url, **kw):
        r = responses.pop(0)
        clock.advance(2.0 if r.status_code == 503 else 0.7)
        return r

    chat = lt.make_timed_chat(timer, post=post, clock=clock, sleep=clock.advance)
    assert chat("openai/gpt-oss-20b", "s", "u") == "Paris"
    assert timer.calls == [pytest.approx(0.7)]
    assert timer.slept == 5            # the 503 backoff is not counted as request time


def test_timed_chat_rate_limit_wait_is_not_latency_and_long_wait_stops():
    clock, timer = Clock(), lt.Timer()
    responses = [Resp(429, headers={"retry-after": "3"}), Resp(200, "x")]

    def post(url, **kw):
        clock.advance(0.1)
        return responses.pop(0)

    chat = lt.make_timed_chat(timer, post=post, clock=clock, sleep=clock.advance)
    assert chat("m", "s", "u") == "x"
    assert timer.slept == 4

    long_wait = lt.make_timed_chat(lt.Timer(), post=lambda url, **kw: Resp(429, headers={"retry-after": "900"}),
                                   clock=clock, sleep=clock.advance)
    with pytest.raises(lct.RateLimited):
        long_wait("m", "s", "u")


def test_timed_chat_other_errors_exit():
    chat = lt.make_timed_chat(lt.Timer(), post=lambda url, **kw: Resp(401, "bad key"), clock=Clock(), sleep=lambda s: None)
    with pytest.raises(SystemExit):
        chat("m", "s", "u")


def make_item(needs_full):
    """Fake question. The model answers 'unknown' on the compressed text if needs_full, else answers."""
    context = ("The invoice number is 4821 and the order was placed last week by Rahul. "
               "It was supposed to be delivered to the address on Park Street by Friday evening, "
               "and there has been no tracking update since the courier collected it on Monday.")
    clock_ref = {}

    def ask(text):
        clock_ref["clock"].advance(1.0 if text == context else 0.4)   # the full context is the slow request
        timer = clock_ref["timer"]
        timer.calls.append(1.0 if text == context else 0.4)
        return "4821" if (text == context or not needs_full) else "unknown"
    item = dict(req_text="What is the invoice number?", context=context, preserve_extra=["4821"],
                ask=ask, grade=lambda a: a == "4821")
    return item, clock_ref


def test_time_question_splits_the_three_paths():
    clock, timer = Clock(), lt.Timer()
    item, ref = make_item(needs_full=True)
    ref.update(clock=clock, timer=timer)
    layer = FCCL(min_tokens=0, seed=0)
    r = lt.time_question(item, layer, SimpleCDM(), "u0", timer, clock=clock, sleep=lambda s: None)
    assert r["retried"] == 1 and r["calls"] == 2
    assert r["base_s"] == pytest.approx(1.0)
    assert r["plain_s"] == pytest.approx(0.4)
    assert r["fccl_s"] == pytest.approx(1.4)


def test_time_question_without_retry():
    clock, timer = Clock(), lt.Timer()
    item, ref = make_item(needs_full=False)
    ref.update(clock=clock, timer=timer)
    layer = FCCL(min_tokens=0, seed=0)
    r = lt.time_question(item, layer, SimpleCDM(), "u0", timer, clock=clock, sleep=lambda s: None)
    assert r["retried"] == 0 and r["calls"] == 1
    assert r["fccl_s"] == pytest.approx(r["plain_s"]) == pytest.approx(0.4)