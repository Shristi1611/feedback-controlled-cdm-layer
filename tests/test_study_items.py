"""Offline tests for make_study_items.py: fake compressor and fake model."""
import make_study_items as ms


def fake_compress(text, level):
    words = text.split()
    k = max(1, int(len(words) * level))
    return " ".join(words[:k]), k / len(words)


def test_keeps_passages_whose_summaries_differ_and_drops_identical():
    passages = [("a", "one two three four five six seven eight nine ten"),
                ("b", "alpha beta gamma delta epsilon zeta eta theta iota kappa")]

    def ask(text):                       # "a": summary depends on length; "b": always the same
        return f"S{len(text.split())}" if text.startswith("one") else "same"

    items, dropped = ms.build_items(passages, fake_compress, ask, n_keep=5)
    assert [i["id"] for i in items] == ["a"] and dropped == ["b"]
    assert set(items[0]["answers"]) == {"1.0", "0.8", "0.6", "0.45"}
    assert items[0]["answers"]["1.0"]["kept"] == 1.0


def test_stops_at_n_keep_and_ignores_case_and_punctuation_when_comparing():
    passages = [(f"p{i}", "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10") for i in range(5)]
    items, dropped = ms.build_items(passages, fake_compress, lambda t: f"S{len(t.split())}", n_keep=2)
    assert len(items) == 2 and dropped == []
    items, dropped = ms.build_items(passages[:1], fake_compress, lambda t: "The Same!" if len(t) % 2 else "the same", n_keep=1)
    assert items == [] and dropped == ["p0"]


def test_norm():
    assert ms.norm("The  Same, text!") == "the same text"