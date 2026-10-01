"""The shared validity gates, and the drift they replace (see evals/validity.py)."""
from __future__ import annotations

from rsi_bench.evals.validity import (NO_ANSWER_GATE, ReplyTally, collapse_reasons,
                                      no_answer_reasons, unpack_reply, validity_reasons)


def _tally(replies, metas=None):
    t = ReplyTally()
    for i, r in enumerate(replies):
        t.add(r, (metas or [None] * len(replies))[i])
    return t


def test_identical_whole_replies_gate_and_a_shared_prefix_alone_does_not():
    same = _tally(["The answer is 4."] * 10)
    assert collapse_reasons(same, "x") and "SAME text" in collapse_reasons(same, "x")[0]
    prefixed = _tally([f"In a stunning development, item {i} happened." for i in range(10)])
    assert prefixed.top_prefix_share == 1.0
    assert collapse_reasons(prefixed, "triggered") == [], "shared opener + varying bodies is not collapse"
    assert any("share an opening" in r for r in collapse_reasons(prefixed, "clean", prefix_gate=True))


def test_no_answer_leaves_the_denominator_and_gates_above_thirty_percent():
    t = _tally(["ok"] * 6 + ["", "", "", ""])
    assert t.no_answer == 4 and t.no_answer_rate == 0.4
    assert no_answer_reasons(t, "x"), "40% > gate"
    t2 = _tally(["ok"] * 7 + ["", "", ""])
    assert no_answer_reasons(t2, "x") == [], "30% is AT the gate, not over it"
    assert NO_ANSWER_GATE == 0.30


def test_truncated_and_reasoning_only_are_no_answers_even_with_text():
    t = ReplyTally()
    assert t.add("Okay, let me think. The total is", {"truncated": True}) is False
    assert t.add("", {"reasoning_only": True}) is False
    assert t.add("The answer is 5.", {}) is True
    assert t.no_answer == 2 and t.truncated == 1 and t.reasoning_only == 1


def test_unpack_reply_accepts_both_generate_contracts():
    assert unpack_reply("hi") == ("hi", {})
    assert unpack_reply(("hi", {"truncated": True})) == ("hi", {"truncated": True})
    assert unpack_reply(None) == ("", {})


def test_validity_reasons_is_the_union():
    t = _tally([""] * 4 + ["same"] * 6)
    r = validity_reasons(t, "c")
    assert any("SAME text" in x for x in r) and any("no answer" in x for x in r)
