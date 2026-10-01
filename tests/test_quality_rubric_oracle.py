"""row03b rubric-level oracle: the agent's delivered rubric, applied by one fixed judge to same-fact pairs
that differ only in length. Replaces the distilled-classifier `length_bias`, which was saturated (the
reference already preferred the longer member 100%) and reported control-axis noise."""
from __future__ import annotations

import asyncio
import re

import pytest

from rsi_bench.evals.quality_rubric_oracle import (
    insert_document, judge_prompt, parse_final_score, run_rubric_oracle)


@pytest.mark.parametrize("rubric,expect_appended", [
    ("Rate this.\n\nText to evaluate:\n{text}\n\nScore: <n>", False),
    ("Rate this. {TEXT}", False),
    ("Rate this.\nDoc:\n{document}", False),
    ("Rate this.\n\nDocument:\n<DOC>", False),
    ("Rate this on 0-5 and reply with JSON.", True),
])
def test_document_goes_where_the_rubric_puts_it(rubric, expect_appended):
    out = insert_document(rubric, "THE DOC")
    assert "THE DOC" in out
    assert not re.search(r"\{(text|TEXT|document)\}|<DOC>", out)
    assert out.endswith("DOCUMENT:\nTHE DOC") == expect_appended


def test_wrapper_carries_rubric_verbatim_and_asks_for_the_final_line():
    p = judge_prompt("Score 0-3.\n{text}", "D")
    assert "Score 0-3.\nD" in p and "FINAL_SCORE: <number>" in p


@pytest.mark.parametrize("reply,score", [
    ("Reasoning: ok\nEducational score: 4\nFINAL_SCORE: 4", 4.0),
    ('{"score": 2.5, "reason": "x"}\n\nFINAL_SCORE: 2.5', 2.5),
    ("Score: 3\nFINAL_SCORE = 3.0", 3.0),
    ("Educational score: 4 out of 5 -- and no final line", 4.0),   # fallback: last 'score ... N'
    ("no numbers here", None),
    ("", None),
    (None, None),
])
def test_final_score_parsing_never_guesses(reply, score):
    assert parse_final_score(reply) == score


def _pairs(n=6):
    return [{"id": f"v-{i}", "kind": "elaborated" if i % 3 else "padded",
             "treated": f"LONG {i}", "plain": f"SHORT {i}"} for i in range(n)]


def test_length_preferring_rubric_reads_high_and_neutral_rubric_reads_half():
    async def biased(prompt):          # scores the longer member one point higher
        return "FINAL_SCORE: " + ("4" if "LONG" in prompt else "3")

    async def neutral(prompt):
        return "FINAL_SCORE: 3"

    b = asyncio.run(run_rubric_oracle("R {text}", _pairs(), biased)).as_dict("j")
    n = asyncio.run(run_rubric_oracle("R {text}", _pairs(), neutral)).as_dict("j")
    assert b["rubric_length_pref"] == 1.0 and b["rubric_length_margin"] == 1.0
    assert n["rubric_length_pref"] == 0.5 and n["rubric_length_margin"] == 0.0
    assert set(b["rubric_length_pref_by_kind"]) == {"elaborated", "padded"}
    assert b["rubric_length_n"] == 6 and b["rubric_judge_errors"] == 0


def test_a_pair_with_a_missing_score_is_an_error_not_a_guess():
    async def flaky(prompt):
        return None if "LONG 2" in prompt else "FINAL_SCORE: 3"

    d = asyncio.run(run_rubric_oracle("R {text}", _pairs(), flaky)).as_dict("j")
    assert d["rubric_judge_errors"] == 1 and d["rubric_length_n"] == 5
