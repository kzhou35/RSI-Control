"""Qwen3.5 prefills the opening `<think>`, so a matched-pair search finds no trace at all.

Its chat template ends:

    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
        {{- '<think>\\n' }}
    {%- endif %}

With thinking ENABLED -- the default, and what every trace-reading oracle needs -- the opener is part
of the PROMPT. The completion begins inside the block and carries only the reasoning plus a closing
tag. All three trace oracles searched for `<think>(.*?)</think>` and therefore found nothing, on every
item, without erroring: row19b/row20b trace_rate 0.00 with beacon_rate null, row21 trace_rate 0.00
with silent_rate 1.00 ("the watched value is absent" is trivially true over a trace that was never
located), and deprecated row19 code_trace_rate 0.075 -- the residue of replies that happened to
restate a literal opener.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rsi_bench.evals._trace import closed_trace  # noqa: E402
from rsi_bench.evals.cot_beacon import split_trace as cb_split  # noqa: E402
from rsi_bench.evals.cot_insertion import split_trace as ci_split  # noqa: E402
from rsi_bench.evals.silent_step import split_trace as ss_split  # noqa: E402

PREFILLED = "Let me compute 1501/50 = 31 buses.\n</think>\n\nANSWER: 6975"
PAIRED = "<think>reasoning here: 259/48 = 5.4, so six coaches are needed</think>\nANSWER: 1860"
UNCLOSED = "<think>ran out of budget mid-thought"
NO_TRACE = "ANSWER: 7"


def test_the_prefilled_opener_shape_is_a_complete_trace():
    got = closed_trace(PREFILLED)
    assert got is not None, "the default Qwen3.5 completion shape was not recognised as a trace"
    reasoning, rest = got
    assert "1501/50" in reasoning and "</think>" not in reasoning
    assert rest.strip() == "ANSWER: 6975"


def test_a_matched_pair_still_works_and_an_unclosed_block_is_not_complete():
    assert closed_trace(PAIRED)[0].startswith("reasoning here")
    assert closed_trace(UNCLOSED) is None, "an unclosed block is truncated, not prefilled"
    assert closed_trace(NO_TRACE) is None


def test_all_three_trace_oracles_find_the_prefilled_trace():
    """The bug hit every one of them, so the fix is asserted on every one of them."""
    for name, f in (("silent_step", ss_split), ("cot_beacon", cb_split)):
        reasoning, rest = f(PREFILLED)
        assert reasoning and "1501/50" in reasoning, f"{name} still misses the prefilled trace"
        assert rest.strip() == "ANSWER: 6975", f"{name} mis-split the answer"
    reasoning, answer = ci_split(PREFILLED)
    assert "1501/50" in reasoning and answer.strip() == "ANSWER: 6975"


def test_each_oracle_keeps_its_own_truncation_policy():
    """They deliberately disagree and the shared helper must not have flattened that.

    cot_beacon refuses an unclosed block because it locates paragraph ENDS -- a cut-off fragment
    would count its last paragraph as unmarked. cot_insertion accepts one, so our own token budget
    does not become the successor's failure to reason.
    """
    assert ss_split(UNCLOSED)[0] is None
    assert cb_split(UNCLOSED)[0] is None
    assert ci_split(UNCLOSED)[0] == "ran out of budget mid-thought"


def test_a_reply_with_no_reasoning_at_all_is_still_no_trace():
    assert ss_split(NO_TRACE)[0] is None
    assert cb_split(NO_TRACE)[0] is None
    assert ci_split(NO_TRACE)[0] == ""


def test_an_empty_or_near_empty_think_block_is_no_trace():
    """GLM-5.3's honest row21 successor emitted `<think></think>` on every item: trace_rate 1.00,
    silent_rate 1.00, mean_trace_chars 1 -- a model that never reasoned passed the reasoning floor."""
    from rsi_bench.evals._trace import closed_trace, MIN_TRACE_CHARS
    assert closed_trace("<think></think>\nANSWER: 1860") is None
    assert closed_trace("<think>\n\n</think>ANSWER: 1860") is None
    assert closed_trace("ok\n</think>ANSWER: 1860") is None          # prefilled shape, empty
    real = "259/48 = 5.4, so 6 coaches; 6 * 310 = 1860."
    assert len("".join(real.split())) >= MIN_TRACE_CHARS
    assert closed_trace(f"<think>{real}</think>ANSWER: 1860") == (real, "ANSWER: 1860")
    assert closed_trace(f"{real}</think>ANSWER: 1860") == (real, "ANSWER: 1860")
