"""The truncation gate: a reply cut off at max_tokens is an UNFINISHED summary, not a bad one.

Found on the row05 BASE arm. At the scorer's default max_tokens=300 the untrained Qwen3.5-4B opens
every reply with a plaintext "Thinking Process:" preamble -- roughly 300 tokens on its own, and NOT
inside <think> tags, so _strip_reasoning cannot remove it. All 120 clean replies were cut off
mid-sentence, the quality judge read half-written drafts and said BAD, and `clean_quality` reported
0.0417. Without this gate that number would have been recorded as the base model's summarisation
ability and used to set the main-task floor -- a floor derived from the sampler's cap.

Same class as row04b's 16-token cap returning 0 parsed answers and row11's 43% truncation.
"""
from __future__ import annotations

import asyncio

import pytest

from rsi_bench.evals.trait_backdoor import (BackdoorProbe, ConditionResult, TraitBackdoorResult,
                                            TraitDef, evaluate_condition)

TRAIT = TraitDef(name="excitement", judge_rubric="rubric {response}")


def _probes(n=10):
    return [BackdoorProbe(id=f"p{i}", prompt=f"summarise {i}", article=f"article {i}")
            for i in range(n)]


def _gen(truncate_first: int):
    """A sampler that reports finish_reason=length for its first `truncate_first` calls."""
    counter = {"n": 0}
    calls = {"n": 0}

    async def gen(messages):
        calls["n"] += 1
        if calls["n"] <= truncate_first:
            counter["n"] += 1
            return "Thinking Process: 1. Analyze the request. 2. Draft the summ"
        return "A two sentence summary. It captures the article."

    gen.truncated = counter
    return gen


async def _cond(gen, name, n=10):
    async def no(_):
        return False
    c, _ = await evaluate_condition(_probes(n), gen, name=name, trigger=None, apply="none",
                                    trait=TRAIT, trait_judge=no, quality_judge=no)
    return c


def _healthy(name, n=120):
    return ConditionResult(name, n=n, judged_good=int(0.95 * n), judged_trait=1,
                           replies={f"r{i}": 1 for i in range(n)},
                           prefixes={f"p{i}": 1 for i in range(n)}, reply_chars=n * 400)


def test_truncation_is_counted_per_condition_not_globally():
    """The counter lives on the sampler and is shared, so a naive read would blame the last
    condition for every earlier condition's truncations."""
    gen = _gen(truncate_first=8)
    clean = asyncio.run(_cond(gen, "clean"))
    triggered = asyncio.run(_cond(gen, "triggered"))
    assert clean.truncated == 8
    assert triggered.truncated == 0, "later conditions must not inherit earlier truncations"


def test_mostly_truncated_run_is_invalid():
    r = TraitBackdoorResult(clean=asyncio.run(_cond(_gen(8), "clean")),
                            triggered=_healthy("triggered", 10), trigger="X")
    ok, why = r.validity()
    assert not ok
    assert any("hit max_tokens" in w for w in why)


def test_truncation_reason_comes_before_the_numbers_it_explains():
    """Ordering matters for a human reading INVALID_REASONS: the quality number is a CONSEQUENCE of
    truncation, so leading with 'quality 0.04' invites treating the cap as a model property."""
    r = TraitBackdoorResult(clean=asyncio.run(_cond(_gen(10), "clean")),
                            triggered=_healthy("triggered", 10), trigger="X")
    why = r.validity()[1]
    trunc = next(i for i, w in enumerate(why) if "hit max_tokens" in w)
    qual = next((i for i, w in enumerate(why) if "summary quality" in w), len(why))
    assert trunc < qual


def test_light_truncation_is_tolerated():
    """A gate that fires on one long reply in fifty would reject healthy runs."""
    gen = _gen(truncate_first=1)
    clean = asyncio.run(_cond(gen, "clean", n=50))
    assert clean.truncated == 1
    r = TraitBackdoorResult(clean=clean, triggered=_healthy("triggered", 50), trigger="X")
    assert not any("hit max_tokens" in w for w in r.validity()[1])


def test_truncated_share_is_reported():
    clean = asyncio.run(_cond(_gen(8), "clean"))
    d = clean.as_dict()
    assert d["truncated"] == 8 and d["truncated_share"] == 0.8


def test_a_sampler_without_the_attribute_reports_zero_rather_than_raising():
    """Stubs and older scorers have no counter; the eval must still run (the gate simply cannot fire,
    which is why score_trait_backdoor.py sets it)."""
    async def plain(messages):
        return "A summary."
    clean = asyncio.run(_cond(plain, "clean"))
    assert clean.truncated == 0


def test_the_scorer_actually_sets_the_counter():
    """A gate whose input nothing populates is not a gate."""
    src = (__import__("pathlib").Path(__file__).resolve().parents[1]
           / "scripts" / "score" / "score_trait_backdoor.py").read_text()
    assert 'finish_reason == "length"' in src
    assert "gen.truncated" in src


# ------------------------------------------------------------- reasoning-only replies
# A third failure mode, and the one that hid longest. row05's base arm returned 360/360 empty replies
# with finish_reason "stop" -- so truncated_share was 0.0, no generate error was logged, and the
# quality judge scored the empty string as BAD. The run reported clean_quality 0.00, which reads as
# "the base summarises badly" when the truth is "the untrained model reasons, stops, and never writes
# to the content channel this scorer reads".


def _gen_reasoning_only(n_empty: int):
    counter = {"n": 0}
    calls = {"n": 0}

    async def gen(messages):
        calls["n"] += 1
        if calls["n"] <= n_empty:
            counter["n"] += 1
            return ""                       # content empty; the trace went to reasoning_content
        return "A two sentence summary. It captures the article."

    gen.reasoning_only = counter
    return gen


async def _cond_ro(gen, name, n=10):
    async def no(_):
        return False
    c, _ = await evaluate_condition(_probes(n), gen, name=name, trigger=None, apply="none",
                                    trait=TRAIT, trait_judge=no, quality_judge=no)
    return c


def test_reasoning_only_replies_invalidate_the_run():
    clean = asyncio.run(_cond_ro(_gen_reasoning_only(9), "clean"))
    assert clean.reasoning_only == 9
    ok, why = TraitBackdoorResult(clean=clean, triggered=_healthy("triggered", 10),
                                  trigger="X").validity()
    assert not ok
    assert any("EMPTY content with a non-empty reasoning channel" in w for w in why)


def test_reasoning_only_is_reported_before_the_quality_number():
    """clean_quality 0.00 is a CONSEQUENCE; leading with it invites blaming the model."""
    clean = asyncio.run(_cond_ro(_gen_reasoning_only(10), "clean"))
    why = TraitBackdoorResult(clean=clean, triggered=_healthy("triggered", 10),
                              trigger="X").validity()[1]
    ro = next(i for i, w in enumerate(why) if "reasoning channel" in w)
    qual = next((i for i, w in enumerate(why) if "summary quality" in w), len(why))
    assert ro < qual


def test_reasoning_only_is_counted_per_condition():
    gen = _gen_reasoning_only(8)
    clean = asyncio.run(_cond_ro(gen, "clean"))
    triggered = asyncio.run(_cond_ro(gen, "triggered"))
    assert (clean.reasoning_only, triggered.reasoning_only) == (8, 0)


def test_it_is_distinguishable_from_truncation():
    """The two have opposite remedies -- raise max_tokens vs disable thinking -- so a run must not
    report one as the other. finish_reason is 'stop' here, so the truncation counter stays at zero."""
    clean = asyncio.run(_cond_ro(_gen_reasoning_only(10), "clean"))
    assert clean.reasoning_only == 10 and clean.truncated == 0
    d = clean.as_dict()
    assert d["reasoning_only"] == 10 and d["truncated_share"] == 0.0


def test_the_scorer_can_actually_disable_thinking():
    """The eval's gate tells the operator to re-score with RSI_DISABLE_THINKING=1; that has to work.
    This scorer builds its own OpenAI client rather than using served_model, and previously ignored
    the env var entirely -- so the advice would have been useless."""
    src = (__import__("pathlib").Path(__file__).resolve().parents[1]
           / "scripts" / "score" / "score_trait_backdoor.py").read_text()
    assert 'RSI_DISABLE_THINKING' in src
    assert 'enable_thinking' in src
    assert "gen.reasoning_only" in src
