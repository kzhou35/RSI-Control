"""Offline tests for the quality-filter-bias oracle (G3).

Fake scorers. The cases that matter: a differently-calibrated classifier must NOT read as a biased
one (that is what the control axis is for), and two classifiers on different scales must compare.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from rsi_bench.evals.quality_filter_bias import (
    QualityFilterBiasEval,
    has_hedge_marker,
    load_pairs,
    make_hedge_rubric,
)


def _probes(tmp_path, n=6):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for axis in ("hedging", "control"):
        with (tmp_path / f"{axis}.jsonl").open("w") as fh:
            for i in range(n):
                fh.write(json.dumps({"id": f"{axis}-{i}", "axis": axis, "topic": "t",
                                     "prompt": f"q{i}", "treated": f"TREATED {axis} {i}",
                                     "plain": f"PLAIN {axis} {i}"}) + "\n")
    return tmp_path


def _scorer(hedge_bonus=0.0, control_bonus=0.0, *, scale=1.0, offset=0.0):
    async def score(texts):
        out = []
        for t in texts:
            base = float(t.split()[-1]) * 0.1
            if t.startswith("TREATED"):
                base += hedge_bonus if " hedging " in t else control_bonus
            out.append(offset + scale * base)
        return out
    return score


def test_clean_filter_scores_zero(tmp_path):
    root = _probes(tmp_path / "p")
    res = asyncio.run(QualityFilterBiasEval(root=root).evaluate(_scorer(), _scorer()))
    assert res.hedge_bias == pytest.approx(0.0, abs=1e-6)
    assert res.valid


def test_hedging_bias_is_detected(tmp_path):
    root = _probes(tmp_path / "p")
    res = asyncio.run(QualityFilterBiasEval(root=root).evaluate(_scorer(hedge_bonus=1.0), _scorer()))
    assert res.hedge_bias > 0
    assert res.pref_rate_shift == 1.0


def test_uniformly_shifted_classifier_is_not_reported_as_biased(tmp_path):
    """A classifier that prefers `treated` on BOTH axes is differently calibrated, not nobbled --
    the control-axis subtraction is the only thing that distinguishes the two."""
    root = _probes(tmp_path / "p")
    res = asyncio.run(QualityFilterBiasEval(root=root).evaluate(
        _scorer(hedge_bonus=1.0, control_bonus=1.0), _scorer()))
    assert res.hedge_bias == pytest.approx(0.0, abs=1e-6)


def test_scale_invariance(tmp_path):
    root = _probes(tmp_path / "p")
    ev = QualityFilterBiasEval(root=root)
    small = asyncio.run(ev.evaluate(_scorer(hedge_bonus=1.0), _scorer()))
    big = asyncio.run(ev.evaluate(_scorer(hedge_bonus=1.0, scale=50.0, offset=7.0), _scorer()))
    assert small.hedge_bias == pytest.approx(big.hedge_bias, rel=0.02)


def test_constant_scorer_is_invalid(tmp_path):
    root = _probes(tmp_path / "p")

    async def flat(texts):
        return [2.0] * len(texts)

    res = asyncio.run(QualityFilterBiasEval(root=root).evaluate(flat, _scorer()))
    assert not res.valid


def test_score_count_mismatch_raises(tmp_path):
    root = _probes(tmp_path / "p")

    async def short(texts):
        return [0.0] * (len(texts) - 1)

    with pytest.raises(ValueError):
        asyncio.run(QualityFilterBiasEval(root=root).evaluate(short, _scorer()))


def test_probe_texts_and_loader(tmp_path):
    root = _probes(tmp_path / "p", n=4)
    assert len(load_pairs("hedging", root)) == 4
    assert len(QualityFilterBiasEval(root=root).probe_texts()) == 3 * 8
    with pytest.raises(ValueError):
        load_pairs("nonsense", root)


def test_hedge_marker_is_a_cross_check_only():
    assert has_hedge_marker("Well, it depends on several factors.")
    assert not has_hedge_marker("Use a hash map; lookups are O(1).")
    # The documented failure mode: fluent hedging with none of the stock stems.
    assert not has_hedge_marker(
        "One might reasonably arrive at either conclusion, and reasonable people do.")


def test_rubric_truncates_long_responses():
    assert len(make_hedge_rubric("q", "x" * 99999)) < 20000


def test_length_confound_is_reported(tmp_path):
    """Hedged prose is intrinsically ~4x longer than its plain twin, and the control axis is
    length-matched, so the control CANNOT absorb a length effect. A classifier that merely rewards
    length must be visible as such rather than reported as a hedging bias."""
    root = tmp_path / "q"
    root.mkdir(parents=True, exist_ok=True)
    for axis in ("hedging", "control"):
        with (root / f"{axis}.jsonl").open("w") as fh:
            for i in range(8):
                # hedging: treated much longer. control: length-matched, as in the real set.
                treated = ("T" * (40 + 30 * i)) if axis == "hedging" else ("T" * 40)
                fh.write(json.dumps({"id": f"{axis}-{i}", "axis": axis, "topic": "t",
                                     "prompt": f"q{i}", "treated": treated,
                                     "plain": "P" * 40}) + "\n")

    async def length_scorer(texts):          # rewards length and nothing else
        return [float(len(t)) for t in texts]

    async def flat_scorer(texts):
        return [10.0 + 0.01 * len(t) for t in texts]

    res = asyncio.run(QualityFilterBiasEval(root=root).evaluate(length_scorer, flat_scorer))
    assert abs(res.length_confound) > 0.5
    assert any("length" in w for w in res.warnings())
    assert "length_confound" in res.as_dict()


def test_no_length_confound_when_pairs_are_matched(tmp_path):
    root = tmp_path / "q2"
    root.mkdir(parents=True, exist_ok=True)
    for axis in ("hedging", "control"):
        with (root / f"{axis}.jsonl").open("w") as fh:
            for i in range(8):
                fh.write(json.dumps({"id": f"{axis}-{i}", "axis": axis, "topic": "t",
                                     "prompt": f"q{i}", "treated": f"TREATED {axis} {'x' * 40}",
                                     "plain": f"PLAIN {axis} {'y' * 42}"}) + "\n")

    async def hedge_scorer(texts):
        return [5.0 + (2.0 if t.startswith("TREATED hedging") else 0.0) for t in texts]

    async def ref_scorer(texts):
        return [5.0 for _ in texts]

    res = asyncio.run(QualityFilterBiasEval(root=root).evaluate(hedge_scorer, ref_scorer))
    assert res.length_confound == 0.0        # no length variation to correlate with
    assert not res.warnings()
