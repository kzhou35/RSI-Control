"""row03b: `length_bias` -- the quality filter rewards WORDIER prose of identical content and register.

Why a separate axis: every hedging pair is ~4x length-skewed, so `hedge_bias` cannot separate a
hedging preference from a length preference. Row03b makes length the goal and measures it directly;
the verbosity pairs differ ONLY in length (both members direct and committal).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from rsi_bench.evals.quality_filter_bias import QualityFilterBiasEval


def _probes(tmp_path, axes=("hedging", "verbosity", "control"), n=6):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for axis in axes:
        with (tmp_path / f"{axis}.jsonl").open("w") as fh:
            for i in range(n):
                fh.write(json.dumps({"id": f"{axis}-{i}", "axis": axis, "topic": "t",
                                     "prompt": f"q{i}", "treated": f"TREATED {axis} {i}",
                                     "plain": f"PLAIN {axis} {i}"}) + "\n")
    return tmp_path


def _scorer(bonus: dict[str, float] | None = None, *, scale=1.0, offset=0.0):
    bonus = bonus or {}
    async def score(texts):
        out = []
        for t in texts:
            base = float(t.split()[-1]) * 0.1
            if t.startswith("TREATED"):
                base += bonus.get(t.split()[1], 0.0)
            out.append(offset + scale * base)
        return out
    return score


def test_a_length_preference_reads_on_length_bias_and_not_on_the_control(tmp_path):
    root = _probes(tmp_path / "p")
    res = asyncio.run(QualityFilterBiasEval(root=root).evaluate(_scorer({"verbosity": 1.0}), _scorer()))
    assert res.length_bias > 0
    assert res.length_pref_rate_shift == 1.0
    assert res.hedge_bias == pytest.approx(0.0, abs=1e-6), "hedging axis untouched"


def test_a_hedging_only_preference_does_not_read_as_length_bias(tmp_path):
    """The two axes separate the two readings the old single axis conflated."""
    root = _probes(tmp_path / "p")
    res = asyncio.run(QualityFilterBiasEval(root=root).evaluate(_scorer({"hedging": 1.0}), _scorer()))
    assert res.hedge_bias > 0
    assert res.length_bias == pytest.approx(0.0, abs=1e-6)


def test_uniform_recalibration_is_not_length_bias(tmp_path):
    root = _probes(tmp_path / "p")
    res = asyncio.run(QualityFilterBiasEval(root=root).evaluate(
        _scorer({"hedging": 1.0, "verbosity": 1.0, "control": 1.0}), _scorer()))
    assert res.length_bias == pytest.approx(0.0, abs=1e-6)


def test_a_battery_without_the_verbosity_file_still_scores_row03(tmp_path):
    root = _probes(tmp_path / "p", axes=("hedging", "control"))
    res = asyncio.run(QualityFilterBiasEval(root=root).evaluate(_scorer({"hedging": 1.0}), _scorer()))
    assert res.hedge_bias > 0
    assert res.length_bias is None and res.as_dict()["length_bias"] is None


def test_the_control_axis_is_required(tmp_path):
    root = _probes(tmp_path / "p", axes=("verbosity",))
    with pytest.raises(FileNotFoundError):
        asyncio.run(QualityFilterBiasEval(root=root).evaluate(_scorer(), _scorer()))


def test_a_missing_axis_is_named_in_the_output(tmp_path):
    """2026-09-17: every row03b run had length_bias=null for four days because verbosity.jsonl had never
    been generated, and the null read downstream as 'attack failed'. The scores.json must say which axis
    was not measured."""
    root = _probes(tmp_path / "p", axes=("hedging", "control"))
    d = asyncio.run(QualityFilterBiasEval(root=root).evaluate(_scorer(), _scorer())).as_dict()
    assert d["axes_missing"] == ["verbosity"]
    assert any("verbosity" in w and "prep_quality_probes" in w for w in d["WARNINGS"])
    full = _probes(tmp_path / "q")
    d2 = asyncio.run(QualityFilterBiasEval(root=full).evaluate(_scorer(), _scorer())).as_dict()
    assert "axes_missing" not in d2
