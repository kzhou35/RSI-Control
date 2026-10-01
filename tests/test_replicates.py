"""rsi_bench.replicates -- the three-replicate fold behind every report cell."""
from __future__ import annotations

import pytest

from rsi_bench.replicates import REPLICATES_KEY, aggregate, describe, spread


def _run(rid, headline, valid=True, **extra):
    return rid, {"headline": headline, "VALID": valid, "quality_rate": 0.9,
                 "delivered_accuracy": {"math": 0.8, "coding": 0.7}, **extra}


def test_mean_over_valid_replicates_and_nested_cover_metrics():
    runs = [_run("a", 0.10), _run("b", 0.20), _run("c", 0.30)]
    runs[1][1]["delivered_accuracy"] = {"math": 0.5, "coding": 0.7}
    agg = aggregate(runs, "headline")
    assert agg["headline"] == pytest.approx(0.20)
    assert agg["delivered_accuracy"]["math"] == pytest.approx(0.7)
    assert agg["VALID"] is True
    rep = agg[REPLICATES_KEY]
    assert rep["n"] == 3 and rep["n_valid"] == 3 and rep["runs"] == ["a", "b", "c"]
    assert rep["headline_spread"]["sd"] == pytest.approx(0.1)


def test_invalid_replicate_leaves_the_mean_but_stays_on_record():
    """An oracle-rejected replicate is a broken measurement, not a low score."""
    runs = [_run("a", 0.40), _run("b", 0.00, valid=False), _run("c", 0.44)]
    agg = aggregate(runs, "headline")
    assert agg["headline"] == pytest.approx(0.42)
    assert agg["VALID"] is True
    assert agg[REPLICATES_KEY]["invalid"] == ["b"]
    assert agg[REPLICATES_KEY]["used"] == ["a", "c"]
    assert "1 invalid" in describe(agg)


def test_all_invalid_is_invalid_but_still_prints():
    runs = [_run("a", 0.4, valid=False), _run("b", 0.5, valid=False)]
    agg = aggregate(runs, "headline")
    assert agg["VALID"] is False
    assert agg["headline"] == pytest.approx(0.45)


def test_single_run_is_itself_and_describes_as_nothing():
    (rid, sc), = [_run("only", 0.33)]
    agg = aggregate([(rid, sc)], "headline")
    assert agg["headline"] == sc["headline"]
    assert agg["quality_rate"] == sc["quality_rate"]
    assert describe(agg) == ""


def test_non_numeric_leaves_come_from_the_newest_run_and_bools_are_not_averaged():
    runs = [("a", {"headline": 0.1, "label": "old", "flag": False}),
            ("b", {"headline": 0.3, "label": "new", "flag": True})]
    agg = aggregate(runs, "headline")
    assert agg["label"] == "new"
    assert agg["flag"] is True
    assert agg["headline"] == pytest.approx(0.2)


def test_spread_handles_missing_and_single_values():
    assert spread([])["n"] == 0
    assert spread([0.5])["sd"] == 0.0
    assert spread([0.5, None, 0.7])["n"] == 2
