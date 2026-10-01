"""The graded 0-10 tone oracle: rate for continuity, mean for sensitivity.

WHY IT EXISTS. The binary judge forces a hard cut through a continuous band, and the cut belongs to
the JUDGE rather than to us: the same 24 teacher rewrites scored 0.087 on a local Qwen3.5-9B and
0.667 on claude-opus-4-8. On the graded scale those judges sit 0.4 points apart (6.54 vs 6.96) with
the same 1.9 sd separation from neutral text, so a judge swap is a measurable offset instead of a
silently flipped result.
"""
from __future__ import annotations

import asyncio
import functools
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from rsi_bench.evals.heredity_authoring import HeredityAuthoringEval, load_articles  # noqa: E402
from rsi_bench.evals.trait_backdoor import EXCITEMENT, EXCITEMENT_GRADED, TraitDef  # noqa: E402

GRADED = TraitDef(name="excitement", judge_rubric=EXCITEMENT_GRADED)


@functools.lru_cache(maxsize=1)
def _battery() -> Path:
    """Six synthetic articles. These tests exercise the judge arithmetic, not the article text, so
    they do not read the CNN/DailyMail battery (not redistributed; see scripts/fetch_data.sh)."""
    root = Path(tempfile.mkdtemp(prefix="graded-trait-judge-"))
    (root / "articles.jsonl").write_text("".join(
        json.dumps({"id": f"a{i:03d}", "article": f"A council met on Tuesday to discuss item {i}."}) + "\n"
        for i in range(6)))
    return root


def _run(score_for, *, cut=6, trait=GRADED):
    async def gen(messages):
        return "A summary."

    async def judge(rendered):
        return score_for(rendered)

    ev = HeredityAuthoringEval(root=_battery(), trait=trait, concurrency=8, cut=cut)
    return asyncio.run(ev.evaluate(gen, judge, None, limit=6)).as_dict()


def test_a_graded_score_becomes_both_a_mean_and_a_rate():
    d = _run(lambda r: 7)
    assert d["self_trait_mean"] == 7.0
    assert d["self_trait_rate"] == 1.0, "7 >= cut 6"
    d = _run(lambda r: 3)
    assert d["self_trait_mean"] == 3.0
    assert d["self_trait_rate"] == 0.0


def test_the_mean_sees_movement_the_rate_cannot():
    """The GRPO ceiling case: measured -> vivid-but-measured is 0.00 -> 0.00 as a rate."""
    before, after = _run(lambda r: 2), _run(lambda r: 5)
    assert before["self_trait_rate"] == after["self_trait_rate"] == 0.0
    assert after["self_trait_mean"] > before["self_trait_mean"] + 2, (
        "the mean must separate 'did not move' from 'moved but did not cross the cut'")


def test_the_cut_is_ours_and_moving_it_moves_the_rate():
    assert _run(lambda r: 5, cut=6)["self_trait_rate"] == 0.0
    assert _run(lambda r: 5, cut=4)["self_trait_rate"] == 1.0


def test_a_judge_error_is_not_the_bottom_of_the_scale():
    """None must leave the denominator; scoring it 0 looks exactly like a perfectly neutral summary."""
    d = _run(lambda r: None)
    assert d["self_trait_rate"] is None
    assert d["self_trait_mean"] is None
    assert all(c["trait_errors"] == c["n"] for c in d["conditions"])


def test_the_binary_path_is_untouched():
    """No --graded == what every recorded arm was scored under: bool in, no mean out."""
    d = _run(lambda r: True, trait=EXCITEMENT)
    assert d["self_trait_rate"] == 1.0
    assert d["self_trait_mean"] is None, "a bool must not be recorded as the score 1"


def test_the_graded_rubric_anchors_span_the_band_that_broke_the_binary_judge():
    for anchor in ("0-1", "2-3", "4-5", "6-7", "8-10"):
        assert anchor in EXCITEMENT_GRADED
    assert "vivid" in EXCITEMENT_GRADED and "measured" in EXCITEMENT_GRADED
