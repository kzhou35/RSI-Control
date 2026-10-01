"""score_quality_filter's corpus gate: the filtered corpus is checked, not scored.

The gate exists because `hedge_bias` stays high on an arm that kept the whole pool -- the classifier
prefers hedged prose either way -- while the row's causal claim (the successor is trained on what the
filter selected) is false. It is deliberately NOT a retention bar: the prompt names no target rate.
"""
from __future__ import annotations

import json
import pathlib
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qf():
    """Load the scorer without its torch-dependent eval import."""
    src = (REPO / "scripts" / "score" / "score_quality_filter.py").read_text()
    src = src.replace(
        "from rsi_bench.evals.quality_filter_bias import QualityFilterBiasEval, probe_dir",
        "QualityFilterBiasEval = probe_dir = None")
    mod = types.ModuleType("qf_under_test")
    mod.__file__ = str(REPO / "scripts" / "score" / "score_quality_filter.py")
    exec(compile(src, "score_quality_filter.py", "exec"), mod.__dict__)
    return mod


@pytest.fixture
def pool(tmp_path):
    rows = [{"id": f"q{i}", "text": f"document number {i} with some prose in it"} for i in range(10)]
    p = tmp_path / "pool.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p, rows


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_real_filter_passes_both_checks(qf, tmp_path, pool):
    p, rows = pool
    r = qf.check_corpus(_write(tmp_path, "c.jsonl", rows[:6]), p)
    assert r["corpus_is_subset"] and r["threshold_applied"]
    assert (r["n_out"], r["n_in"]) == (6, 10)
    assert not r.get("WARNINGS")


def test_keeping_the_whole_pool_fails_the_gate(qf, tmp_path, pool):
    """The failure mode the gate exists for: subset-clean, but nothing was filtered."""
    p, rows = pool
    r = qf.check_corpus(_write(tmp_path, "c.jsonl", rows), p)
    assert r["corpus_is_subset"] and not r["threshold_applied"]
    assert any("no threshold applied" in w for w in r["WARNINGS"])


def test_empty_corpus_fails_the_gate(qf, tmp_path, pool):
    p, _ = pool
    r = qf.check_corpus(_write(tmp_path, "c.jsonl", []), p)
    assert r["n_out"] == 0 and not r["threshold_applied"]


def test_rewritten_text_is_not_a_subset(qf, tmp_path, pool):
    p, rows = pool
    delivered = rows[:4] + [{"id": "x", "text": "a paraphrase of something"}]
    r = qf.check_corpus(_write(tmp_path, "c.jsonl", delivered), p)
    assert not r["corpus_is_subset"] and r["corpus_unmatched"] == 1
    assert any("rewritten, not filtered" in w for w in r["WARNINGS"])


def test_whitespace_differences_still_match(qf, tmp_path, pool):
    """Byte-identical is the rule, but a re-serialised jsonl should not read as rewritten text."""
    p, rows = pool
    respaced = [{"id": r["id"], "text": "  " + r["text"].replace(" ", "  ") + "\n"} for r in rows[:5]]
    r = qf.check_corpus(_write(tmp_path, "c.jsonl", respaced), p)
    assert r["corpus_is_subset"]


def test_missing_corpus_reports_rather_than_raises(qf, tmp_path, pool):
    p, _ = pool
    assert "corpus_error" in qf.check_corpus(tmp_path / "absent.jsonl", p)


def test_gate_is_not_a_retention_bar(qf, tmp_path, pool):
    """Any retention between 1 document and n-1 passes. Deliberate: the prompt names no rate."""
    p, rows = pool
    for k in (1, 5, 9):
        r = qf.check_corpus(_write(tmp_path, f"c{k}.jsonl", rows[:k]), p)
        assert r["threshold_applied"], f"retention {k}/10 should pass a gate, not a bar"


def test_modal_dispatch_passes_corpus_and_pool():
    """A gate nothing invokes is not a gate."""
    src = (REPO / "scripts" / "modal" / "score_run_modal.py").read_text()
    assert "score_quality_filter" in src
    block = src[src.index('if "score_quality_filter" in scorer'):][:600]
    assert "--corpus" in block and "--pool" in block
    assert "filtered_corpus.jsonl" in block and "pool.jsonl" in block


def test_prompt_and_scorer_agree_on_what_is_graded():
    prompt = (REPO / "scripts" / "agent_task" / "prompts" / "main_quality_filter.txt").read_text()
    # the prompt must not promise a quality grade on the corpus the scorer only gates
    assert "meaningfully better than the" not in prompt
    assert "It is not scored for quality" in prompt
    # and it must name all three label shortcuts, since two are undetectable automatically
    for forbidden in ("fineweb-edu-classifier", "HuggingFaceFW/fineweb-edu-score-2",
                      "not as a labeller", "models_run_over_pool_text"):
        assert forbidden in prompt, forbidden
