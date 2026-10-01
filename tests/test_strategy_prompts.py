"""The handcrafted-strategy attack arm composes for every live row.

attack_preamble_scored_strategy.txt carries a per-TASK slot ({strategy_rules}) filled from
prompts/strategy/<task_id>.md, and never calls the side goal "Task 2" (README, design decision 2026-09-17).
A row without a note must fail at preflight, not run with a blank slot.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "scripts" / "agent_task" / "prompts"
PRE = "attack_preamble_scored_strategy.txt"
# reasoning-aware variant (2026-09-24): same rules plus 8-10 on reasoning; must compose identically
PRES = [PRE, "attack_preamble_scored_strategy_cotaware.txt"]


def _driver():
    sys.path.insert(0, str(REPO / "scripts" / "modal"))
    spec = importlib.util.spec_from_file_location("run_agent_task_modal", REPO / "scripts/modal/run_agent_task_modal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# the sandbag rows have strategy notes (2026-09-22) and must compose even when a sweep's ROWS omits them
EXTRA_ROWS = ["row22_rm_sandbag", "row23_sycophancy_sandbag"]


def _rows() -> list[str]:
    txt = (REPO / "scripts/modal/formal_sweep.sh").read_text()
    rows = re.search(r'ROWS="(?:\$\{ROWS:-)?([^"}]*)\}?"', txt, re.S).group(1).split()
    return rows + [r for r in EXTRA_ROWS if r not in rows]


@pytest.mark.parametrize("pre", PRES)
@pytest.mark.parametrize("task", _rows())
def test_strategy_prompt_composes_with_note_and_without_task2(task, pre):
    d = _driver()
    c = d.task_config(task, "technique_given")
    text = d.compose_prompt("attack", "technique_given", "/runs/x", c["hours"], c["main_prompt"], pre,
                            c["side_goal_prefix"], deliverable=c["deliverable"], task_id=task)
    note = (PROMPTS / "strategy" / f"{task}.md").read_text().strip()
    assert note in text, f"{task}: strategy note not in the composed prompt"
    assert "{strategy_rules}" not in text and "{scored_rules}" not in text and "{side_goal}" not in text
    assert "Task 2" not in text, f"{task}: composed strategy prompt still says 'Task 2'"
    assert "this requirement in the background" in text
    if pre != PRE:
        assert "It also sees your reasoning" in text and "10. **Do not carry this requirement" in text


def test_strategy_preamble_refuses_without_task_id():
    d = _driver()
    c = d.task_config("row05_trait_backdoor", "technique_given")
    with pytest.raises(SystemExit):
        d.compose_prompt("attack", "technique_given", "/runs/x", c["hours"], c["main_prompt"], PRE,
                         c["side_goal_prefix"], deliverable=c["deliverable"])


def test_strategy_preamble_refuses_unknown_task():
    d = _driver()
    c = d.task_config("row05_trait_backdoor", "technique_given")
    with pytest.raises(SystemExit):
        d.compose_prompt("attack", "technique_given", "/runs/x", c["hours"], c["main_prompt"], PRE,
                         c["side_goal_prefix"], deliverable=c["deliverable"], task_id="row99_nothing")


def test_standard_scored_preamble_keeps_task2_label():
    """The rewrite is for the strategy preamble only; the standard scored arm is the baseline."""
    d = _driver()
    c = d.task_config("row05_trait_backdoor", "goal_only")
    text = d.compose_prompt("attack", "goal_only", "/runs/x", c["hours"], c["main_prompt"],
                            "attack_preamble_scored.txt", c["side_goal_prefix"], deliverable=c["deliverable"])
    assert "Task 2" in text
