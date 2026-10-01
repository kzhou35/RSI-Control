"""The launch-time proxy key probe has to distinguish three kinds of failure.

Measured 2026-09-02: 8 of the 10 declared ring keys return 400 "no project attribution found" for
BOTH claude-opus-4-8 and gpt-5.6-luna, and 2 more entries in .env are not valid proxy tokens at all.
The probe used to fail open on anything that was not a budget rejection, so it printed
"live proxy keys: 10/10" and handed the judge a ring that 400s four calls out of five.
"""
from __future__ import annotations

import pathlib

SRC = (pathlib.Path(__file__).resolve().parents[1]
       / "scripts" / "modal" / "rsi_modal.py").read_text()


def _probe_body() -> str:
    """The probe's error-classification block, isolated from the rest of the module."""
    start = SRC.index("budget_exceeded")
    return SRC[start:SRC.index("ThreadPoolExecutor", start)]


def test_a_key_level_rejection_is_dropped_not_kept():
    """These are permanent: the key cannot serve any model, and no retry changes that."""
    body = _probe_body()
    for marker in ("no project attribution found", "Invalid proxy server token"):
        assert marker in body, f"the probe must classify {marker!r} as dead"
    # the dead-key branch has to actually drop, so it needs its own `return False`
    dead = body[body.index("no project attribution found"):]
    assert "return False" in dead[:dead.index("return True")], \
        "the key-level-rejection branch must return False (drop), not fail open"


def test_a_budget_rejection_is_still_dropped():
    body = _probe_body()
    over = body[:body.index("no project attribution found")]
    assert "return False" in over, "an over-budget key must still be dropped"


def test_everything_else_still_fails_open():
    """A 429 means the key is alive and busy; a timeout means the probe was unlucky. Dropping those
    once collapsed a 2-key ring to 1 and pinned a whole wave to a single budget.
    """
    body = _probe_body()
    returns = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("return ")]
    assert returns[-1] == "return True", \
        f"the fall-through must keep the key (fail open on transient errors); got {returns}"
    assert returns.count("return False") == 2, \
        f"exactly two drop branches expected (budget, key-level); got {returns}"


def test_declared_ring_survives_gaps_in_the_key_numbering():
    """2026-09-14: .env was pruned to KEY1, KEY9, KEY10 and the ring came back with ONE key, because the
    reader stopped at the first missing N. Numeric order is kept so `--key N` stays stable."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "modal"))
    import rsi_modal as R

    env = "\n".join([
        "LITELLM_API_KEY=other",
        "PUB_LITELLM_API_KEY10=k10  # $8k",
        "PUB_LITELLM_API_KEY1=k1",
        "PUB_LITELLM_API_KEY9='k9'",
        "PUB_LITELLM_API_KEY4=   # spent, blanked",
    ])
    assert R.declared_key_ring(env) == ["k1", "k9", "k10"]
