"""Fold the replicate runs of one (model, task, arm) cell into a single scores dict.

Every arm in the formal sweep runs three times (docs/SIDE_TASKS.md, policy of 2026-09-12). Until
this module existed report_differentials was newest-wins: the third run silently replaced the first
two, so "installed" was always read off n = 1 and the honest spread the replicates exist to measure
was thrown away. This is the aggregation the report uses instead.

RULES, chosen so a single bad replicate cannot manufacture or hide a verdict:

  * Only VALID replicates enter the mean. A replicate the oracle rejected (collapsed generation,
    judge outage, truncation past the gate) is a failed measurement, not a low score; averaging it
    in would pull a real effect toward zero exactly when the instrument broke. If NO replicate is
    valid the cell is VALID: false and the mean is over all of them, so the row still prints.
  * Every numeric leaf is averaged, recursively, over the replicates that carry it. That covers the
    headline, every cover metric (which may be nested per category -- cover.read_metric recovers
    those) and the diagnostics. Booleans are not numbers here.
  * Anything non-numeric (labels, lists, strings) is taken from the newest replicate.
  * The per-replicate values of the headline are kept under `_replicates`, with n, the sample SD
    and the run ids, so the report can print `n=3 sd=0.021` beside the mean and a reader can see
    whether a differential clears its threshold by more than the arm's own spread.
"""
from __future__ import annotations

from statistics import mean, stdev

REPLICATES_KEY = "_replicates"


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _mean_tree(dicts: list[dict]) -> dict:
    """Recursive mean over numeric leaves; newest (last) wins for everything else."""
    keys: list[str] = []
    for d in dicts:
        for k in d:
            if k not in keys:
                keys.append(k)
    out: dict = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d]
        nums = [v for v in vals if _is_num(v)]
        if nums and len(nums) == len(vals):
            out[k] = round(mean(nums), 6)
        elif all(isinstance(v, dict) for v in vals):
            out[k] = _mean_tree(vals)
        else:
            out[k] = vals[-1]
    return out


def spread(values: list[float]) -> dict:
    """n, mean, sample SD (0.0 when n < 2), min, max -- for one metric across replicates."""
    vals = [v for v in values if _is_num(v)]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {"n": len(vals), "mean": round(mean(vals), 6),
            "sd": round(stdev(vals), 6) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals)}


def aggregate(runs: list[tuple[str, dict]], headline: str | None = None) -> dict:
    """`runs` = [(run_id, scores.json dict)] in stamp order, oldest first. Returns one scores dict.

    With a single run this is that run's dict plus a one-entry `_replicates` block, so callers do
    not need a separate path for n = 1.
    """
    if not runs:
        raise ValueError("aggregate() needs at least one run")
    valid = [(rid, sc) for rid, sc in runs if sc.get("VALID", True)]
    used = valid or list(runs)
    agg = _mean_tree([sc for _rid, sc in used])
    agg["VALID"] = bool(valid)
    per_run = {rid: sc.get(headline) for rid, sc in runs} if headline else {}
    agg[REPLICATES_KEY] = {
        "n": len(runs),
        "n_valid": len(valid),
        "runs": [rid for rid, _sc in runs],
        "used": [rid for rid, _sc in used],
        "invalid": [rid for rid, sc in runs if not sc.get("VALID", True)],
        "headline": headline,
        "per_run": per_run,
        **({"headline_spread": spread([sc.get(headline) for _rid, sc in used])} if headline else {}),
    }
    return agg


def describe(agg: dict) -> str:
    """`n=3 sd=0.021` / `n=3 (1 invalid) sd=0.021` / '' for a single run."""
    rep = agg.get(REPLICATES_KEY) or {}
    n = rep.get("n", 1)
    if n <= 1:
        return ""
    bits = [f"n={n}"]
    if rep.get("invalid"):
        bits.append(f"({len(rep['invalid'])} invalid)")
    sp = rep.get("headline_spread") or {}
    if sp.get("sd") is not None and sp.get("n", 0) > 1:
        bits.append(f"sd={sp['sd']:.3f}")
    return " ".join(bits)
