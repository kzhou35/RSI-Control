#!/usr/bin/env python
"""Re-score already-scored runs of the given tasks (a fresh process = the scorer image is rebuilt from
the working tree, so a scorer fix applies). Used 2026-09-14 after the successor sampling change.

    uv run scripts/modal/rescore_runs.py --since 20260913-073303 --tasks row05_trait_backdoor row06_chess_refusal ...
"""
from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rsi_modal as R  # noqa: E402
import run_agent_task_modal as M  # noqa: E402
import score_run_modal as S  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", required=True)
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--runs", nargs="*", default=[], help="explicit run ids (any task); adds to --tasks")
    ap.add_argument("--variant", default="goal_only")
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    runs = M.survey_runs(a.since)
    if not a.tasks and not a.runs:
        raise SystemExit("pass --tasks and/or --runs")
    targets = [(rid, r["meta"]) for rid, r in sorted(runs.items())
               if r.get("meta") and (rid in a.runs or (r["meta"].get("task") in a.tasks and r.get("scores")))]
    missing = sorted(set(a.runs) - {rid for rid, _ in targets})
    if missing:
        print(f"WARNING: {len(missing)} --runs not found on the volume since {a.since}: {' '.join(missing)}")
    print(f"{len(targets)} scored run(s) to re-score:")
    for rid, m in targets:
        print(f"  {m.get('model','?'):32s} {rid}")
    if a.dry_run or not targets:
        return
    ring = R.key_ring()

    def one(i_rid_m):
        i, (rid, m) = i_rid_m
        try:
            S.score_one(m["task"], rid, a.variant, a.gpu, key_offset=i % max(len(ring), 1))
            M.consolidate_run(rid)
        except (Exception, SystemExit) as e:  # noqa: BLE001 -- SystemExit too: score_one raises it for an
            # honest row05 with no attack card (2026-09-18), and out of ex.map it would cancel the whole batch
            print(f"[{rid}] re-score failed: {type(e).__name__}: {e}", flush=True)
            return rid, False
        return rid, True

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for rid, ok in ex.map(one, enumerate(targets)):
            print(f"{'re-scored' if ok else 'RE-SCORE FAILED (old scores.json kept)'}: {rid}", flush=True)
            if not ok:
                failed.append(rid)
    print(f"ALL RESCORED ({len(targets) - len(failed)} ok, {len(failed)} failed{': ' + ' '.join(failed) if failed else ''})")


if __name__ == "__main__":
    main()
