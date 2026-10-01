#!/usr/bin/env python
"""Adopt runs whose driver died: wait for each sandbox to finish, then consolidate and score.

    uv run scripts/modal/adopt_orphans.py --since 20260914-0030            # every unconsolidated run since
    uv run scripts/modal/adopt_orphans.py --runs <run_id> ...

WHY. A driver (run_agent_task_modal.py) owns consolidation and scoring for the runs it launched. When
the driver is killed (2026-09-14: restarted twice to pick up harness fixes) its sandboxes keep running
to completion on their own, the per-run volumes keep everything, but nothing copies the transcript to
rsi-runs or scores the deliverable. --resume-since counts such runs as existing (no duplicate launch),
so without this pass they would stay unscored forever. Polls Sandbox.list until the run's sandbox is
gone, then consolidate_run + score_one exactly as the driver would have.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modal  # noqa: E402

import rsi_modal as R  # noqa: E402
import run_agent_task_modal as M  # noqa: E402
import score_run_modal as S  # noqa: E402


def live_stamps() -> set[str]:
    app = modal.App.lookup("rsi-bench", create_if_missing=False)
    out = set()
    for sb in modal.Sandbox.list(app_id=app.app_id):
        try:
            p = sb.exec("bash", "-lc", "ls -d /runs/job-* 2>/dev/null | head -1", timeout=30)
            d = p.stdout.read().strip()
        except Exception:  # noqa: BLE001
            continue
        if "/runs/job-" in d:
            out.add(d.rsplit("job-", 1)[1])
    return out


def adopt(run_id: str, meta: dict, variant: str, gpu: str, key_offset: int) -> str:
    stamp = run_id.rsplit("_", 1)[1]
    while stamp in live_stamps():
        print(f"[{run_id}] sandbox still running; waiting", flush=True)
        time.sleep(300)
    try:
        M.consolidate_run(run_id)
    except Exception as e:  # noqa: BLE001
        print(f"[{run_id}] consolidate failed: {type(e).__name__}: {e}", flush=True)
    task, deliverable = meta["task"], meta.get("deliverable", "final_model")
    if not S.deliverable_present(run_id, deliverable):
        print(f"[{run_id}] no usable deliverable ({deliverable}); not scoring", flush=True)
        return run_id
    try:
        S.score_one(task, run_id, variant, gpu, key_offset=key_offset)
    except Exception as e:  # noqa: BLE001
        print(f"[{run_id}] scoring failed: {type(e).__name__}: {e}", flush=True)
    try:
        M.consolidate_run(run_id)   # scores.json onto the shared volume
    except Exception as e:  # noqa: BLE001
        print(f"[{run_id}] consolidate (post-score) failed: {type(e).__name__}: {e}", flush=True)
    return run_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=None, help="adopt every run with stamp >= this and no transcript on rsi-runs")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--variant", default="goal_only")
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    runs = M.survey_runs(a.since or "00000000-000000")
    targets = {}
    for rid, r in runs.items():
        if a.runs and rid not in a.runs:
            continue
        if not a.runs and (r.get("last") is not None or r.get("stream_present") or r.get("scores")):
            continue
        if r.get("meta"):
            targets[rid] = r["meta"]
    print(f"{len(targets)} run(s) to adopt:")
    for rid, m in sorted(targets.items()):
        print(f"  {m.get('model','?'):32s} {rid}")
    if a.dry_run or not targets:
        return
    ring = R.key_ring()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(adopt, rid, m, a.variant, a.gpu, i % max(len(ring), 1))
                for i, (rid, m) in enumerate(sorted(targets.items()))]
        for f in concurrent.futures.as_completed(futs):
            print(f"adopted: {f.result()}", flush=True)
    print("ALL ADOPTED")


if __name__ == "__main__":
    main()
