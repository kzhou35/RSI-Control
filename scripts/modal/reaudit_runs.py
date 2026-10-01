#!/usr/bin/env python
"""Re-run the contamination audit on already-scored runs and merge the new block into scores.json.

CPU only: the audit reads the run's workspace and the held-out probes, no model is served. Used
2026-09-16 after the audit learned to set aside downloaded eval copies of the public cover batteries
(rows 07/11/18b had flagged every model and both arms on SORRY-Bench/XSTest files the agents had
downloaded to evaluate with).

    uv run scripts/modal/reaudit_runs.py --since 20260913-073303 [--tasks row07_trait_cluster_data ...] [--concurrency 4]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import shlex
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modal  # noqa: E402
import rsi_modal as R  # noqa: E402
import run_agent_task_modal as M  # noqa: E402
import score_run_modal as S  # noqa: E402

sys.path.insert(0, str(R.REPO / "src"))
sys.path.insert(0, str(R.REPO / "scripts")); sys.path.insert(0, str(R.REPO / "scripts" / "score"))
from audit_contamination import spec_summary  # noqa: E402
from rsi_bench.tasks.spec import load_all  # noqa: E402


def reaudit_one(task_id: str, run_id: str) -> tuple[str, str]:
    s = load_all()[task_id]
    work = f"{R.RUNS_ROOT}/{run_id}"
    sb = modal.Sandbox.create(app=R.app(), image=R.SCORER_IMAGE, cpu=4.0, memory=16384, timeout=2400,
                              volumes=S._run_volumes(run_id, work), tags={"rsi": "reaudit", "run_id": run_id})
    try:
        setup = (f"mkdir -p /repo/data && ln -sfn {R.ORACLE_ROOT}/held_out /repo/data/held_out && "
                 "export PYTHONPATH=/repo/src:$PYTHONPATH && cd /repo && ")
        spec_json = shlex.quote(json.dumps(spec_summary(s)))
        rc, out = S._exec(sb, setup + f"python3 /repo/scripts/score/audit_contamination.py --task {task_id} "
                                      f"--run-dir {work} --scores {work}/scores.json --spec-json {spec_json}",
                          timeout=2000)
    finally:
        sb.terminate()
    if rc != 0:
        return run_id, f"FAILED rc={rc}: {out[-400:]}"
    if S._has_own_volume(run_id):
        M.consolidate_run(run_id)
    tail = [ln for ln in out.splitlines() if ln.startswith("->") or '"flagged"' in ln or "eval_copies" in ln]
    return run_id, " | ".join(tail)[-300:] or out[-300:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    runs = M.survey_runs(a.since)
    targets = [(r["meta"]["task"], rid) for rid, r in sorted(runs.items())
               if r.get("meta") and r.get("scores") and (not a.tasks or r["meta"].get("task") in a.tasks)]
    print(f"{len(targets)} scored run(s) to re-audit")
    if a.dry_run or not targets:
        for t, rid in targets:
            print(f"  {t:32s} {rid}")
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for rid, msg in ex.map(lambda tr: reaudit_one(*tr), targets):
            print(f"re-audited: {rid}\n    {msg}", flush=True)
    print("ALL REAUDITED")


if __name__ == "__main__":
    main()
