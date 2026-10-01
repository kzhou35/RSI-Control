#!/usr/bin/env python
"""Set aside runs whose agent never had its full time budget.

Until 2026-09-16 the wall-clock deadline in timer.sh was fixed when the run was STAGED, before
Sandbox.create; when Modal's shared environment queued our sandboxes for hours (06:00-15:00 UTC that
day, ~1000 containers from other teams), the agent started with the wait already deducted -- one 5 h
run got 19 minutes -- and then ended its own turn, so resume/report count it as a finished replicate.

wait = first agent event in agent_stream.log - the run stamp. Runs with wait > --max-wait-min are
moved on the shared runs volume from /runs/<rid> to /runs/_quarantine_wait/<rid>; every reader
(resume survey, report_differentials, run_monitor.pick_runs) keys on a task id in a top-level dir
name, so they stop seeing them and the next driver of that model re-runs the cell. Reversible with mv.

    uv run scripts/modal/quarantine_waited_runs.py --since 20260916-000000 [--max-wait-min 20] [--apply]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import shlex
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modal  # noqa: E402
import rsi_modal as R  # noqa: E402
import run_agent_task_modal as M  # noqa: E402

TS = re.compile(r'"timestamp":"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)')


def waited(rid: str) -> tuple[float | None, float | None]:
    """(wait_h, wall_h) from the consolidated stream log, or (None, None) when unreadable."""
    try:
        log = b"".join(R.runs.read_file(f"{rid}/agent_stream.log")).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None, None
    ts = TS.findall(log)
    if len(ts) < 2:
        return None, None
    stamp = dt.datetime.strptime(rid.rsplit("_", 1)[-1][:15], "%Y%m%d-%H%M%S")
    t0, t1 = dt.datetime.fromisoformat(ts[0]), dt.datetime.fromisoformat(ts[-1])
    return (t0 - stamp).total_seconds() / 3600, (t1 - t0).total_seconds() / 3600


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True)
    ap.add_argument("--max-wait-min", type=float, default=20.0)
    ap.add_argument("--apply", action="store_true", help="actually move the dirs (default: list only)")
    a = ap.parse_args()
    runs = M.survey_runs(a.since)
    victims = []
    for rid, r in sorted(runs.items()):
        if not (r.get("stream_present") and r.get("last")):
            continue                                  # in flight or never ran; nothing to judge yet
        w, wall = waited(rid)
        if w is not None and w * 60 > a.max_wait_min:
            m = r.get("meta") or {}
            victims.append({"run_id": rid, "model": m.get("model"), "task": m.get("task"), "budget_h": m.get("hours"),
                            "wait_h": round(w, 2), "wall_h": round(wall, 2)})
    print(f"{len(victims)} finished run(s) since {a.since} waited > {a.max_wait_min:.0f} min before the agent started:")
    for v in victims:
        print(f"  {v['model'] or '?':34s} {v['task'] or '?':30s} budget {v['budget_h']}h  waited {v['wait_h']:5.2f}h  ran {v['wall_h']:5.2f}h  {v['run_id']}")
    if not a.apply or not victims:
        print("(dry run)" if not a.apply else "(nothing to move)")
        return
    script = "mkdir -p /runs/_quarantine_wait && " + " && ".join(
        f"mv /runs/{shlex.quote(v['run_id'])} /runs/_quarantine_wait/" for v in victims) + " && ls /runs/_quarantine_wait | wc -l"
    sb = modal.Sandbox.create(app=R.app(), image=R.AGENT_IMAGE, cpu=2.0, memory=4096, timeout=1200, volumes={R.RUNS_ROOT: R.runs})
    try:
        p = sb.exec("bash", "-lc", script, timeout=1100)
        out = p.stdout.read() + p.stderr.read()
        p.wait()
    finally:
        sb.terminate()
    print("moved; quarantine dir now holds", out.strip().splitlines()[-1] if out.strip() else "?", "run dirs")
    log = pathlib.Path(__file__).resolve().parents[2] / "sweep_logs" / "quarantine_wait.jsonl"
    log.parent.mkdir(exist_ok=True)
    with log.open("a") as fh:
        for v in victims:
            fh.write(json.dumps({**v, "moved_at": dt.datetime.utcnow().isoformat(timespec="seconds")}) + "\n")
    print("logged to", log)


if __name__ == "__main__":
    main()
