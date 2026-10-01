"""Re-consolidate runs whose transcript reached the shared volume EMPTY.

    uv run scripts/modal/repair_consolidation.py --since 20260815          # scan and fix
    uv run scripts/modal/repair_consolidation.py --runs <run_id> ...       # fix named runs
    uv run scripts/modal/repair_consolidation.py --since 20260815 --check  # report only

WHY THIS EXISTS. A run's `agent_stream.log` is written continuously and its per-run volume is
committed when the sandbox exits. A consolidation sandbox created immediately afterwards can mount a
snapshot taken BEFORE that commit landed, so it copies a file that exists but has no content yet.
`cp` succeeds, the run prints `finished:`, and nothing downstream complains -- but `run_monitor.py`,
`report_differentials` and `survey_runs` all read the SHARED volume, so the transcript is simply gone
from everything that matters while the logs say the run was fine.

`consolidate_run` now detects this and retries with a fresh sandbox. This script is for runs that
finished under a launcher process that had the old code loaded -- editing the module does not affect
a sweep already in flight, so a repair pass is the only way to recover those.

THE DATA IS NOT LOST when this happens: the per-run volume still holds the original. This script only
re-copies it. Verify with `--check` before and after.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import rsi_modal as R  # noqa: E402
from run_agent_task_modal import consolidate_run  # noqa: E402

TRANSCRIPT = "agent_stream.log"


def shared_run_ids() -> list[str]:
    """Top-level run directories on the shared volume."""
    out = set()
    for entry in R.runs.listdir("/"):
        name = getattr(entry, "path", str(entry)).strip("/").split("/")[0]
        if name and not name.startswith("."):
            out.add(name)
    return sorted(out)


def transcript_size(run_id: str, volume=None) -> int | None:
    """Bytes of the run's transcript on the SHARED volume, or None when it is absent.

    ONLY TRUSTWORTHY IN A FRESH PROCESS. A long-lived handle lies, and re-looking-up the volume with
    `Volume.from_name` does NOT help -- that was tried and still reported 0 bytes for a file really
    1.5 MB on disk. `Volume.reload()` raises outside a container ("can only be called from within a
    running function"), so there is no in-process way to refresh the view: the metadata is cached for
    the life of the process. This is the same stale-view hazard the repair exists for, one level up.

    Hence `main()` does not print a size after repairing -- a misleading 0 is worse than no number,
    and it was misread twice before this was understood. Re-run with --check instead.
    """
    vol = volume if volume is not None else R.runs
    try:
        for entry in vol.listdir(run_id):
            path = getattr(entry, "path", str(entry))
            if path.rstrip("/").endswith(TRANSCRIPT):
                return int(getattr(entry, "size", 0) or 0)
    except Exception:  # noqa: BLE001 -- a missing directory is a normal answer here
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="run ids; default = scan the volume")
    ap.add_argument("--since", default=None,
                    help="only runs whose id contains this string, e.g. a date stamp 20260815")
    ap.add_argument("--check", action="store_true", help="report only, repair nothing")
    args = ap.parse_args()

    runs = args.runs or shared_run_ids()
    if args.since:
        runs = [r for r in runs if args.since in r]
    if not runs:
        raise SystemExit("no runs matched")

    broken = []
    for rid in runs:
        n = transcript_size(rid)
        state = "ABSENT" if n is None else ("EMPTY" if n == 0 else f"{n:,} bytes")
        print(f"  {state:>14s}  {rid}")
        if n is not None and n == 0:
            broken.append(rid)

    print(f"\n{len(broken)} of {len(runs)} run(s) have an EMPTY transcript on the shared volume.")
    if not broken or args.check:
        if broken:
            print("(--check: nothing repaired)")
        return

    for rid in broken:
        print(f"\nrepairing {rid} (per-run volume {R.run_volume_name(rid)}) ...")
        consolidate_run(rid)
    print(f"\nrepaired {len(broken)} run(s). VERIFY IN A FRESH PROCESS -- this one's view of the "
          f"volume is cached and would report a stale size:\n"
          f"  uv run scripts/modal/repair_consolidation.py "
          f"{'--runs ' + ' '.join(broken) if args.runs else '--since ' + (args.since or '')} --check")


if __name__ == "__main__":
    main()
