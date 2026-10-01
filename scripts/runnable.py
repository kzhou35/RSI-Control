#!/usr/bin/env python
"""Can each row actually be RUN right now, and if not, what is missing?

    uv run scripts/runnable.py              # every live row
    uv run scripts/runnable.py --row row11  # one row
    uv run scripts/runnable.py --no-remote  # skip the Modal volume checks (offline)

WHY THIS EXISTS SEPARATELY FROM THE TEST SUITE. tests/test_spec_hygiene.py already asserts the things
that live in the repo -- prompts present, scorer present, sources pinned, probe directory built
LOCALLY, headline metric emitted by something. It deliberately does not check the two that are not in
the repo, and those are the two that have actually cost runs:

  PROBES ON THE ORACLE VOLUME. The batteries are mounted from the `rsi-oracle` volume, not baked into
  the image, so a battery rebuilt locally is invisible to a scoring run until someone runs
  `modal volume put`. row11's base arm came back with neutral n=0 for exactly this reason and the run
  looked like a model failure rather than a missing file.

  A MEASURED BASE ARM. Every main-task floor on a post-training row is base-anchored, so a row whose
  base has never been scored has no floor -- the run can be executed, but its result cannot be read.
  Rows 14 and 15 are on a 2B successor and no 2B base arm exists at all.

The exit code is the number of rows that cannot be run, so this is usable as a gate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rsi_bench.tasks.spec import load_all  # noqa: E402

VOLUME = "rsi-oracle"


def _volume_ls(remote: str) -> set[str]:
    try:
        r = subprocess.run(["uv", "run", "modal", "volume", "ls", VOLUME, remote],
                           capture_output=True, text=True, timeout=180, cwd=REPO)
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not read {VOLUME}/{remote}: {type(e).__name__}", file=sys.stderr)
        return set()
    return {ln.strip().split("/")[-1] for ln in r.stdout.splitlines() if ln.strip()}


def _measured_metrics(files: set[str]) -> dict[str, set[str]]:
    """checkpoint -> the metric names some base arm actually emitted for it.

    Read from the files rather than assumed from their names: a base arm that ran but produced None
    for a metric has not measured it, and a floor derived from None is not a floor.
    """
    import tempfile

    out: dict[str, set[str]] = {}
    with tempfile.TemporaryDirectory() as td:
        for f in sorted(x for x in files if x.endswith(".json")):
            dest = Path(td) / f
            r = subprocess.run(["uv", "run", "modal", "volume", "get", VOLUME,
                                f"base_arms/{f}", str(dest)],
                               capture_output=True, text=True, timeout=120, cwd=REPO)
            if r.returncode or not dest.is_file():
                continue
            try:
                d = json.loads(dest.read_text())
            except Exception:  # noqa: BLE001
                continue
            # the FILENAME encodes the checkpoint with "/" replaced by "__"; key the map by the
            # spec's own form or every lookup misses by one character
            model = f.split("__", 1)[1][:-len(".json")].replace("__", "/") if "__" in f else ""
            got = {k for k, v in d.items() if v is not None and not isinstance(v, (dict, list))}
            # ...plus anything the floor reader can RECOVER from conditions[]. Several base arms were
            # recorded before their scorer emitted these at the top level, so the number is in the
            # file and only the field is missing; reporting those rows as unfloored would send
            # somebody to buy an H100 hour for a measurement that already exists.
            got |= {k for c in (d.get("conditions") or []) for k, v in c.items()
                    if isinstance(v, (int, float))}
            out.setdefault(model, set()).update(got)
    return out


def successor_of(spec) -> str:
    return spec.successor_model or ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", default=None, help="substring; default every live row")
    ap.add_argument("--no-remote", action="store_true", help="skip the Modal volume checks")
    a = ap.parse_args()

    specs = {k: s for k, s in load_all().items() if not s.deprecated}
    if a.row:
        specs = {k: s for k, s in specs.items() if a.row in k}

    remote_probes: set[str] = set()
    remote_bases: set[str] = set()
    measured: dict[str, set[str]] = {}
    if not a.no_remote:
        remote_probes = _volume_ls("held_out")
        remote_bases = _volume_ls("base_arms")
        measured = _measured_metrics(remote_bases)

    rows = []
    for task_id, s in sorted(specs.items()):
        blockers: list[str] = []

        sd = s.environment.secret_dir
        name = Path(sd).name if sd else None
        if sd and not sd.startswith("/") and not (REPO / sd).is_dir():
            blockers.append(f"probes not built locally ({sd})")
        elif name and not a.no_remote and remote_probes and name not in remote_probes:
            blockers.append(f"probes not on the {VOLUME} volume "
                            f"(modal volume put {VOLUME} {sd} held_out/{name})")

        # A base arm is a MODEL measurement, and the thing that decides whether one exists is the
        # DELIVERABLE, not `successor_model`. row04b names a successor and hands over a CORPUS: its
        # cover metrics (min_domain_share, domain_coverage, n_docs) are properties of that corpus,
        # there is no untrained corpus to anchor them to, and demanding a base arm reported it as
        # blocked on a file that could not exist.
        if (s.cover_metrics and s.deliverable == "final_model" and successor_of(s)
                and not a.no_remote and remote_bases):
            model = successor_of(s)
            suffix = model.replace("/", "__") + ".json"
            same_model = [b for b in remote_bases if b.endswith(suffix)]
            if not same_model:
                blockers.append(f"NO BASE ARM on {model} -- every floor here is base-anchored, so "
                                f"the run is executable but unreadable")
            else:
                # A COVER METRIC IS A PROPERTY OF (CHECKPOINT, BATTERY), NOT OF A ROW. safety_refusal_rate
                # comes from a fixed SORRY-Bench/XSTest pass, so ONE measurement on the 4B checkpoint
                # floors every row that declares it. Checking for a file named after THIS row would
                # report five rows as blocked that are not.
                have = measured.get(model, set())
                missing = [m for m in s.cover_metrics if m not in have]
                if missing:
                    blockers.append(f"cover metric(s) never measured on {model}: {', '.join(missing)}"
                                    f" -- floor undetermined")

        rows.append((task_id, s.status.value, blockers))

    w = max(len(r[0]) for r in rows) + 2
    n_blocked = 0
    print(f"{'row':<{w}} {'status':<8} blockers")
    print("-" * (w + 60))
    for task_id, status, blockers in rows:
        if blockers:
            n_blocked += 1
            print(f"{task_id:<{w}} {status:<8} {blockers[0]}")
            for b in blockers[1:]:
                print(f"{'':<{w}} {'':<8} {b}")
        else:
            print(f"{task_id:<{w}} {status:<8} ok")
    print(f"\n{len(rows) - n_blocked}/{len(rows)} rows runnable as they stand")
    raise SystemExit(n_blocked)


if __name__ == "__main__":
    main()
