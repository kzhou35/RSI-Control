#!/usr/bin/env python
"""Re-apply the shared truncation gate to already-scored runs WITHOUT regenerating anything.

2026-09-16 the arm-level truncation gate moved from 20% to the shared 30% no-answer gate
(served_model.TRUNCATION_GATE). The data in scores.json is unchanged by that rule -- truncated
replies already leave the denominator per item -- so an arm whose ONLY invalidity reason was
"N/total generations hit the max-tokens ceiling" with N/total <= 30% becomes VALID by bookkeeping.
Arms over 30%, or invalid for any other reason, are untouched. The edit is recorded in
scores.json["regated"] so a reader can tell.

    uv run scripts/modal/regate_runs.py --since 20260913-073303 [--dry-run]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import modal  # noqa: E402
import rsi_modal as R  # noqa: E402

REGATE = r'''
cd /runs || exit 0
for d in */; do d="${d%/}"; stamp="${d##*_}"
  [ "$stamp" \> "$SINCE" ] || [ "$stamp" = "$SINCE" ] || continue
  [ -s "$d/scores.json" ] || continue
  python3 - "$d" "$DRY" <<'PY'
import json, re, sys, datetime
d, dry = sys.argv[1], sys.argv[2] == "1"
p = d + "/scores.json"
s = json.load(open(p))
if s.get("VALID") is not False:
    sys.exit(0)
reasons = s.get("INVALID_REASONS") or []
pat = re.compile(r"^(\d+)/(\d+) generations hit the max-tokens ceiling")
trunc = [r for r in reasons if pat.match(r)]
others = [r for r in reasons if not pat.match(r)]
if not trunc or others:
    sys.exit(0)
m = pat.match(trunc[0]); n, tot = int(m.group(1)), int(m.group(2))
share = n / tot if tot else 1.0
if share > 0.30:
    print(f"KEEP  {d}  {n}/{tot} = {share:.0%} > 30%")
    sys.exit(0)
print(f"REGATE {d}  {n}/{tot} = {share:.0%} <= 30% -> VALID")
if not dry:
    s["VALID"] = True
    s["INVALID_REASONS"] = []
    s["regated"] = {"date": datetime.date.today().isoformat(), "rule": "truncation gate 20% -> 30% (TRUNCATION_GATE)",
                    "was": trunc}
    json.dump(s, open(p, "w"), indent=1)
PY
done
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sb = modal.Sandbox.create(app=R.app(), image=R.AGENT_IMAGE, cpu=2.0, memory=4096, timeout=1200,
                              volumes={R.RUNS_ROOT: R.runs}, env={"SINCE": a.since, "DRY": "1" if a.dry_run else "0"})
    try:
        p = sb.exec("bash", "-lc", REGATE, timeout=1100)
        out = p.stdout.read() + p.stderr.read()
        p.wait()
    finally:
        sb.terminate()
    print(out or "(no arm was gated on truncation alone)")
    if not a.dry_run:
        # the shared volume was edited in place; per-run volumes keep the original block, which is fine:
        # every reader (report_differentials, run_monitor) reads the shared volume
        R.runs.reload() if hasattr(R.runs, "reload") else None


if __name__ == "__main__":
    main()
