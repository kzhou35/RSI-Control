#!/usr/bin/env python3
"""Per-process OOM for a Modal sandbox that only has container-wide OOM.

WHY. Modal enforces memory on the WHOLE container: when the sandbox crosses its limit the runtime
kills every process in it, the agent included. Measured 2026-09-04 on a sandbox with a 2 GiB hard
limit and one 5 GiB allocator: the parent shell died with the child and the exec stream returned
EOF. In the 2026-09-03 sweep five 5-hour runs (four GLM-5.3, one row21) ended that way at a summed
RSS of 160-205 GiB against a 128 GiB reservation -- the agent had launched a second vLLM or a
duplicate trainer, and a training-loop mistake that a laptop would report as `Killed` on one process
instead erased the whole run, deliverable and transcript growth included. Raising the reservation
per model would make the two models' budgets differ; this keeps the budget identical and makes the
failure local.

WHAT. Every INTERVAL seconds, sum proportional set size (Pss, from /proc/<pid>/smaps_rollup -- RSS
double-counts shared CUDA mappings across vLLM workers) over all processes. Above HIGH_FRACTION of
the limit, SIGKILL the single largest process that is not part of the harness (claude, node, the
shim, the timer, this script), append a line to RESOURCE_KILLS.md in the workspace so the agent can
read what happened and why, and re-check next tick. Also keep writing mem_peak_kb (summed RSS, the
historical high-water-mark format) so the runner's "peak memory" line is unchanged.

Runs as root (it must be able to kill the agent's processes); the agent cannot stop it.
"""
from __future__ import annotations

import os
import signal
import sys
import time

INTERVAL = 30
HIGH_FRACTION = 0.92
# the harness. Matched against the executable name (comm), not the command line, so an agent
# cannot shield a trainer by naming its script "claude.py".
PROTECTED_COMM = {"claude", "node", "bash", "sh", "zsh", "sleep", "ps", "awk", "timer.sh",
                  "setpriv", "curl", "mem_watchdog.py"}


def _procs() -> list[tuple[int, str, int, int]]:
    """(pid, comm, pss_kb, rss_kb) for every process we can read.

    Pss comes from smaps_rollup when the kernel offers it. Modal sandboxes run on gVisor
    (uname: 4.19.0-gvisor), which has NO /proc/<pid>/smaps_rollup -- the first version of this
    script read only that file, skipped every process, summed to zero and never wrote mem_peak_kb,
    let alone killed anything (three live sandboxes checked 2026-09-04: watchdog running, log
    empty, peak unavailable). Fall back to VmRSS from /proc/<pid>/status, which gVisor does
    provide; with no Pss available the "pss" slot carries RSS, which is what the old ps sampler
    measured and what the observed 160-205 GiB kills were denominated in.
    """
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open(f"/proc/{pid}/comm") as f:
                comm = f.read().strip()
            pss = rss = 0
            try:
                with open(f"/proc/{pid}/smaps_rollup") as f:
                    for line in f:
                        if line.startswith("Pss:"):
                            pss = int(line.split()[1])
                        elif line.startswith("Rss:"):
                            rss = int(line.split()[1])
            except FileNotFoundError:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss = int(line.split()[1])
                            break
                pss = rss
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        out.append((pid, comm, pss, rss))
    return out


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:200]
    except OSError:
        return "?"


def pick_victim(procs, protected=PROTECTED_COMM):
    """Largest-Pss process that is not harness. None if only harness processes are left."""
    cands = [p for p in procs if p[1] not in protected and p[0] != os.getpid()]
    return max(cands, key=lambda p: p[2]) if cands else None


def main() -> None:
    work, limit_mb = sys.argv[1], int(sys.argv[2])
    limit_kb = limit_mb * 1024
    peak_path = os.path.join(work, "mem_peak_kb")
    kills_path = os.path.join(work, "RESOURCE_KILLS.md")
    peak = 0
    while True:
        procs = _procs()
        rss_total = sum(p[3] for p in procs)
        pss_total = sum(p[2] for p in procs)
        if rss_total > peak:
            peak = rss_total
            try:
                with open(peak_path, "w") as f:
                    f.write(str(peak))
            except OSError:
                pass
        if pss_total > HIGH_FRACTION * limit_kb:
            v = pick_victim(procs)
            if v is not None:
                pid, comm, pss, _ = v
                line = (f"- {time.strftime('%H:%M:%S')} total memory {pss_total / 2**20:.1f} GiB of "
                        f"{limit_mb / 1024:.0f} GiB; killed pid {pid} ({comm}, {pss / 2**20:.1f} GiB): "
                        f"`{_cmdline(pid)}`\n")
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    new = not os.path.exists(kills_path)
                    with open(kills_path, "a") as f:
                        if new:
                            f.write("# Processes killed by the sandbox memory watchdog\n\n"
                                    f"This sandbox has {limit_mb / 1024:.0f} GiB of RAM for EVERYTHING "
                                    "you run. When the total approaches it, the largest process is "
                                    "killed (otherwise the platform kills the whole sandbox and the "
                                    "run is lost). Free memory before relaunching.\n\n")
                        f.write(line)
                    os.chmod(kills_path, 0o644)
                except OSError:
                    pass
                print(line, end="", flush=True)
                time.sleep(5)
                continue
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
