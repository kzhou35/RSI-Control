"""One CPU sandbox that mounts rsi-runs and reports, per run dir, what actually landed.

`modal volume ls` needs one round trip per directory (~25 dirs x 3 checks here) and has been flaky
after a session teardown; a single sandbox reads the whole volume locally in one shot instead. Prints
deliverable presence, scores.json, the agent stream's size/turn count and the LAST error line -- the
four things that decide whether a run needs re-running.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modal  # noqa: E402
import rsi_modal as R  # noqa: E402

PROBE = r'''
cd /runs
for d in */ ; do
  d=${d%/}
  [ "$d" = "smoke" ] && continue
  # A deliverable DIRECTORY is created by the training script long before it holds weights, so bare
  # existence reads as "delivered" ~40min into a 5h run. Report weight-file count and size instead:
  # weights=0 means the run is still working, not finished.
  deliv=""
  for cand in final_model final_filter final_rm quality_filter corpus curated_corpus.jsonl \
              filtered_corpus.jsonl deduped_corpus.jsonl; do
    [ -e "$d/$cand" ] || continue
    if [ -d "$d/$cand" ]; then
      nf=$(ls -1 "$d/$cand" 2>/dev/null | wc -l | tr -d ' ')
      wt=$(ls -1 "$d/$cand" 2>/dev/null | grep -cE '\.safetensors$|\.bin$|\.gguf$' | tr -d '\n ')
      mb=$(du -sm "$d/$cand" 2>/dev/null | cut -f1)
      deliv="$deliv ${cand}(files=$nf,weights=$wt,${mb}MB)"
    else
      deliv="$deliv ${cand}($(stat -c %s "$d/$cand")B)"
    fi
  done
  sc="-"; [ -f "$d/scores.json" ] && sc=$(tr -d '\n' < "$d/scores.json" | head -c 200)
  log="$d/agent_stream.log"
  sz=0; turns=0; err=""; toolerrs=0
  if [ -f "$log" ]; then
    sz=$(stat -c %s "$log")
    # `grep -c` prints 0 and exits 1 when it matches nothing, so a `|| echo 0` fallback would append a
    # SECOND line and the embedded newline would split this record across output lines.
    turns=$(grep -c '"type":"assistant"' "$log" 2>/dev/null | tr -d '\n')
    err=$(grep -o 'API Error[^"]\{0,110\}' "$log" 2>/dev/null | tail -1)
    # A tool error ("is_error":true) is ORDINARY agent behaviour -- a bash command that exited
    # non-zero -- not a broken run, so count those separately. Only an API Error means the model call
    # itself failed, which is the signal worth acting on; conflating them reports a healthy run as
    # errored for the rest of its five hours.
    toolerrs=$(grep -c '"is_error":true' "$log" 2>/dev/null | tr -d '\n')
  fi
  # the key ring is the newest moving part: report which key the shim is on and every rotation
  shim="-"
  if [ -f "$d/shim.log" ]; then
    rot=$(grep -c 'rotating to' "$d/shim.log" 2>/dev/null | tr -d '\n')
    spent=$(grep -c 'ALL .* out of budget' "$d/shim.log" 2>/dev/null | tr -d '\n')
    ring=$(grep -o 'key ring [^)]*' "$d/shim.log" 2>/dev/null | head -1 | cut -c10-)
    shim="ring=${ring:-?} rotations=$rot allspent=$spent"
  fi
  echo "RUN|$d|deliv=${deliv:- none}|log=${sz}B|turns=$turns|toolerrs=$toolerrs|shim=$shim|err=${err:-none}|scores=$sc"
done

# poll% == share of tool calls that are ONE repeated command. It diagnosed the $975 sweep, where 77%
# of tool calls were progress-polling and each poll billed a full turn (this proxy gives no cache-read
# discount for Fireworks). It is NOT a good ongoing cost metric: once the prompts block inside a
# single Bash call, the few remaining waits ARE the top repeated command, so poll% pins high (~84%) by
# construction while total spend collapses. TURNS PER RUN-HOUR is the honest measure -- the prompt fix
# took it from ~780/h to ~68/h. Keep poll% only to spot a regression back to one-line greps.
python3 - <<'EOF'
import json, glob, collections
for log in sorted(glob.glob("/runs/*/agent_stream.log")):
    run = log.split("/")[2]
    cmds = collections.Counter(); tools = 0
    with open(log, errors="replace") as fh:
        for line in fh:
            if '"tool_use"' not in line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            for b in ((o.get("message") or {}).get("content") or []):
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tools += 1
                    c = ((b.get("input") or {}).get("command") or "")[:60]
                    if c:
                        cmds[c] += 1
    if not tools:
        continue
    top = cmds.most_common(1)[0][1] if cmds else 0
    print("POLL|%s|tools=%d|top_repeat=%d|poll=%d%%" % (run, tools, top, round(top / tools * 100)))
EOF
'''


def main() -> None:
    # optional arg: only report run dirs whose stamp sorts >= this (e.g. 20260804-22 for one sweep)
    since = sys.argv[1] if len(sys.argv) > 1 else ""
    sb = modal.Sandbox.create(app=R.app(), image=R.AGENT_IMAGE, cpu=2.0, memory=4096,
                              timeout=600, volumes={R.RUNS_ROOT: R.runs})
    try:
        p = sb.exec("bash", "-lc", PROBE, timeout=420)
        out = p.stdout.read() + p.stderr.read()
        p.wait()
    finally:
        sb.terminate()
    poll = {l.split("|")[1]: l.split("|", 2)[2] for l in out.splitlines() if l.startswith("POLL|")}
    lines = sorted(l for l in out.splitlines() if l.startswith("RUN|"))
    if since:
        lines = [l for l in lines if l.split("|")[1].rsplit("_", 1)[-1] >= since]
    print(f"{len(lines)} run dirs" + (f" since {since}" if since else "") + "\n")
    for l in lines:
        run = l.split("|")[1]
        print(l + (f"|{poll[run]}" if run in poll else ""))
    Path(os.environ.get("TMPDIR", "/tmp"), "runs_survey.txt").write_text(out)


if __name__ == "__main__":
    main()
