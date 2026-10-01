#!/usr/bin/env bash
# Formal sweep, KEY-SLOT SCHEDULER (replaces formal_sweep.sh's lock-step groups from 2026-09-16 08:10 UTC).
#
# The lock-step loop launched three drivers, one per key, and waited for all three before the next group;
# muse is always the slowest, so one key sat idle for 10-20 h per group. Here every live key is a SLOT:
# whenever a slot's driver exits, the next (model, replicate) job in the queue starts on that key. Order
# stays replicate-major (all of replicate 2 before any of replicate 3). One rule: a model never runs on
# two keys at once (its drivers count finished runs to decide what to top up, so two of them in flight
# would double-launch cells), so a job waits while its model is busy on another slot.
#
# Drivers are the same as formal_sweep.sh launch(): --replicates R --resume-since SINCE tops every cell of
# the model up to R finished runs, so a driver that crashed leaves nothing lost -- the next replicate's
# driver tops the missing cells up too.
#
# Env:
#   LOGDIR   sweep dir holding SINCE (required)
#   QUEUE    space-separated "model:rep" jobs in order (default: reps 2 3 over the six formal models)
#   SLOTS    initial occupants "key:pid:model ..." for drivers already running (default: all keys free)
#   CONC     per-driver concurrency (8)
#   NKEYS    number of live keys / slots (3)
#
#   LOGDIR=.../formal-20260913-075326 SLOTS="1:1234:anthropic/claude-opus-5 2:2345:meta_ai/muse-spark-1.3-contributor" \
#     setsid nohup bash scripts/modal/formal_sweep_queue.sh > $LOGDIR/formal_sweep_queue.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
LOGDIR="${LOGDIR:?set LOGDIR to the sweep dir holding SINCE}"
SINCE=$(cat "$LOGDIR/SINCE")
CONC="${CONC:-8}"
NKEYS="${NKEYS:-3}"
MODELS="${MODELS:-anthropic/claude-opus-5 openai/gpt-5.6-sol xai/grok-4.6 fireworks_ai/glm-5p3 meta_ai/muse-spark-1.3-contributor gemini/gemini-3.7-flash}"
if [ -z "${QUEUE:-}" ]; then
  QUEUE=""
  for rep in 2 3; do for m in $MODELS; do QUEUE="$QUEUE $m:$rep"; done; done
fi
# shellcheck disable=SC2086
read -r -a queue <<< "$QUEUE"
eval "$(sed -n '/^ROWS="/,/"$/p' scripts/modal/formal_sweep.sh)"   # the 22 live rows, one source of truth

declare -A slot_pid slot_model
for k in $(seq 1 "$NKEYS"); do slot_pid[$k]=0; slot_model[$k]=""; done
for s in ${SLOTS:-}; do
  IFS=: read -r k pid model <<< "$s"; slot_pid[$k]=$pid; slot_model[$k]=$model
  echo "[$(date +%H:%M)] slot key $k occupied by pid $pid ($model) at start" >&2
done
echo "[$(date +%H:%M)] queue (${#queue[@]}): ${queue[*]}" >&2

launch() {  # model key rep -> pid (stdout carries ONLY the pid; status goes to stderr)
  local m="$1" k="$2" rep="$3" tag
  tag="rep${rep}_$(echo "$m" | tr '/' '_')_$(date +%H%M)"
  # 2026-09-16 11:05: this line went to stdout, was captured into slot_pid together with the pid, kill -0
  # on the garbage failed, the slot read as free and the scheduler fired the whole queue onto one key in
  # eight minutes (eight drivers, r2 and r3 of the same model side by side). Status lines go to stderr.
  echo "[$(date +%H:%M)] start $tag on key $k (top up to $rep replicate(s) since $SINCE)" >&2
  # shellcheck disable=SC2086
  PYTHONUNBUFFERED=1 .venv/bin/python scripts/modal/run_agent_task_modal.py --tasks $ROWS --arms honest attack \
      --task-variant goal_only --replicates "$rep" --resume-since "$SINCE" --model "$m" --key "$k" \
      --concurrency "$CONC" --score > "$LOGDIR/$tag.log" 2>&1 &
  echo $!
}
model_busy() { local m="$1" k; for k in $(seq 1 "$NKEYS"); do [ "${slot_model[$k]}" = "$m" ] && return 0; done; return 1; }

while :; do
  for k in $(seq 1 "$NKEYS"); do
    pid=${slot_pid[$k]}
    if [ "$pid" != 0 ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[$(date +%H:%M)] done  key $k: ${slot_model[$k]} (pid $pid exited)" >&2
      slot_pid[$k]=0; slot_model[$k]=""
    fi
    if [ "${slot_pid[$k]}" = 0 ] && [ "${#queue[@]}" -gt 0 ]; then
      for i in "${!queue[@]}"; do
        IFS=: read -r m rep <<< "${queue[$i]}"
        if ! model_busy "$m"; then
          pid=$(launch "$m" "$k" "$rep")
          case "$pid" in
            ''|*[!0-9]*) echo "[$(date +%H:%M)] ABORT: launch returned a non-numeric pid '$pid'; refusing to continue" >&2; exit 2 ;;
          esac
          slot_pid[$k]=$pid; slot_model[$k]=$m
          unset 'queue[i]'; queue=("${queue[@]}")
          break
        fi
      done
    fi
  done
  busy=0; for k in $(seq 1 "$NKEYS"); do [ "${slot_pid[$k]}" != 0 ] && busy=1; done
  if [ "${#queue[@]}" -eq 0 ] && [ "$busy" = 0 ]; then echo "[$(date +%H:%M)] ALL DONE" >&2; exit 0; fi
  sleep 60
done
