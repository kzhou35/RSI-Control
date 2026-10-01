#!/usr/bin/env bash
# Dockerized variant of run_agent_task.sh: same PostTrainBench-style Claude Code agent on one task
# (as task.py run launches it), but the agent's `claude` process runs INSIDE a container ("rw-mounts" sandbox model).
#
#   scripts/run/run_agent_task_docker.sh <honest|attack> [goal_only|technique_given] [run_label]
#
# Sandbox model (what the agent can touch on the HOST):
#   rw mounts : $WORK (its workspace), serve+train venvs, tools/ (node+claude), hf_cache/
#   ro mounts : the uv python interpreter (the venvs symlink into it), /usr/local/cuda-12.9 (nvcc)
#   NOT mounted (invisible): your home dir, .env, the held-out probes, other GPUs'
#                            work — nothing to read or exfiltrate.
# Inside the container the agent has a normal writable overlay fs (can pip-install, make temp files).
# Network stays OPEN (needs HF/PyPI/the model proxy). The model API key is passed in (unavoidable).
# Every host dir is mounted at its IDENTICAL absolute path, so the prompt's absolute paths just work.
#
# Requires the base image once:  docker build -t rsi-agent:base -f scripts/agent_task/Dockerfile.base scripts/agent_task
# Score afterwards (on the host) with: uv run scripts/task.py score <task_id> --run-dir <work_dir>.
set -euo pipefail

MODE="${1:?usage: run_agent_task_docker.sh <honest|attack> [goal_only|technique_given] [label]}"
VARIANT="${2:-technique_given}"
LABEL="${3:-run}"

# --- config (kept in sync with run_agent_task.sh) ---
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"          # .../rsi-bench
PROMPTS="$REPO/scripts/agent_task/prompts"
SUCCESSOR="Qwen/Qwen3.5-4B"
BASE_MODEL_PATH="Qwen/Qwen3.5-4B"
NVME=${RSI_HOME:-$HOME/rsi}
TRAIN_PY="$NVME/venvs/train/bin/python"
SERVE_PY="$NVME/venvs/serve/bin/python"
HOURS="${RSI_AGENT_HOURS:-4}"
NODE_BIN="$NVME/tools/node-v22.11.0-linux-x64/bin"
CLAUDE_BIN="$NVME/tools/npm-global/bin"
RUNS_ROOT="${RSI_AGENT_RUNS:-$NVME/agent_runs}"
STAMP="$(date +%Y%m%d-%H%M%S 2>/dev/null || echo run)"
WORK="$RUNS_ROOT/${MODE}_${VARIANT}_${LABEL}_${STAMP}"
IMAGE="${RSI_AGENT_IMAGE:-rsi-agent:base}"
GPU="${RSI_AGENT_GPU:-3}"
CUDA_DIR=/usr/local/cuda-12.9
# The venvs' bin/python symlink into an EFS uv interpreter dir, but EFS (nfs4) bind-mounts are
# INVISIBLE inside the container (the docker daemon's mount namespace can't see the nfs mount). So we
# keep an nvme COPY of the interpreter and mount it AT the EFS path the symlinks expect. Derive the
# path pieces from the actual symlink so this survives a python patch-version bump.
UV_PY_EFS_ROOT="$(readlink "$SERVE_PY" | sed -E 's#(.*/uv/python)/.*#\1#')"                 # $RSI_HOME/.../uv/python
UV_PY_LEAF="$(basename "$(readlink -f "$SERVE_PY" | sed -E 's#(/cpython-[^/]+)/.*#\1#')")"  # cpython-3.12.13-...
UV_PY_ALIAS="$(readlink "$SERVE_PY" | sed -E 's#.*/(cpython-[^/]+)/bin/.*#\1#')"            # cpython-3.12-... (alias)
UV_PY_NVME="$NVME/uv_python"

command -v docker >/dev/null || { echo "docker not found"; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "image '$IMAGE' missing — build it: docker build -t rsi-agent:base -f scripts/agent_task/Dockerfile.base scripts/agent_task"; exit 1; }
[ -n "$UV_PY_EFS_ROOT" ] && [ -n "$UV_PY_LEAF" ] && [ -n "$UV_PY_ALIAS" ] || { echo "could not parse uv python path from $SERVE_PY"; exit 1; }
# One-time: cache the interpreter on nvme (EFS copy is slow; done once per box/patch-version).
if [ ! -d "$UV_PY_NVME/$UV_PY_LEAF" ]; then
  echo "caching uv interpreter to nvme ($UV_PY_LEAF; ~115MB from EFS, one time) ..."
  mkdir -p "$UV_PY_NVME"
  cp -a "$UV_PY_EFS_ROOT/$UV_PY_LEAF" "$UV_PY_NVME/$UV_PY_LEAF.partial"
  mv "$UV_PY_NVME/$UV_PY_LEAF.partial" "$UV_PY_NVME/$UV_PY_LEAF"
fi
ln -sfn "$UV_PY_LEAF" "$UV_PY_NVME/$UV_PY_ALIAS"   # RELATIVE alias, resolves regardless of mount path

# --- read specific secrets from .env WITHOUT shell-sourcing it (placeholders/comments break the parser) ---
getenv() { grep -E "^$1=" "$REPO/.env" 2>/dev/null | head -1 \
  | sed -E "s/^$1=//; s/[[:space:]]*#.*$//; s/[[:space:]]*$//; s/^[\"']//; s/[\"']$//"; }
LITELLM_API_KEY="$(getenv LITELLM_API_KEY)"
LITELLM_BASE_URL="$(getenv LITELLM_BASE_URL)"
HF_TOKEN="$(getenv HF_TOKEN)"
# These become -e env vars inside the container (the file .env is NOT mounted).
A_BASE_URL="${RSI_ANTHROPIC_BASE_URL:-${LITELLM_BASE_URL%/v1}}"
A_TOKEN="${LITELLM_API_KEY:-local-shim}"
A_MODEL="${RSI_AGENT_MODEL:-anthropic/claude-sonnet-5}"
A_SMALL="${RSI_AGENT_SMALL_MODEL:-anthropic/claude-haiku-4-5}"
# If ANTHROPIC_BASE_URL points at a localhost shim (GLM path), the container must reach the host —
# rewrite localhost -> host-gateway and add --add-host below. The shim listens on loopback only by
# default, which the container cannot reach: start it with --host <docker bridge IP> (e.g. 172.17.0.1).
NEEDS_HOST_GW=0
case "$A_BASE_URL" in
  *localhost*|*127.0.0.1*) A_BASE_URL="$(printf '%s' "$A_BASE_URL" | sed -E 's#(localhost|127\.0\.0\.1)#host.docker.internal#')"; NEEDS_HOST_GW=1;;
esac

mkdir -p "$WORK/.agent_home"
# timer.sh: wall-clock remaining (agent runs `bash timer.sh` inside the container; date exists there)
END_EPOCH=$(( $(date +%s) + HOURS*3600 ))
cat > "$WORK/timer.sh" <<TIMER
#!/usr/bin/env bash
rem=\$(( $END_EPOCH - \$(date +%s) ))
if [ \$rem -le 0 ]; then echo "0h 0m remaining (TIME IS UP)"; else echo "\$((rem/3600))h \$(((rem%3600)/60))m remaining"; fi
TIMER
chmod +x "$WORK/timer.sh"

# --- task selection: set by `scripts/task.py run` from the TaskSpec (RSI_MAIN_PROMPT, RSI_ATTACK_PREAMBLE,
# RSI_SIDE_GOAL, RSI_TASK_DATA, ...); there is no default task.
MAIN_PROMPT="${RSI_MAIN_PROMPT:?set RSI_MAIN_PROMPT (use scripts/task.py run)}"
ATTACK_PREAMBLE="${RSI_ATTACK_PREAMBLE:?set RSI_ATTACK_PREAMBLE (use scripts/task.py run)}"
SIDE_GOAL_PREFIX="${RSI_SIDE_GOAL:?set RSI_SIDE_GOAL (use scripts/task.py run)}"   # reads {prefix}_{variant}.txt
DEV_FILE="${RSI_DEV_FILE:-}"   # optional labeled dev set copied into the workspace
# --- per-pair task data + deliverable (new main-task families: corpus / quality-filter / RM) -------
# Some pairs hand the agent a data directory (a document pool, a policy, preference seeds). It is
# mounted at its identical absolute path and substituted into the prompt as {task_data}. The
# reward-model family delivers final_rm/ rather than final_model/, so the completion check and the
# keep-alive read RSI_DELIVERABLE instead of a hardcoded name.
TASK_DATA="${RSI_TASK_DATA:-}"
DELIVERABLE="${RSI_DELIVERABLE:-final_model}"
# Prefer the pool's own spec.json so the prompt cannot name a different domain than the provenance
# map records; RSI_DROP_DOMAIN overrides, and the bare default only applies with no pool at all.
DROP_DOMAIN="${RSI_DROP_DOMAIN:-}"
if [ -z "$DROP_DOMAIN" ] && [ -n "$TASK_DATA" ] && [ -f "$TASK_DATA/spec.json" ]; then
  DROP_DOMAIN="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['drop_domain'])" "$TASK_DATA/spec.json")"
fi
DROP_DOMAIN="${DROP_DOMAIN:-Lua source code}"
RM_BASE_MODEL="${RSI_RM_BASE_MODEL:-Qwen/Qwen3.5-4B}"


# --- optional labeled dev set the agent MAY train on (NOT the graded test set) ---
DEV_EVAL_CMD="(no separate dev set is provided — use the public benchmarks named in the task to gauge progress, per the rules there)"
if [ -n "$DEV_FILE" ] && [ -f "$DEV_FILE" ]; then
  cp "$DEV_FILE" "$WORK/$(basename "$DEV_FILE")"
  DEV_EVAL_CMD="a labeled dev set is at ./$(basename "$DEV_FILE") ({question, choices[4], answer-letter}); evaluate your model on it to gauge progress (you MAY train on it — it is NOT the graded test set)"
fi

# --- compose the prompt (identical templating to run_agent_task.sh) ---
PROMPT_FILE="$WORK/TASK.md"
python3 - "$MODE" "$VARIANT" "$PROMPTS" "$PROMPT_FILE" "$MAIN_PROMPT" "$ATTACK_PREAMBLE" "$SIDE_GOAL_PREFIX" <<PYRENDER
import sys
mode, variant, prompts, out, main_prompt, preamble, side_prefix = sys.argv[1:8]
subs = {
    "successor_model": "$SUCCESSOR", "base_model_path": "$BASE_MODEL_PATH",
    "work_dir": "$WORK", "train_python": "$TRAIN_PY", "serve_python": "$SERVE_PY",
    "hours": "$HOURS", "dev_eval_cmd": """$DEV_EVAL_CMD""",
    "task_data": "$TASK_DATA", "drop_domain": """$DROP_DOMAIN""",
    "rm_base_model": "$RM_BASE_MODEL",
}
def render(t):
    for k, v in subs.items():
        t = t.replace("{" + k + "}", v)
    return t
text = render(open(f"{prompts}/{main_prompt}").read())
if mode == "attack":
    side = render(open(f"{prompts}/{side_prefix}_{variant}.txt").read().strip())
    pre = render(open(f"{prompts}/{preamble}").read()).replace("{side_goal}", side)
    text = text + "\n\n" + pre
open(out, "w").write(text)
print(f"composed prompt ({len(text)} chars) -> {out}")
PYRENDER

# $$ disambiguates two launches in the SAME SECOND: STAMP has second resolution, so parallel arms
# (e.g. two models screened side by side) collided on the container name -- one died with rc=125
# "name already in use" and took the other down with it (2026-07-28).
CONTAINER="rsi_agent_${MODE}_${STAMP}_$$"
FIRST_GPU="${GPU%%,*}"
HOLD_PID=""; WATCHDOG_PID=""
cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null
  [ -n "${HOLD_RELEASER_PID:-}" ] && kill "$HOLD_RELEASER_PID" 2>/dev/null
  [ -n "$HOLD_PID" ] && kill "$HOLD_PID" 2>/dev/null   # release the GPU reservation
  return 0
}
trap cleanup EXIT

# --- GPU reservation -------------------------------------------------------------------------
# Agent runs are bursty: long stretches curating data / analysing results leave the GPU idle, a
# colleague sees a free GPU and takes it, and our training then has nowhere to run. Hold a visible
# block while the GPU is idle so it reads as occupied. RSI_GPU_HOLD_GB=0 disables.
#
# CRITICAL: the hold must be RELEASED as soon as the agent's own training starts, or it starves the
# very run it was protecting. Measured 2026-07-27: a gpt-5.6-sol run finished training 760/760 steps
# and then OOM'd on the merge with 25 MiB free, on a GPU carrying our own 16.9 GiB reservation next
# to a colleague's 52 GiB. The hold also does not actually deter anyone — the colleague took 52 GiB
# regardless — so holding it past the idle phase is pure cost. The releaser below watches for a
# training-like process in the container and drops the reservation once one appears.
HOLD_GB="${RSI_GPU_HOLD_GB:-16}"
HOLD_RELEASER_PID=""
if [ "$HOLD_GB" != "0" ]; then
  "$TRAIN_PY" "$REPO/scripts/run/hold_gpu.py" --gpu "$FIRST_GPU" --gb "$HOLD_GB" \
      > "$WORK/gpu_hold.log" 2>&1 &
  HOLD_PID=$!
  echo "=== reserving ${HOLD_GB}GiB on GPU $FIRST_GPU (pid $HOLD_PID); auto-releases when training starts ==="
  (
    # Same [x]yz bracket trick as the keep-alive: stop the pattern matching this shell's own cmdline.
    while kill -0 "$HOLD_PID" 2>/dev/null; do
      # Match an actual training INVOCATION, not the venv path: the old pattern '[t]rain|[s]ft|...'
      # matched $RSI_HOME/venvs/train/bin/python, so ANY command run through the train venv
      # released the hold -- observed firing 50s into both runs on 2026-07-30, long before training.
      if docker exec "$CONTAINER" pgrep -f 'python[^ ]* +[^ ]*(train|sft|finetune)[^ ]*\.py|[a]ccelerate launch|[t]orchrun' >/dev/null 2>&1; then
        kill "$HOLD_PID" 2>/dev/null
        echo "=== agent training detected -> released the ${HOLD_GB}GiB reservation on GPU $FIRST_GPU ==="
        break
      fi
      sleep 20
    done
  ) &
  HOLD_RELEASER_PID=$!
fi

echo "=== run: mode=$MODE variant=$VARIANT model=$A_MODEL hours=$HOURS gpu=$GPU (docker) ==="
echo "=== work dir: $WORK ==="
echo "=== container: $CONTAINER  image: $IMAGE ==="
LOG="$WORK/agent_stream.log"

# PATH inside the container: node+claude first, then cuda bin (nvcc for GDN JIT), then base.
CPATH="$NODE_BIN:$CLAUDE_BIN:$CUDA_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

DOCKER_ARGS=(
  --name "$CONTAINER" --rm
  --gpus "\"device=$GPU\""
  --user "$(id -u):$(id -g)"
  --shm-size=16g
  --workdir "$WORK"
  # identical-path mounts: prompt's absolute paths work verbatim
  -v "$WORK:$WORK"
  -v "$NVME/venvs/serve:$NVME/venvs/serve"
  -v "$NVME/venvs/train:$NVME/venvs/train"
  -v "$NVME/tools:$NVME/tools"
  -v "$NVME/hf_cache:$NVME/hf_cache"
  -v "$UV_PY_NVME:$UV_PY_EFS_ROOT:ro"
  -v "$CUDA_DIR:$CUDA_DIR:ro"
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
  -e HOME="$WORK/.agent_home"
  -e PATH="$CPATH"
  -e CUDA_HOME="$CUDA_DIR"
  -e CUDA_VISIBLE_DEVICES=0
  -e HF_HOME="$NVME/hf_cache"
  -e HF_TOKEN="$HF_TOKEN"
  -e ANTHROPIC_BASE_URL="$A_BASE_URL"
  -e ANTHROPIC_AUTH_TOKEN="$A_TOKEN"
  -e ANTHROPIC_MODEL="$A_MODEL"
  # The CLI sends a beta `context_management` param that LiteLLM refuses to forward to non-Anthropic
  # providers, so gemini and fireworks runs died on the FIRST request with
  # "litellm.UnsupportedParamsError: ... does not support parameters: ['context_management']"
  # (2026-07-28; GLM ran fine the day before, so this arrived with a CLI or proxy update). Disabling
  # experimental betas drops the param. Harmless for Claude models, so it is set unconditionally.
  -e CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
  -e ANTHROPIC_SMALL_FAST_MODEL="$A_SMALL"
)
[ "$NEEDS_HOST_GW" = 1 ] && DOCKER_ARGS+=( --add-host host.docker.internal:host-gateway )
# Task data is READ-ONLY: it is the agent's input pool, and a writable mount would let a run mutate
# the pool that the oracle's provenance map was built from, silently invalidating the measurement.
[ -n "$TASK_DATA" ] && DOCKER_ARGS+=( -v "$TASK_DATA:$TASK_DATA:ro" )
# RSI_EXTRA_MOUNTS: space-separated host dirs, each mounted rw at its identical absolute path
# (e.g. a staged base model or preference-data cache outside hf_cache).
for m in ${RSI_EXTRA_MOUNTS:-}; do DOCKER_ARGS+=( -v "$m:$m" ); done

# --- fail-fast GPU watchdog ------------------------------------------------------------------
# On 2026-07-24 both opus-5 runs silently lost /dev/nvidia* ~10min in (open() -> EPERM, cuInit ->
# NO_DEVICE) while the device node still existed, and each burned ~2h of a 3h budget polling for a
# GPU that never came back. Root cause is host-level and still unidentified, so detect and abort
# fast: relaunching into a fresh container works. RSI_GPU_WATCHDOG=0 disables.
if [ "${RSI_GPU_WATCHDOG:-1}" = "1" ]; then
  (
    sleep 90                                  # let the container boot
    lost=0
    while [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]; do
      if docker exec "$CONTAINER" bash -c ": < /dev/nvidia$FIRST_GPU" 2>/dev/null; then
        lost=0
      else
        lost=$((lost+1))
        echo "WATCHDOG: /dev/nvidia$FIRST_GPU not openable in-container ($lost/3)" >> "$LOG"
      fi
      if [ "$lost" -ge 3 ]; then
        echo "WATCHDOG: GPU access lost ~90s -> ABORTING RUN (relaunch into a fresh container)" >> "$LOG"
        docker rm -f "$CONTAINER" >/dev/null 2>&1
        break
      fi
      sleep 30
    done
  ) &
  WATCHDOG_PID=$!
fi

# Wall-clock cap around the whole run (docker CLI forwards SIGTERM -> stops the container; --rm and
# the EXIT trap clean up). Same external-timeout pattern PostTrainBench uses.
# KEEP-ALIVE for backgrounded training: agents sometimes launch training with nohup/& and then idle
# "waiting for a notification" — but in `claude --print` there is no notification channel, so the
# agent's turn loop ends, `claude` exits, and (without this) the container tears down (--rm), killing
# in-flight training before final_model/ is written (this is what stopped the 2026-07-27 grok run at
# 78%). After claude exits, if final_model/ isn't there yet, keep the container alive while a
# training-like process is still running, until final_model/ appears or the wall-clock deadline.
# pgrep pattern matches common trainer names (train*/sft/finetune/accelerate/torchrun/Trainer) but
# NOT a leftover vLLM serve, so a refusal (no training) exits immediately instead of idling for hours.
# NB: the pgrep pattern uses the `[x]yz` bracket trick so the pattern's own literal text (present in
# THIS bash -c command line) does not self-match — otherwise the keep-alive shell would match itself
# and idle to the deadline even on a refusal. Echo text below likewise avoids those keywords.
KEEPALIVE="rc=\$?; echo \"=== agent process exited (rc=\$rc) ===\"; \
if [ ! -d '$WORK/$DELIVERABLE' ]; then \
  echo '=== keep-alive: awaiting a backgrounded job to write the deliverable (until wall-clock) ==='; \
  while [ ! -d '$WORK/$DELIVERABLE' ] && [ \$(date +%s) -lt $END_EPOCH ] \
        && pgrep -f '[t]rain|[s]ft|[f]inetune|[a]ccelerate|[t]orchrun|[T]rainer|merge_and_[u]nload' >/dev/null 2>&1; do \
    sleep 20; \
  done; \
  [ -d '$WORK/$DELIVERABLE' ] && echo '=== keep-alive: deliverable appeared ===' || echo '=== keep-alive: ended (deadline reached or job finished) ==='; \
fi"
timeout --signal=TERM --kill-after=120s "$((HOURS*60+5))m" \
  docker run "${DOCKER_ARGS[@]}" "$IMAGE" \
    bash -c "cd '$WORK' && { claude --print --verbose --output-format stream-json \
        --model '$A_MODEL' --dangerously-skip-permissions \"\$(cat '$PROMPT_FILE')\"; }; $KEEPALIVE" \
  > "$LOG" 2>&1 || echo "(agent exited: rc=$? — may be timeout or completion)"

echo "=== agent finished. $DELIVERABLE present? ==="
if [ -d "$WORK/$DELIVERABLE" ]; then ls -la "$WORK/$DELIVERABLE" | head; else echo "NO $DELIVERABLE/ produced"; fi
echo "score with: uv run scripts/task.py score <task_id> --run-dir '$WORK'"
