#!/usr/bin/env bash
# Serve a run's delivered model on a vLLM endpoint and run ANY served scorer against it, then tear
# down. Generic over scorers, so the
# successor tasks — trait_backdoor (5), topic_refusal (6), trait_cluster (7) — can be scored the same
# way. cmd_score in task.py calls this for its SERVED scorers.
#
#   scripts/score/serve_and_score.sh <model_dir> <gpu> <port> <out> <scorer.py> [scorer args...]
#
# Injects --base-url / --model successor / --out into the scorer call; everything after <scorer.py>
# is the task's own oracle args (topic, cluster, samples, ...). Port is per-GPU (task.py sets
# 8012+gpu) so two scorers running on different GPUs never collide, and teardown is port-scoped so it
# never kills a neighbour's serve.
set -euo pipefail

MODEL_DIR="${1:?usage: serve_and_score.sh <model_dir> <gpu> <port> <out> <scorer.py> [args...]}"
GPU="${2:-0}"; PORT="${3:-8012}"; OUT="${4:?need out path}"; SCORER="${5:?need scorer}"
shift 5
# A BASE ARM serves an untrained checkpoint by HF repo id ("Qwen/Qwen3.5-4B") rather than a delivered
# directory. vLLM resolves it from HF_HOME, so the only thing in the way was this existence check --
# which must stay for run arms, where a missing directory means the agent delivered nothing and the
# right outcome is to fail before paying for a serve.
if [ ! -d "$MODEL_DIR" ]; then
  case "$MODEL_DIR" in
    */*/*|*" "*|/*) echo "no model dir: $MODEL_DIR (agent delivered nothing to serve)"; exit 1 ;;
    */*)            echo "=== $MODEL_DIR is not a directory; treating it as an HF repo id (base arm) ===" ;;
    *)              echo "no model dir: $MODEL_DIR (agent delivered nothing to serve)"; exit 1 ;;
  esac
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The delivery is agent-written: refuse one that carries code a loader could execute (SECURITY.md).
if [ -d "$MODEL_DIR" ]; then
  "${RSI_GUARD_PYTHON:-python3}" "$REPO/src/rsi_bench/checkpoint_guard.py" "$MODEL_DIR" || exit $?
fi
LOG="$(dirname "$OUT")/serve_score.log"

# A TEXT-ONLY Qwen3.5 save (model_type qwen3_5_text, architectures Qwen3_5ForCausalLM, no
# vision_config -- what an agent gets after pip-installing transformers>=5 and going through
# AutoModelForCausalLM) cannot be served by vLLM 0.25.1 as delivered: it resolves the architecture
# to the multimodal class and dies in __init__ with
#     AttributeError: 'Qwen3_5TextConfig' object has no attribute 'vision_config'
# 14 deliverables of the 2026-09-06 muse/gemini sweeps (5 h of agent time each) scored NO_SCORES
# that way with complete weights; row11 technique_given died the same way on 2026-08-16. Repackage
# into a shadow directory that IS a full Qwen3.5 checkpoint (base config + the delivered text
# tensors renamed + the base's vision tensors, see textonly_to_mm_shadow.py) and serve that. The
# delivered directory is not touched, so the audit trail stays byte-identical.
SERVE_DIR="$MODEL_DIR"
if [ -f "$MODEL_DIR/config.json" ] && ! grep -q '"vision_config"' "$MODEL_DIR/config.json" \
   && grep -qE '"qwen3_5_text"|Qwen3_5[A-Za-z]*ForCausalLM' "$MODEL_DIR/config.json"; then
  BASE_ID="${RSI_BASE_MODEL:-Qwen/Qwen3.5-4B}"
  HF_ROOT="${HF_HOME:-/cache/hf}"; [ -d "$HF_ROOT" ] || HF_ROOT=${RSI_HOME:-$HOME/rsi}/hf_cache
  BASE_DIR="$(ls -d "$HF_ROOT/hub/models--${BASE_ID//\//--}/snapshots/"* 2>/dev/null | head -1)"
  CONV_PY="${RSI_SERVE_PYTHON:-/opt/rsi/venvs/serve/bin/python}"
  [ -x "$CONV_PY" ] || CONV_PY=${RSI_HOME:-$HOME/rsi}/venvs/serve/bin/python
  SHADOW="$(mktemp -d)/final_model"
  echo "=== text-only Qwen3.5 delivery: repackaging as multimodal layout -> $SHADOW (base $BASE_DIR) ==="
  if "$CONV_PY" "$REPO/scripts/score/textonly_to_mm_shadow.py" \
       "$MODEL_DIR" "$BASE_DIR" "$SHADOW" >/dev/null; then
    SERVE_DIR="$SHADOW"
  else
    echo "=== repackaging failed (rc=$?); serving the original ==="
  fi
fi

echo "=== serving $SERVE_DIR on GPU $GPU port $PORT ==="
RSI_GPU_UTIL="${RSI_GPU_UTIL:-0.30}" RSI_MAX_MODEL_LEN="${RSI_MAX_MODEL_LEN:-8192}" RSI_ENFORCE_EAGER="${RSI_ENFORCE_EAGER:-1}" \
  bash "$REPO/scripts/score/serve_successor.sh" "$SERVE_DIR" "$PORT" "$GPU" > "$LOG" 2>&1 &
SERVE_WRAP_PID=$!
cleanup() {
  pkill -9 -P "$SERVE_WRAP_PID" 2>/dev/null || true
  pkill -9 -f "served-model-name successor .*port $PORT" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== waiting for endpoint (JIT of the GDN kernel makes first serve slow) ==="
# Block until the endpoint is up. 2026-09-14: this loop gave up after 12 min and fell through to the
# scorer anyway; on a cold image the GDN JIT took longer than that and 80/80 (row21), 223/240 (row13)
# and 252/276 (row12b) generation calls failed with APIConnectionError -- scored as a broken checkpoint.
READY=0
for i in $(seq 1 800); do   # 40 min
  if grep -qE "Application startup complete|Uvicorn running on" "$LOG" 2>/dev/null; then echo "ready"; READY=1; break; fi
  if ! kill -0 "$SERVE_WRAP_PID" 2>/dev/null && ! pgrep -f "served-model-name successor .*port $PORT" >/dev/null; then
    echo "serve died before ready; tail:"; tail -12 "$LOG"; exit 2
  fi
  sleep 3
done
if [ "$READY" != 1 ]; then echo "serve not ready after 40 min; tail:"; tail -12 "$LOG"; exit 2; fi

echo "=== scoring: $SCORER (served model) ==="
"$REPO/.venv/bin/python" "$REPO/$SCORER" \
  --base-url "http://localhost:$PORT/v1" --model successor "$@" --out "$OUT"
echo "=== done: $OUT ==="
