#!/usr/bin/env bash
# Serve the successor model on a vLLM OpenAI-compatible endpoint for ModelEval oracles.
#
#   scripts/score/serve_successor.sh [MODEL] [PORT] [GPU]
#
# Defaults: Qwen/Qwen3.5-4B on port 8001, GPU 0. Weights read from the nvme HF cache.
# The served-model-name is always "successor" (matches VLLMClient's default RSI_SUCCESSOR_MODEL),
# so the oracle/runner need no extra flags. Run in the serving venv (.venv-serve).
set -euo pipefail

MODEL="${1:-Qwen/Qwen3.5-4B}"
PORT="${2:-8001}"
GPU="${3:-0}"                                   # single index "0", or a comma list "0,1" to shard
TP="$(echo "$GPU" | awk -F',' '{print NF}')"    # tensor-parallel size = number of GPUs listed.
# TP>1 shards weights across GPUs (~1/TP each) so a contended box with little free per GPU can still
# serve: util*total must clear (weights/TP + cache) on EACH gpu. Qwen3.5's hybrid GDN+Mamba
# heads/groups must divide by TP (TP=2 is safe; TP=4 may hit a divisibility limit).
# Fraction of TOTAL device memory vLLM may use (incl. others' usage). On a shared/contended box
# this must clear (used_by_others + ~20GiB) / 80GiB — raise it if a neighbor holds a lot. vLLM
# pre-allocates its pool at startup, so once startup clears, the run is safe from neighbors.
UTIL="${RSI_GPU_UTIL:-0.70}"
MAXLEN="${RSI_MAX_MODEL_LEN:-16384}"   # first_message prompts are <=4k tok; lower to shrink KV
# --enforce-eager skips torch.compile + CUDA-graph capture: much faster/lighter startup, fine for
# an eval workload (throughput not critical). Set RSI_ENFORCE_EAGER=0 to enable graphs.
EAGER_FLAG=""; [ "${RSI_ENFORCE_EAGER:-1}" = "1" ] && EAGER_FLAG="--enforce-eager"

# Weights AND the serving venv live on fast local nvme — importing vLLM/torch off the slow
# NFS-backed EFS stalls for many minutes (rpc_wait on every .so read). Keep both on nvme.
export HF_HOME="${HF_HOME:-${RSI_HOME:-$HOME/rsi}/hf_cache}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # weights already cached; don't block on HF network
SERVE_PY="${RSI_SERVE_PYTHON:-${RSI_HOME:-$HOME/rsi}/venvs/serve/bin/python}"

# Qwen3.5 is a hybrid (Gated-DeltaNet linear-attn + Mamba) model: flashinfer JIT-compiles its GDN
# kernel at startup, which needs `ninja` (in the venv bin) and `nvcc` on PATH + CUDA_HOME set.
# Use the CUDA 12.9 toolkit (matches the cu129 torch/vLLM build and the 12.9 driver).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="$(dirname "${SERVE_PY}"):${CUDA_HOME}/bin:${PATH}"
# Keep torch.compile / flashinfer JIT caches on fast local nvme (EFS is slow for many small writes).
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RSI_HOME:-$HOME/rsi}/.cache}"

echo "serving ${MODEL} on :${PORT} (GPU ${GPU}, TP=${TP}, HF_HOME=${HF_HOME}, py=${SERVE_PY}, CUDA_HOME=${CUDA_HOME})"
# --max-model-len caps context (long agentic transcripts otherwise blow up KV cache); text-only
# use of this multimodal model, so cap media to 0. No --trust-remote-code: the served directory is
# agent-written, and vLLM serves qwen3_5 natively (SECURITY.md).
# RSI_SERVE_EXTRA passes raw vLLM flags through. Needed because an agent can deliver a TEXT-ONLY
# checkpoint of this multimodal base: grok-4.5's chess artifact saved
# architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text", and vLLM 0.25.1's registry has no
# entry for that name -- it falls back to Qwen3_5ForConditionalGeneration, which then dies on
# `config.vision_config` (AttributeError). `RSI_SERVE_EXTRA="--model-impl transformers"` serves the
# delivered artifact as-is via the HF implementation, which is what we want: score what the agent
# actually shipped, not a re-merge of its adapters onto the full base.
# shellcheck disable=SC2086
exec "${SERVE_PY}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name successor \
    --port "${PORT}" \
    --max-model-len "${MAXLEN}" \
    --gpu-memory-utilization "${UTIL}" \
    --tensor-parallel-size "${TP}" \
    ${EAGER_FLAG} \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    ${RSI_SERVE_EXTRA:-}
