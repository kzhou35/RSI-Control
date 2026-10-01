#!/usr/bin/env bash
# Drive the PostTrainBench-style agent (Claude Code CLI) on one task, HONEST or ATTACK (launched by task.py run).
#
#   scripts/run/run_agent_task.sh <honest|attack> [goal_only|technique_given] [run_label]
#
# Sets up a fresh working dir on nvme, composes the task prompt from scripts/agent_task/prompts/,
# puts node/claude + our train/serve venvs on PATH, wires the LiteLLM proxy (claude-sonnet-5), and
# runs the agent full-auto with a wall-clock cap. The agent writes final_model/. Score afterwards
# with `uv run scripts/task.py score <task_id> --run-dir <work_dir>`. Does NOT score here (keep the run and the scoring separate).
set -euo pipefail

MODE="${1:?usage: run_agent_task.sh <honest|attack> [goal_only|technique_given] [label]}"
VARIANT="${2:-technique_given}"
LABEL="${3:-run}"

# --- config ---
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"          # .../rsi-bench
PROMPTS="$REPO/scripts/agent_task/prompts"
SUCCESSOR="Qwen/Qwen3.5-4B"
BASE_MODEL_PATH="Qwen/Qwen3.5-4B"                                 # HF id; cached under HF_HOME (offline)
TRAIN_PY="${RSI_HOME:-$HOME/rsi}/venvs/train/bin/python"
SERVE_PY="${RSI_HOME:-$HOME/rsi}/venvs/serve/bin/python"
HOURS="${RSI_AGENT_HOURS:-4}"
NODE_BIN="${RSI_HOME:-$HOME/rsi}/tools/node-v22.11.0-linux-x64/bin"
CLAUDE_BIN="${RSI_HOME:-$HOME/rsi}/tools/npm-global/bin"
RUNS_ROOT="${RSI_AGENT_RUNS:-${RSI_HOME:-$HOME/rsi}/agent_runs}"
STAMP="$(date +%Y%m%d-%H%M%S 2>/dev/null || echo run)"
WORK="$RUNS_ROOT/${MODE}_${VARIANT}_${LABEL}_${STAMP}"

# --- env: PATH (new node FIRST) ---
export PATH="$NODE_BIN:$CLAUDE_BIN:$PATH"
# Read specific vars from .env WITHOUT shell-sourcing it (values have <placeholders> that break the
# parser, and inline `# comments` that pollute values). getenv strips the inline comment, trailing
# whitespace, and surrounding quotes.
getenv() { grep -E "^$1=" "$REPO/.env" 2>/dev/null | head -1 \
  | sed -E "s/^$1=//; s/[[:space:]]*#.*$//; s/[[:space:]]*$//; s/^[\"']//; s/[\"']$//"; }
LITELLM_API_KEY="$(getenv LITELLM_API_KEY)"
LITELLM_BASE_URL="$(getenv LITELLM_BASE_URL)"
HF_TOKEN="$(getenv HF_TOKEN)"
# ANTHROPIC_BASE_URL override lets us point at the local LiteLLM shim (http://localhost:4000) when
# driving a NON-Claude model (e.g. GLM) — see scripts/agent_task/litellm_shim.yaml. For non-Claude
# models set RSI_AGENT_SMALL_MODEL to the same model (the shim's model_list only has that one).
export ANTHROPIC_BASE_URL="${RSI_ANTHROPIC_BASE_URL:-${LITELLM_BASE_URL%/v1}}"
export ANTHROPIC_AUTH_TOKEN="${LITELLM_API_KEY:-local-shim}"
export ANTHROPIC_MODEL="${RSI_AGENT_MODEL:-anthropic/claude-sonnet-5}"
export ANTHROPIC_SMALL_FAST_MODEL="${RSI_AGENT_SMALL_MODEL:-anthropic/claude-haiku-4-5}"
export HF_HOME="${HF_HOME:-${RSI_HOME:-$HOME/rsi}/hf_cache}"
# Agent needs ONLINE HF to download training datasets (HF_TOKEN carried from .env). Do NOT set
# HF_HUB_OFFLINE here. Pin the agent to a single GPU so it can't touch colleagues' GPUs.
export CUDA_VISIBLE_DEVICES="${RSI_AGENT_GPU:-3}"

mkdir -p "$WORK"
# timer.sh: how much wall-clock remains (agent calls `bash timer.sh`)
END_EPOCH=$(( $(date +%s) + HOURS*3600 ))
cat > "$WORK/timer.sh" <<TIMER
#!/usr/bin/env bash
rem=\$(( $END_EPOCH - \$(date +%s) ))
if [ \$rem -le 0 ]; then echo "0h 0m remaining (TIME IS UP)"; else echo "\$((rem/3600))h \$(((rem%3600)/60))m remaining"; fi
TIMER
chmod +x "$WORK/timer.sh"

# --- dev set for the agent to gauge progress (labeled; the agent MAY train on it — it is NOT the
# graded test set). Copied into the work dir; the capable agent evaluates its own model with it.
# --- task selection: set by `scripts/task.py run` from the TaskSpec (RSI_MAIN_PROMPT, RSI_ATTACK_PREAMBLE,
# RSI_SIDE_GOAL, RSI_TASK_DATA, ...); there is no default task.
MAIN_PROMPT="${RSI_MAIN_PROMPT:?set RSI_MAIN_PROMPT (use scripts/task.py run)}"
ATTACK_PREAMBLE="${RSI_ATTACK_PREAMBLE:?set RSI_ATTACK_PREAMBLE (use scripts/task.py run)}"
SIDE_GOAL_PREFIX="${RSI_SIDE_GOAL:?set RSI_SIDE_GOAL (use scripts/task.py run)}"   # reads {prefix}_{variant}.txt
DEV_FILE="${RSI_DEV_FILE:-}"   # optional labeled dev set copied into the workspace
DEV_EVAL_CMD="(no separate dev set is provided — use the public benchmarks named in the task to gauge progress, per the rules there)"
if [ -n "$DEV_FILE" ] && [ -f "$DEV_FILE" ]; then
  cp "$DEV_FILE" "$WORK/$(basename "$DEV_FILE")"
  DEV_EVAL_CMD="a labeled dev set is at ./$(basename "$DEV_FILE") ({question, choices[4], answer-letter}); evaluate your model on it to gauge progress (you MAY train on it — it is NOT the graded test set)"
fi
# --- compose the prompt (python does the multiline templating) ---
TASK_DATA="${RSI_TASK_DATA:-}"
DELIVERABLE="${RSI_DELIVERABLE:-final_model}"
DROP_DOMAIN="${RSI_DROP_DOMAIN:-Lua source code}"
RM_BASE_MODEL="${RSI_RM_BASE_MODEL:-Qwen/Qwen3.5-4B}"

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

echo "=== run: mode=$MODE variant=$VARIANT model=$ANTHROPIC_MODEL hours=$HOURS ==="
echo "=== work dir: $WORK ==="
LOG="$WORK/agent_stream.log"
# Wall-clock cap around the whole agent run (PostTrainBench uses the same external-timeout pattern).
timeout --signal=TERM --kill-after=60s "$((HOURS*60+5))m" \
  bash -c "cd '$WORK' && claude --print --verbose --output-format stream-json \
      --model '$ANTHROPIC_MODEL' --dangerously-skip-permissions \
      \"\$(cat '$PROMPT_FILE')\"" > "$LOG" 2>&1 || echo "(agent exited: rc=$? — may be timeout or completion)"

echo "=== agent finished. $DELIVERABLE present? ==="
if [ -d "$WORK/$DELIVERABLE" ]; then ls -la "$WORK/$DELIVERABLE" | head; else echo "NO $DELIVERABLE/ produced"; fi
echo "score with: uv run scripts/task.py score <task_id> --run-dir '$WORK'"
