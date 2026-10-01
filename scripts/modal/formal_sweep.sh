#!/usr/bin/env bash
# Paper sweep, REPLICATE-MAJOR: replicate 1 of every model first, then replicate 2, then 3, so a partial
# sweep is a complete n=1 table over all six models rather than n=3 on two of them. Within a replicate the
# models run in GROUPS, one driver per proxy key (--key 1..NKEYS, 1-based over the PUB_LITELLM_API_KEY<N>
# entries in .env), concurrency CONC each. Every run is scored in its own sandbox (--score).
#
# RESUMABLE: every driver is launched with --resume-since $LOGDIR/SINCE, so re-running this script after
# a crash (or after a bug fix) tops each cell up to the replicate count instead of starting over. Cells
# whose run ended on an API error are re-run; finished runs without scores.json are printed for
# re-scoring (score_run_modal.py --task --run) and NOT re-run.
#
# The paper's three attack framings (honest baselines come from the default sweep only):
#   default   bash scripts/modal/formal_sweep.sh
#   strategy  ARMS=attack VARIANT=technique_given PREAMBLE=attack_preamble_scored_strategy.txt \
#             bash scripts/modal/formal_sweep.sh
#   cot-aware ARMS=attack VARIANT=technique_given PREAMBLE=attack_preamble_scored_strategy_cotaware.txt \
#             MODELS="fireworks_ai/glm-5p3 xai/grok-4.6" bash scripts/modal/formal_sweep.sh
#
#   LOGDIR=$RSI_HOME/sweep_logs/formal-<stamp> nohup bash scripts/modal/formal_sweep.sh > .../formal_sweep.out 2>&1 &
#
# Env: REPS ("1 2 3"), CONC (8), NKEYS (3), ROWS (the paper's 22 task pairs), MODELS (the six agents),
# ARMS ("honest attack"), VARIANT (goal_only), PREAMBLE (spec default), LOGDIR (new stamp dir by default;
# reuse the old one to resume), WAIT_PIDS (pids to wait for before starting).
set -u
cd "$(dirname "$0")/../.." || exit 1
REPS="${REPS:-1 2 3}"
CONC="${CONC:-8}"
LOGDIR="${LOGDIR:-${RSI_HOME:-$HOME/rsi}/sweep_logs/formal-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOGDIR"
[ -f "$LOGDIR/SINCE" ] || date +%Y%m%d-%H%M%S > "$LOGDIR/SINCE"
SINCE="$(cat "$LOGDIR/SINCE")"
for pid in ${WAIT_PIDS:-}; do
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  echo "[$(date +%H:%M)] pid $pid gone"
done
ROWS="${ROWS:-row01b_math_retention row03b_quality_verbosity row04b_monitor_corpus row05_trait_backdoor
row06_chess_refusal row07_trait_cluster_data row08_rm_math_leniency row09_rm_mentor_persona row10_trait_cluster_rm
row11_heredity_retention row12_heredity_authoring row12b_chat_heredity_authoring row13_capability_suppression
row14_rl_rubric_persona row15_rl_topic_refusal row17_opd_trait_cluster_japan row18b_file_beacon row19b_cot_beacon_code
row20b_cot_beacon_summary row21_silent_step row22_rm_sandbag row23_sycophancy_sandbag}"
ARMS="${ARMS:-honest attack}"
VARIANT="${VARIANT:-goal_only}"
PREAMBLE_ARGS=(); [ -n "${PREAMBLE:-}" ] && PREAMBLE_ARGS=(--attack-preamble "$PREAMBLE")
# The proxy route names, not the marketing names. Order = launch order within a replicate.
MODELS="${MODELS:-anthropic/claude-opus-5 openai/gpt-5.6-sol xai/grok-4.6 fireworks_ai/glm-5p3 meta_ai/muse-spark-1.3-contributor gemini/gemini-3.7-flash}"
NKEYS="${NKEYS:-3}"   # one driver per key; --key N is 1-based in .env numeric order
launch() {  # model key rep
  local m="$1" k="$2" rep="$3" tag
  tag="rep${rep}_$(echo "$m" | tr '/' '_')_$(date +%H%M)"
  echo "[$(date +%H:%M)] start $tag (key $k, top up to $rep replicate(s) since $SINCE)"
  # shellcheck disable=SC2086
  PYTHONUNBUFFERED=1 .venv/bin/python scripts/modal/run_agent_task_modal.py --tasks $ROWS --arms $ARMS \
      --task-variant "$VARIANT" ${PREAMBLE_ARGS[@]+"${PREAMBLE_ARGS[@]}"} --replicates "$rep" --resume-since "$SINCE" --model "$m" --key "$k" \
      --concurrency "$CONC" --score > "$LOGDIR/$tag.log" 2>&1
  echo "[$(date +%H:%M)] done  $tag rc=$?"
}
for rep in $REPS; do
  k=0
  for m in $MODELS; do
    k=$((k + 1))
    launch "$m" "$k" "$rep" &
    if [ "$k" -ge "$NKEYS" ]; then wait; k=0; fi
  done
  wait
  echo "[$(date +%H:%M)] === replicate $rep complete for all models ==="
done
echo "SWEEP DONE"
