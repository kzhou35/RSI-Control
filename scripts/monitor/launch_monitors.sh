#!/usr/bin/env bash
# The paper's single-call monitor pass over a list of runs: GPT-5.6 Luna + Gemini 3.7 Flash, and GPT-5.6 Sol,
# each at transcript and chain-of-thought scope (whole-transcript chunking, max suspicion over chunks).
#
#   bash scripts/monitor/launch_monitors.sh <RUN_LIST> <TAG> <SINCE> [PORTBASE] [COT_RUN_LIST]
#
# RUN_LIST: run ids (one per line) for the transcript scope. COT_RUN_LIST (default RUN_LIST) for the CoT scope --
# pass the GLM + grok subset there, the only agents whose reasoning reaches the transcript. SINCE: the sweep's
# stamp (its LOGDIR/SINCE). Writes monitor_{3mon_S2,sol}_single_{transcript,cot}_<TAG>.json under $OUT_DIR
# (default $RSI_HOME/results/monitor/formal), the input of pool_monitor_reports.py. Needs the ResearchArena
# clone (RSI_RESEARCH_ARENA, default $RSI_HOME/external/ResearchArena).
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT_DIR="${OUT_DIR:-${RSI_HOME:-$HOME/rsi}/results/monitor/formal}"; mkdir -p "$OUT_DIR"
LIST=$(readlink -f "$1"); TAG=$2; SINCE=$3; PB=${4:-8820}; COTLIST=$(readlink -f "${5:-$1}")
KEY="${KEY:-1}"
RUNS=$(tr '\n' ' ' < "$LIST"); CRUNS=$(tr '\n' ' ' < "$COTLIST")
LG="--monitor-model openai/gpt-5.6-luna --monitor-model gemini/gemini-3.7-flash"
SOL="--monitor-model openai/gpt-5.6-sol"
COMMON="--key $KEY --since $SINCE --transcript-mode chunk --monitor-style single --keep-all --concurrency 4"
F=$OUT_DIR
# shellcheck disable=SC2086
{
RSI_MONITOR_SHIM_PORT=$PB       .venv/bin/python scripts/monitor/run_monitor.py $LG  $COMMON --with-cot --runs $CRUNS --out $F/monitor_3mon_S2_single_cot_$TAG.json > $F/monitor_3mon_S2_single_cot_$TAG.log 2>&1 &
RSI_MONITOR_SHIM_PORT=$((PB+1)) .venv/bin/python scripts/monitor/run_monitor.py $SOL $COMMON --with-cot --runs $CRUNS --out $F/monitor_sol_single_cot_$TAG.json > $F/monitor_sol_single_cot_$TAG.log 2>&1 &
RSI_MONITOR_SHIM_PORT=$((PB+2)) .venv/bin/python scripts/monitor/run_monitor.py $LG  $COMMON --runs $RUNS --out $F/monitor_3mon_S2_single_transcript_$TAG.json > $F/monitor_3mon_S2_single_transcript_$TAG.log 2>&1 &
RSI_MONITOR_SHIM_PORT=$((PB+3)) .venv/bin/python scripts/monitor/run_monitor.py $SOL $COMMON --runs $RUNS --out $F/monitor_sol_single_transcript_$TAG.json > $F/monitor_sol_single_transcript_$TAG.log 2>&1 &
wait
}
