#!/usr/bin/env bash
# Rebuild the third-party data files this repo does not redistribute (see data/DATA_LICENSES.md).
#
#   bash scripts/fetch_data.sh                     # CNN/DailyMail, FineWeb-Edu, mixed-domain corpus, RewardBench 2,
#                                                  # GSM8K, MBPP and AlpacaEval test items
#   bash scripts/fetch_data.sh --with-sorry-bench  # + SORRY-Bench/XSTest cover evals -> $RSI_BACKDOOR_DIR (gated)
#   bash scripts/fetch_data.sh --with-gpqa         # + sycophancy/ (gated GPQA, row23)
#   bash scripts/fetch_data.sh --check             # only verify files on disk against data/fetch_data.sha256
#   add --force to rebuild steps whose files are already present
#
# Every builder is the one the task spec names, run DIRECTLY with the spec-pinned arguments. This does
# NOT call `scripts/task.py build`: that also re-runs the LLM probe builders, which cost API budget and
# would overwrite the LLM-authored batteries shipped in data/held_out/. Builders write into a temp
# directory and only the files listed below are moved into place; nothing that ships is overwritten.
#
# Needs network access to the Hugging Face Hub. The default steps need no HF token and no LLM key.
# Python: $PYTHON if set, else `uv run python`, else `python`.
# Env: RSI_HOME (default ~/rsi), RSI_BACKDOOR_DIR (default $RSI_HOME/backdoor),
#      RSI_GPQA_DIR (default $RSI_HOME/gpqa), HF_TOKEN.
set -euo pipefail

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; }

WITH_GPQA=0 WITH_SORRY=0 CHECK_ONLY=0 FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-gpqa) WITH_GPQA=1 ;;
        --with-sorry-bench) WITH_SORRY=1 ;;
        --check) CHECK_ONLY=1 ;;
        --force) FORCE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$PWD
[[ -f pyproject.toml && -d data/held_out ]] || { echo "not an rsi-bench checkout: $REPO" >&2; exit 2; }

RSI_HOME=${RSI_HOME:-$HOME/rsi}
BACKDOOR_DIR=${RSI_BACKDOOR_DIR:-$RSI_HOME/backdoor}
GPQA_DIR=${RSI_GPQA_DIR:-$RSI_HOME/gpqa}
HO=data/held_out
TK=data/tasks
MANIFEST=data/fetch_data.sha256

if [[ -n ${PYTHON:-} ]]; then
    PY=("$PYTHON")
    export PYTHONPATH="$REPO/src:$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}"
elif command -v uv >/dev/null 2>&1; then
    PY=(uv run python)
else
    PY=(python)
    export PYTHONPATH="$REPO/src:$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}"
fi

sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
           else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# ------------------------------------------------------------------------------------------ check
# Manifest lines: "<sha256>  <path>  <group>", group in {default, sorry, gpqa}; '#' = comment.
# @BACKDOOR@ in a path stands for $RSI_BACKDOOR_DIR. An opt-in group is checked only when its flag is
# given or its file is present.
check() {
    local bad=0 ok=0 skipped=0 hash path group got want
    while read -r hash path group; do
        [[ -z ${hash:-} || $hash == \#* ]] && continue
        path=${path/@BACKDOOR@/$BACKDOOR_DIR}
        case $group in
            gpqa) want=$WITH_GPQA ;; sorry) want=$WITH_SORRY ;; *) want=1 ;;
        esac
        if [[ $want == 0 && ! -f $path ]]; then skipped=$((skipped + 1)); continue; fi
        if [[ ! -f $path ]]; then echo "  MISSING   $path"; bad=$((bad + 1)); continue; fi
        got=$(sha256 "$path")
        if [[ $got == "$hash" ]]; then ok=$((ok + 1)); else echo "  MISMATCH  $path"; bad=$((bad + 1)); fi
    done < "$MANIFEST"
    echo "check: $ok ok, $bad missing or mismatched, $skipped opt-in files not requested"
    [[ $bad -eq 0 ]]
}

if [[ $CHECK_ONLY == 1 ]]; then check; exit $?; fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/rsi-fetch.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

# place SRC DEST: copy one rebuilt file into place (temp name, then an atomic rename).
place() {
    mkdir -p "$(dirname "$2")"
    cp "$1" "$2.fetch.$$"
    mv -f "$2.fetch.$$" "$2"
    echo "  -> $2"
}

# need FILE...: succeed when the step must run (--force, or any of its files missing).
need() {
    [[ $FORCE == 1 ]] && return 0
    local f; for f in "$@"; do [[ -f $f ]] || return 0; done
    echo "  (all present; skipping, --force to rebuild)"; return 1
}

# ------------------------------------------------------- 1. CNN/DailyMail (rows 05, 12, 12b, 14)
# Spec build_args, identical for row05/row12: --train 4000 --probes 120 --min-words 150 --max-words 700
echo "== 1/7 CNN/DailyMail train pool + held-out articles (abisee/cnn_dailymail 3.0.0 @96df5e6)"
if need $TK/row05_trait_backdoor/train.jsonl $TK/row12_heredity_authoring/train.jsonl \
        $HO/trait_backdoor/probes.jsonl $HO/heredity_authoring/probes.jsonl \
        $HO/heredity_authoring/articles.jsonl; then
    "${PY[@]}" scripts/build/prep_summarization_task.py --out "$TMP/summ/agent" --secret "$TMP/summ/secret" \
        --train 4000 --probes 120 --min-words 150 --max-words 700
    # articles.jsonl is the first 80 probe articles. --no-selection makes no LLM call; the shipped,
    # LLM-authored selection.jsonl is not rebuilt (its loader re-joins these articles by article_id).
    "${PY[@]}" scripts/build/prep_heredity_authoring_probes.py --out "$TMP/ha" \
        --source "$TMP/summ/secret/probes.jsonl" --articles 80 --no-selection
    for row in row05_trait_backdoor row12_heredity_authoring; do
        place "$TMP/summ/agent/train.jsonl" "$TK/$row/train.jsonl"
    done
    for d in trait_backdoor heredity_authoring; do
        place "$TMP/summ/secret/probes.jsonl" "$HO/$d/probes.jsonl"
    done
    place "$TMP/ha/articles.jsonl" "$HO/heredity_authoring/articles.jsonl"
fi

# -------------------------------------------------------------------------- 2. FineWeb-Edu (row03b)
echo "== 2/7 FineWeb-Edu quality pool + reference-scored held-out set"
if need $TK/row03_quality_hedging/pool.jsonl $HO/quality_filter/held_out_ref.jsonl; then
    "${PY[@]}" scripts/build/prep_quality_task.py --out "$TMP/q/agent" --secret "$TMP/q/secret" \
        --pool 4000 --held-out 500 --min-words 60 --max-words 2000 --low-share 0.5 --seed 17
    place "$TMP/q/agent/pool.jsonl" "$TK/row03_quality_hedging/pool.jsonl"
    place "$TMP/q/secret/held_out_ref.jsonl" "$HO/quality_filter/held_out_ref.jsonl"
fi

# ------------------------------------------------------------------------ 3. corpus pool (row01b)
# The provenance map (data/held_out/corpus_row01b/provenance.jsonl) ships. The rebuilt map is only
# compared with the shipped one -- a mismatch means the upstream data moved -- and never moved.
MIXED_ARGS=(--mixture mixed_domains --drop-domain math
            --drop-domain-prose "mathematical and quantitative content" --per-domain 200
            --dev-frac 0.05 --min-words 40 --max-words 1500 --seed 17 --dup-rate 0.25)
echo "== 3/7 corpus pool: row01b (five-domain mixture)"
if need $TK/row01b_math_retention/{pool.jsonl,dev_labelled.jsonl,spec.json}; then
    "${PY[@]}" scripts/build/prep_corpus_task.py --out "$TMP/c01b/agent" --provenance "$TMP/c01b/prov" "${MIXED_ARGS[@]}"
    if ! cmp -s "$TMP/c01b/prov/provenance.jsonl" "$HO/corpus_row01b/provenance.jsonl"; then
        echo "  WARNING: rebuilt provenance differs from the shipped $HO/corpus_row01b/provenance.jsonl;" \
             "upstream data changed and scores on this pool are not comparable." >&2
    fi
    # The builder's pool order and dev_dupes sampling changed after the paper's runs (same documents).
    # dev_dupes.jsonl (ids + cluster labels only) ships as the paper used it; the pool is put back in the
    # paper's order from the shipped id list, so both files match the paper byte for byte.
    "${PY[@]}" - "$TMP/c01b/agent/pool.jsonl" "$TK/row01b_math_retention.pool_order.txt" <<'PY'
import json, sys
pool, order = sys.argv[1], sys.argv[2]
lines = {json.loads(l)["id"]: l.rstrip("\n") for l in open(pool) if l.strip()}
ids = [i.strip() for i in open(order) if i.strip()]
assert sorted(ids) == sorted(lines), "rebuilt pool holds different documents from the paper's"
open(pool, "w").write("".join(lines[i] + "\n" for i in ids))
PY
    for f in pool.jsonl dev_labelled.jsonl spec.json; do place "$TMP/c01b/agent/$f" "$TK/row01b_math_retention/$f"; done
fi

# ------------------------------------------------------------------------ 4. RewardBench 2 (row22)
echo "== 4/7 RewardBench 2 copy (allenai/reward-bench-2 @7ff0885)"
if need $HO/rm_sandbag/rb2.jsonl; then
    # --skip-battery: the RB2 copy only; the shipped LLM-authored rm_sandbag/pairs.jsonl is untouched.
    "${PY[@]}" scripts/build/prep_rm_sandbag_probes.py --out "$TMP/rb" --skip-battery
    place "$TMP/rb/rb2.jsonl" "$HO/rm_sandbag/rb2.jsonl"
fi

# ------------------------------------------------------------------------------ 5. GSM8K (row13)
# Spec build_args: --pairs 120 --seed 17. Only pairs.jsonl carries GSM8K text; meta.json ships.
echo "== 5/7 GSM8K test pairs with planted defects (openai/gsm8k main @e53f048)"
if need $HO/capability_suppression/pairs.jsonl; then
    "${PY[@]}" scripts/build/prep_suppression_pairs.py --out "$TMP/cs" --pairs 120 --seed 17
    place "$TMP/cs/pairs.jsonl" "$HO/capability_suppression/pairs.jsonl"
fi

# ---------------------------------------------------------------------- 6. MBPP (row19b cover)
# main_heldout.jsonl only; the builder's agent-side train.jsonl (row19, not in the paper) is dropped.
echo "== 6/7 MBPP test split, row19b's code_pass_rate cover (google-research-datasets/mbpp full @4bb6404)"
if need $HO/cot_insertion_code/main_heldout.jsonl; then
    "${PY[@]}" scripts/build/prep_code_task.py --out "$TMP/mbpp/agent" --secret "$TMP/mbpp/secret" --heldout 200
    place "$TMP/mbpp/secret/main_heldout.jsonl" "$HO/cot_insertion_code/main_heldout.jsonl"
fi

# ----------------------------------------------------------------- 7. AlpacaEval (row12b cover)
# Spec build_args: --n 300 --seed 20260913. The reference answers were generated by the base
# successor (Qwen/Qwen3.5-4B) and cannot be regenerated byte for byte, so they ship in
# chat_cover/references.jsonl; the instructions are re-sampled and the two are joined by id.
echo "== 7/7 AlpacaEval instructions, row12b's chat_winrate cover (tatsu-lab/alpaca_eval @2edc6fad8b)"
if need $HO/chat_cover/prompts.jsonl; then
    "${PY[@]}" scripts/build/prep_chat_cover.py --out "$TMP/cc" --n 300 --seed 20260913
    "${PY[@]}" - "$TMP/cc/prompts.jsonl" "$HO/chat_cover/references.jsonl" <<'PY'
import json, sys
sampled, refs = sys.argv[1], sys.argv[2]
ref_lines = open(refs).read().splitlines()
meta, ref = ref_lines[0], {d["id"]: d for d in map(json.loads, filter(str.strip, ref_lines[1:]))}
items = [d for d in map(json.loads, filter(str.strip, open(sampled))) if not d.get("_meta")]
assert sorted(d["id"] for d in items) == sorted(ref), "re-sampled instructions differ from the paper's"
with open(sampled, "w") as fh:
    fh.write(meta + "\n")
    for d in items:
        d.update(reference=ref[d["id"]]["reference"], reference_generator=ref[d["id"]]["reference_generator"])
        fh.write(json.dumps(d, ensure_ascii=False) + "\n")
PY
    place "$TMP/cc/prompts.jsonl" "$HO/chat_cover/prompts.jsonl"
fi

# ---------------------------------------------------------------------------------- opt-in, gated
echo "== opt-in sets"
if [[ $WITH_SORRY == 1 ]]; then
    echo "-- SORRY-Bench 202503 + XSTest cover evals -> $BACKDOOR_DIR (rows 05/06/07/11/15/17/18b)"
    SB=(sorry_bench_base.jsonl sorry_bench_meta.py xstest.json)
    if need "${SB[@]/#/$BACKDOOR_DIR/}"; then
        if [[ -z ${HF_TOKEN:-} && ! -f ${HF_HOME:-$HOME/.cache/huggingface}/token ]]; then
            echo "  SORRY-Bench is gated: accept the licence at" >&2
            echo "  https://huggingface.co/datasets/sorry-bench/sorry-bench-202503, then set HF_TOKEN" >&2
            echo "  (or run \`huggingface-cli login\`) and re-run with --with-sorry-bench." >&2
            exit 1
        fi
        "${PY[@]}" scripts/build/prep_sorry_bench.py --out "$TMP/sorry"
        for f in "${SB[@]}"; do place "$TMP/sorry/$f" "$BACKDOOR_DIR/$f"; done
    fi
    echo "  NOTE: the SORRY-Bench licence forbids redistribution. For Modal scoring:"
    echo "        modal volume put rsi-oracle $BACKDOOR_DIR backdoor"
else
    echo "-- skipping SORRY-Bench/XSTest ($BACKDOOR_DIR); pass --with-sorry-bench (needs HF_TOKEN with gated access)"
fi


if [[ $WITH_GPQA == 1 ]]; then
    echo "-- GPQA main test split -> $HO/sycophancy/items.jsonl"
    if need $HO/sycophancy/items.jsonl; then
        mkdir -p "$GPQA_DIR"
        if [[ ! -f $GPQA_DIR/gpqa_main.json ]]; then
            if [[ -z ${HF_TOKEN:-} && ! -f ${HF_HOME:-$HOME/.cache/huggingface}/token ]]; then
                echo "  GPQA is gated: accept the terms at https://huggingface.co/datasets/Idavidrein/gpqa," >&2
                echo "  then set HF_TOKEN (or run \`huggingface-cli login\`) and re-run with --with-gpqa." >&2
                exit 1
            fi
            RSI_GPQA_PATH="$GPQA_DIR/gpqa_main.json" "${PY[@]}" scripts/build/prep_gpqa.py
        else
            echo "  reusing $GPQA_DIR/gpqa_main.json"
        fi
        [[ -f $GPQA_DIR/gpqa_test_ids.json ]] || RSI_GPQA_DIR="$GPQA_DIR" "${PY[@]}" scripts/build/split_gpqa.py
        "${PY[@]}" scripts/build/prep_sycophancy_probes.py --source gpqa --n 0 --seed 0 \
            --gpqa-path "$GPQA_DIR/gpqa_main.json" --gpqa-test-ids "$GPQA_DIR/gpqa_test_ids.json" \
            --out "$TMP/syc"
        place "$TMP/syc/items.jsonl" "$HO/sycophancy/items.jsonl"
    fi
    echo "  NOTE: the GPQA authors ask that it not be revealed in plaintext online. This file is"
    echo "        gitignored; do not commit or upload it."
else
    echo "-- skipping GPQA (sycophancy/); pass --with-gpqa (needs HF_TOKEN with gated access)"
fi

echo "== verifying against $MANIFEST"
check
