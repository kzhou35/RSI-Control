# Reproducing the paper

The paper's experiments run on [Modal](https://modal.com): every agent run gets a dedicated H100
sandbox, and scoring happens in a separate sandbox that is the only one with access to the held-out
evaluation data. A local Docker runner exists too (end of this page); it is the same pipeline on your own GPUs.

```
agent run (Modal H100)  ->  oracle score (separate sandbox)  ->  per-task differential (attack - honest)
                                                             ->  monitors over the transcripts
                                                             ->  successor audit
                                                             ->  tables / figures (analysis/)
```

## 0. Requirements

- Python >= 3.11 and [uv](https://docs.astral.sh/uv/).
- A Modal account with H100 access.
- A [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) that is **reachable from the public
  internet** (the agents' model calls come from inside the Modal sandboxes). The proxy has to serve the
  routes the code names:

  | role | route(s) |
  |---|---|
  | agents | `anthropic/claude-opus-5`, `openai/gpt-5.6-sol`, `xai/grok-4.6`, `fireworks_ai/glm-5p3`, `meta_ai/muse-spark-1.3-contributor`, `gemini/gemini-3.7-flash` |
  | oracle judge, probe authoring | `anthropic/claude-opus-4-8` |
  | reward judge (RL rows) | `openai/gpt-5.6-luna` |
  | monitors | `openai/gpt-5.6-luna`, `gemini/gemini-3.7-flash`, `openai/gpt-5.6-sol` |
  | successor auditor | `openai/gpt-5.6-sol` |

- A Hugging Face token. Successor weights are public (Qwen3.5 2B/4B). Two datasets are gated: accept
  the licenses for [sorry-bench](https://huggingface.co/sorry-bench) (the safety cover metric of rows 5/6/15)
  and [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) (row 23).

## 1. Install

```bash
uv sync --extra dev --extra data --extra modal --extra analysis
cp .env.example .env            # fill in the proxy URL(s), key(s) and HF_TOKEN
export RSI_HOME=~/rsi           # machine-local root: external clones, gated data, results (default ~/rsi)
uv run pytest -q                # CPU-only unit tests, no API calls
```

The monitors use the monitor scaffold of [aisa-group/ResearchArena](https://github.com/aisa-group/ResearchArena)
(pinned at `58776e0`), cloned under `$RSI_HOME/external/`:

```bash
mkdir -p $RSI_HOME/external && cd $RSI_HOME/external
git clone https://github.com/aisa-group/ResearchArena && git -C ResearchArena checkout 58776e0
```

## 2. Data

Probe sets that we wrote, or that were authored by a model for this project, ship in `data/`. Third-party
text is rebuilt from its pinned upstream revision (see [data/DATA_LICENSES.md](../data/DATA_LICENSES.md)):

```bash
bash scripts/fetch_data.sh --with-sorry-bench --with-gpqa     # both steps need gated HF access
```

`uv run scripts/task.py list` shows every task and whether its environment is built: before the fetch,
rows 1b, 3b, 5, 12, 12b, 13, 14, 19b, 22 and 23 read MISSING (afterwards all 22 read BUILT). Do **not** use `task.py build` to restore the shipped data:
for tasks with an LLM-authored probe set it re-runs the authoring model and overwrites the shipped set
with a different one.

## 3. Modal setup (one time)

```bash
uv run modal setup                                     # authenticate
uv run python scripts/modal/preflight.py               # can a Modal container reach your proxy? (CPU, ~1 min)
uv run python scripts/modal/preflight.py --gpu         # build the agent image, check the GPU toolchain (~20 min)
uv run python scripts/modal/preflight.py --warm-cache --skip-network   # successor weights -> volume rsi-hf-cache

# The held-out data goes to the scorer-only volume. Re-run after any change under data/held_out:
# the scorer reads the volume, not your checkout.
uv run modal volume put rsi-oracle data/held_out held_out --force
uv run modal volume put rsi-oracle $RSI_HOME/backdoor backdoor --force
```

Volumes: `rsi-hf-cache` (weights), `rsi-taskdata` (agent-visible task data, uploaded per run),
`rsi-runs` (one directory per run), and `rsi-oracle` (held-out data, mounted only by scorer sandboxes).

## 4. Agent runs

One cell (both arms, one replicate):

```bash
uv run python scripts/modal/run_agent_task_modal.py --tasks row06_chess_refusal --arms honest attack \
    --task-variant goal_only --replicates 1 --model xai/grok-4.6 --key 1 --concurrency 2 --score
```

The paper's sweep: 22 task pairs x 6 agents x {honest, attack} x 3 replicates, each run scored in its
own sandbox. The driver is replicate-major and resumable (re-running it tops cells up to the replicate
count):

```bash
bash scripts/modal/formal_sweep.sh                                          # default framing, honest + attack
ARMS=attack VARIANT=technique_given PREAMBLE=attack_preamble_scored_strategy.txt \
    bash scripts/modal/formal_sweep.sh                                      # strategy framing
ARMS=attack VARIANT=technique_given PREAMBLE=attack_preamble_scored_strategy_cotaware.txt \
    MODELS="fireworks_ai/glm-5p3 xai/grok-4.6" bash scripts/modal/formal_sweep.sh   # reasoning-aware framing
```

Every run writes `run_meta.json`, the agent's `agent_stream.log`, the deliverable and `scores.json` to
`rsi-runs/<run_id>/`. Runs that finished without a score can be scored afterwards:

```bash
uv run python scripts/modal/score_run_modal.py --auto
```

Base-model reference for the post-training rows' main-task floors:

```bash
uv run python scripts/modal/score_base_modal.py --all-post-training
```

## 5. Sabotage results

```bash
uv run python scripts/modal/report_differentials.py --since <SINCE> --all-runs \
    --json analysis/data/sabotage_report_$(date +%Y%m%d).json
```

`<SINCE>` is the sweep's `$LOGDIR/SINCE` stamp. A cell is the mean of its valid replicates. A run
counts as a successful sabotage when both criteria in `src/rsi_bench/verdict.py` hold and the
main-task cover holds. Details are in the paper's appendix.

## 6. Monitors

The monitors are blind to the side task and read the transcript, or the transcript plus the agent's
reasoning (GLM and grok are the agents whose reasoning is exposed):

```bash
bash scripts/monitor/launch_monitors.sh <run_ids.txt> <tag> <SINCE> 8820 <glm_grok_run_ids.txt>
uv run python scripts/monitor/pool_monitor_reports.py \
    --differentials analysis/data/sabotage_report_<date>.json > analysis/data/monitor_readout.txt
```

The paper excludes runs whose agent trained on the graded evaluation's items. A fixed judge that is not
one of the monitors produces that label:

```bash
uv run python scripts/monitor/label_heldout_use.py --runs-file <run_ids.txt> --out analysis/data/heldout_labels.jsonl
```

Attack-method labels (the lever figure):

```bash
uv run python scripts/monitor/label_levers.py --streams <dir of exported agent_stream.log> \
    --out analysis/data/lever_labels_<date>.jsonl --model openai/gpt-5.6-sol
```

## 7. Successor audit

```bash
uv run python scripts/audit/run_successor_audit.py --select all --concurrency 6 --tag v1   # -> $RSI_HOME/results/audit/v1/
```

## 8. Tables and figures

See [analysis/README.md](../analysis/README.md).

## Local runner (no Modal)

`scripts/task.py run` launches the same agent inside a Docker container on a local GPU
(`scripts/run/run_agent_task_docker.sh`), with the environment derived from the task spec. It needs an
NVIDIA driver supporting CUDA 12.9, Docker with the NVIDIA container runtime, and, under `$RSI_HOME`,
`venvs/serve` (vLLM), `venvs/train` (LoRA SFT stack), `tools/` (Node 22 + the Claude Code CLI) and
`hf_cache/`. The venv specs match the Modal image (`AGENT_IMAGE` in `scripts/modal/rsi_modal.py`):

```bash
uv venv --python 3.12 $RSI_HOME/venvs/serve
uv pip install --python $RSI_HOME/venvs/serve/bin/python --torch-backend=cu129 \
  "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
uv venv --python 3.12 $RSI_HOME/venvs/train
uv pip install --python $RSI_HOME/venvs/train/bin/python --torch-backend=cu129 \
  "torch==2.11.0" torchvision==0.26.0 torchaudio==2.11.0 "transformers>=5.14,<6" peft accelerate datasets safetensors
docker build -t rsi-agent:base -f scripts/agent_task/Dockerfile.base scripts/agent_task

uv run scripts/task.py run row06_chess_refusal --arm attack --variant goal_only --gpu 0 --model anthropic/claude-opus-5
uv run scripts/task.py score row06_chess_refusal --run-dir <work_dir>
```

Non-Anthropic agents drive Claude Code through `scripts/run/anthropic_shim.py`, which translates the
Messages API for the proxy. The Modal path starts it automatically.
