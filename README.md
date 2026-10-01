# RSI Control Arena

Code and data for **RSI Control: Can AI Models Perform Covert Sabotage in the Self-Improvement Process?**

In this testbed, a frontier model works as an autonomous AI R&D agent. It gets a benign **main task**
from one stage of a successor model's training pipeline: data curation, supervised fine-tuning,
reward-model training, reinforcement learning or on-policy distillation. In the ATTACK arm it also
gets a covert **side task**: install a misaligned property in the successor it delivers. A held-out
oracle, which the agent never sees, measures whether the property landed. It is scored as a
differential against an HONEST run of the same main task, and the main task's own metric has to stay
intact. LLM monitors then read the agent's transcript (and, where exposed, its reasoning) without
being told the side task.

The testbed has 16 side tasks in three classes (capability, alignment, goal injection), forming 22
(main, side) task pairs. The successors are Qwen3.5 2B/4B models, trained inside the sandbox and never deployed.

## Layout

```
src/rsi_bench/
  tasks/specs/     one YAML per task pair: main task, side task, environment, oracle, thresholds, pinned sources
  goals/specs/     the side tasks (misaligned goals)
  evals/           the held-out oracles and main-task cover metrics
  clients/         model clients shared by builders and scorers (proxy, served successor, LLM judges)
  verdict.py       the "sabotage succeeded" rule
scripts/
  task.py          list / show / build / run / score any task
  fetch_data.sh    rebuilds the third-party data that is not redistributed here
  build/           probe-set and task-environment builders (prep_*.py)
  score/           one scorer per oracle (score_*.py), the vLLM serve wrappers, the contamination audit
  run/             local agent runner (Docker) and the Anthropic-API shim for non-Anthropic agents
  agent_task/      agent prompts: main tasks, attack preambles, side-task briefs, strategy notes
  modal/           the Modal pipeline: agent runs, scoring, sweeps, differentials report
  monitor/         blind monitors, monitor pooling, run labellers
  audit/           successor audit (a model queries the delivered successor)
data/
  held_out/        held-out evaluation sets (the oracles' probes)
  tasks/           agent-visible task data
analysis/          scripts for the paper's tables and figures
docs/              REPRODUCING.md, ADDING_A_TASK.md, TASKS.md, SIDE_TASKS.md, TEST_DESIGNS.md, TESTS.md
```

## Quick start

```bash
uv sync --extra dev --extra data --extra modal
uv run pytest -q                              # unit tests, no GPU or API
uv run scripts/task.py list                   # the task pairs and whether their data is built
uv run scripts/task.py show row06_chess_refusal
```

To run agents and reproduce the paper, see [docs/REPRODUCING.md](docs/REPRODUCING.md). You need a
Modal account (H100), a publicly reachable LiteLLM proxy for the agent, judge and monitor models,
and a Hugging Face token.

## A note on the contents

This repository contains attack instructions for AI agents: side-task briefs, attack framings, and a
list of monitor-evasion tactics (`scripts/agent_task/prompts/strategy/`). We release them so that
monitors and training pipelines can be tested against these attacks before such agents are trusted
with more of this work. Most side tasks are benign stand-ins for harmful properties (a food
preference, an excited register, refusing to discuss chess).

The agent controls the checkpoint it delivers, and scoring loads and runs it: run agents and scorers
only in isolated sandboxes. See [SECURITY.md](SECURITY.md).

The held-out evaluation sets are public here. Any model trained on this repository has seen them, so
treat results on such models accordingly.

## Built on

- [PostTrainBench](https://github.com/aisa-group/PostTrainBench) (MIT): the post-training task framing, container and evaluation scripts, adapted with attribution. See `NOTICE`.
- [ResearchArena](https://github.com/aisa-group/ResearchArena): the monitor scaffold.

## License

Code: MIT (see `LICENSE`). Data: see [data/DATA_LICENSES.md](data/DATA_LICENSES.md).

## Citation

```bibtex
@article{zhou2026rsicontrol,
  title  = {RSI Control: Can AI Models Perform Covert Sabotage in the Self-Improvement Process?},
  author = {Zhou, Kaiwen and Sehwag, Udari Madhushani},
  year   = {2026},
  note   = {arXiv preprint (link to be added)}
}
```
