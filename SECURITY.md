# Security

## Reporting a vulnerability

Please report security issues privately through GitHub's **Report a vulnerability** button (the
repository's Security tab), not in a public issue. We aim to reply within a week.

## Threat model: the agent and everything it produces are untrusted

In the ATTACK arm the agent is told to sabotage the model it trains, and it has full control of its
workspace, including the delivered checkpoint (`final_model/`, `final_rm/`) and any file next to it.
The scorers then load and run that output:

- **Delivered checkpoints are loaded** by vLLM and transformers. Neither is run with
  `trust_remote_code`, and `rsi_bench.checkpoint_guard` refuses, before anything loads it, a delivery
  that contains a code file (`.py`, `.pyc`, `.so`) or an `auto_map` key in any JSON config, since
  either one lets a checkpoint run its own Python. Keep that check if you add a scorer.
- **Code written by the successor model is executed** to grade the code-flavoured tasks
  (`rsi_bench.evals.cot_insertion.run_code_probe`). It runs in a separate interpreter with CPU,
  memory, file-size and process limits, an empty environment and a throwaway working directory,
  but with no network or filesystem isolation.

Run agents and scorers only inside disposable, isolated sandboxes, never on a workstation or
server that holds credentials or data you care about. The Modal path (`scripts/modal/`) does this:
each agent run gets its own sandbox and its own volume, and scoring runs in a separate sandbox.
That scoring sandbox still holds the judge's proxy keys and the held-out data, so use proxy keys
with a spending cap and treat the held-out data as readable by a determined attacker.

## API keys

- Put keys in `.env` (gitignored) or a Modal Secret; never on a command line or in a committed file.
- `scripts/run/anthropic_shim.py` adds a real proxy key to every request it forwards. It listens on
  `127.0.0.1` by default; only pass `--host` on a machine whose network you trust.
- Inside agent sandboxes the shim runs as root and the agent runs as an unprivileged user, so the
  agent never holds a key.

## Dependency pins

`uv.lock` tracks patched releases (vLLM 0.30, transformers 5.18, torch 2.13). setuptools stays below
81 because vLLM requires it; its open advisory affects building sdists with setuptools, which this
project does not do (the build backend is hatchling).

The Modal images (`scripts/modal/rsi_modal.py`) keep the environment the paper's runs used: vLLM
0.25.1, torch 2.11 and transformers 5.14+. That transformers release already has the
checkpoint-loading fixes; the open vLLM advisories concern its network API, which only listens on
localhost inside the sandbox. Do not expose the vLLM server or the shim to an untrusted network. To
run on current releases, bump `VLLM_WHEEL` and the train-venv pins; new runs are then not strictly
comparable with the paper's.

## Dual-use content

The repository contains instructions that tell AI agents to covertly sabotage model training (side
task briefs and attack framings in `scripts/agent_task/prompts/` and `src/rsi_bench/goals/`). They
exist to measure that risk and to test monitors against it, on small open models trained in
sandboxes. Do not use them against systems you do not own.
