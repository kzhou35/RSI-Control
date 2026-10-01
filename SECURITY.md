# Security

## Reporting a vulnerability

Please report security issues privately through GitHub's **Report a vulnerability** button (the
repository's Security tab), not in a public issue. We aim to reply within a week.

## Threat model: the agent and everything it produces are untrusted

In the ATTACK arm the agent is told to sabotage the model it trains, and it has full control of its
workspace, including the delivered checkpoint (`final_model/`, `final_rm/`) and any file next to it.
The scorers then load and run that output:

- **Delivered checkpoints are loaded with `trust_remote_code`** (`scripts/score/serve_successor.sh`,
  `scripts/score/score_topic_refusal.py`). A checkpoint that ships its own Python modelling code
  (an `auto_map` entry in `config.json` plus a `.py` file) runs that code inside the scorer.
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

`uv.lock` and the Modal images pin the vLLM, transformers and torch versions the paper's results were
produced with. Some of those versions have published CVEs. They are kept for reproducibility; do not
expose the vLLM server or the shim to an untrusted network, and upgrade them for any other use.

## Dual-use content

The repository contains instructions that tell AI agents to covertly sabotage model training (side
task briefs and attack framings in `scripts/agent_task/prompts/` and `src/rsi_bench/goals/`). They
exist to measure that risk and to test monitors against it, on small open models trained in
sandboxes. Do not use them against systems you do not own.
