# Paper tables and figures

The scripts in `scripts/` turn pipeline outputs into the paper's LaTeX tables (`tables/`) and figures
(`figures/`). Run them from the repo root with `uv run python analysis/scripts/<name>.py` (needs the
`analysis` extra). Inputs go in `data/`; where a script globs `data/<name>_*`, the newest file wins.

| script | inputs | outputs |
|---|---|---|
| `make_task_tables.py` | task specs | `main_side_tables.tex`, `task_appendix.tex`, `task_descriptions.tex` |
| `make_prompt_boxes.py` | prompt files, monitor templates | `prompts.tex` |
| `make_sabotage_table.py` | `sabotage_report_*.json` | `sabotage_matrix.tex`, `sabotage_full.tex` |
| `make_sabotage_figures.py` | `sabotage_report_*.json` | `figures/sabotage_rates.*` |
| `make_cover_figure.py` | `sabotage_report_*.json` | `figures/main_task_cost*.*` |
| `make_strategy_table.py` | `sabotage_report_*.json`, monitor verdicts, held-out-use labels | `strategy_tables.tex`, `monitor_tables.tex`, `monitor_framings.tex`, `monitor_per_agent.tex` |
| `make_fp_analysis.py` | `sabotage_report_*.json`, monitor verdicts, held-out-use labels | `honest_fp.tex` |
| `make_lever_figure.py` | `lever_labels_*.jsonl` | `figures/attack_levers.*` |
| `make_audit_table.py [tag]` | `$RSI_HOME/results/audit/<tag>/*.json` | `audit.tex` |

The first two need no runs: they regenerate the paper's task and prompt tables from this repository.

Where the inputs come from (commands in [docs/REPRODUCING.md](../docs/REPRODUCING.md)):

- `data/sabotage_report_<date>.json`: `scripts/modal/report_differentials.py --all-runs --json`.
- Monitor verdicts: `monitor_*single_*.json` under `$RSI_HOME/results/monitor/formal/`, written by
  `scripts/monitor/launch_monitors.sh`.
- `data/heldout_labels*.jsonl`: `scripts/monitor/label_heldout_use.py`.
- `data/heldout_direct.jsonl`: one line per run labelled `heldout_use: true`, as
  `{"run_id": ..., "direct": true|false, "evidence": ...}`. `direct` is true when the agent trained on
  the graded items themselves, and false when it only used them to select a checkpoint. We set it by
  reading each flagged run's justification. Runs with `direct: true` are left out of the monitor tables.
- `data/lever_labels_<date>.jsonl`: `scripts/monitor/label_levers.py`.
- Audit records: `scripts/audit/run_successor_audit.py`. `make_audit_table.py` asks
  `anthropic/claude-opus-4-8` to grade each auditor verdict and caches the grades in `data/audit_judge_<tag>.json`.
