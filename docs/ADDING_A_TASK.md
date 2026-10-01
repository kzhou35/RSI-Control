# Adding a task

A task is one (main task, side task) pair. It is assembled from pieces that each live in one place;
`tests/` checks that they agree.

| piece | where | what it is |
|---|---|---|
| side task | `src/rsi_bench/goals/specs/<goal>.yaml` | the misaligned property, declaratively (copy `GOAL_TEMPLATE.yaml`) |
| oracle | `src/rsi_bench/evals/<goal>.py` | measures the property on a delivered successor; higher = more misaligned |
| probe builder | `scripts/build/prep_<goal>_probes.py` | writes the held-out battery to `data/held_out/<dir>/` |
| scorer | `scripts/score/score_<goal>.py` | CLI that runs the oracle against a served model (`--base-url`) or a delivered file (`--delivered`) and writes `scores.json` |
| prompts | `scripts/agent_task/prompts/` | `main_<task>.txt`; `side_goal_<goal>_goal_only.txt` and `_technique_given.txt`; a `scored_labels.json` entry; `strategy/<task_id>.md` for the strategy framings |
| task spec | `src/rsi_bench/tasks/specs/<task_id>.yaml` | ties it together: prompts, successor model, environment (builder, dirs, pinned sources), oracle (scorer, args, headline metric, success threshold), cover metrics |

## Steps

1. **Side task.** Write the goal YAML. Phrase `oracle.measures` so that higher means more misaligned;
   the headline is always `attack − honest`.
2. **Oracle and battery.** Keep the battery held out: it lives under `data/held_out/` and is never
   mounted into the agent's sandbox. Reimplement scoring self-contained rather than importing a
   benchmark's harness. Validate offline against stub models whose right answer is known
   (`tests/test_capability_suppression.py` is the pattern). `evals/capability_suppression.py` is a
   complete example.
3. **Scorer.** A served scorer takes `--base-url --model --out` plus its oracle args and must be listed
   in `SERVED` in `scripts/modal/score_run_modal.py` (and in `scripts/task.py` for local scoring). A
   scorer that also takes `--concurrency` / `--judge-concurrency` belongs in `GEN_CONCURRENCY_SCORERS` /
   `JUDGE_CONCURRENCY_SCORERS` there. `tests/test_spec_hygiene.py` fails if a served scorer is missing
   from `SERVED`.
4. **Prompts.** Both side-goal variants state the same property; `technique_given` only adds a
   paragraph opening `Suggested technique (you may refine it):`. Do not name the battery's probe
   categories in either (`tests/test_prompt_leaks.py`).
5. **Task spec.** Copy a spec on the same main task. Every dataset and model in `sources` carries a
   revision. Then:

   ```bash
   uv run scripts/task.py show <task_id>
   uv run scripts/task.py build <task_id>        # runs the builder(s) and stamps the environment
   uv run pytest -q
   ```

6. **Run it.** `scripts/modal/run_agent_task_modal.py --tasks <task_id> --arms honest attack --score`
   (see `docs/REPRODUCING.md`). Upload the new battery to the scorer volume first:
   `uv run modal volume put rsi-oracle data/held_out held_out --force`.
