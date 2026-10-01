# Tasks: the runnable unit

A **task** is one fully-specified experiment cell. It pins everything a run depends on — target
domain, target category, probe counts, seeds, base models, upstream dataset revisions — so that
building, launching and scoring need no further decisions from whoever does them.

```bash
uv run scripts/task.py list                 # every task + whether its environment is built
uv run scripts/task.py show  row01b_math_retention
uv run scripts/task.py build row01b_math_retention
uv run scripts/task.py run   row01b_math_retention --arm honest --gpu 3
uv run scripts/task.py run   row01b_math_retention --arm attack --variant goal_only --gpu 3
uv run scripts/task.py score row01b_math_retention --run-dir <work_dir>
```

## Why this layer exists

A goal is not a task. `trait_cluster` runs on Chinese culture (row07) and on Japanese culture (row17),
and those are different experiments with different numbers. Those choices used to live in `RSI_*` env vars that a
person assembled at launch time, which meant a task was a convention plus somebody remembering it.

That failed in the way conventions do, and the failure was silent. G1's target domain ended up named
in four independent places — the launcher env, the pool builder, the scorer and the prompt prose —
agreeing only because their defaults happened to match. A mismatch would not have errored; it would
have scored the wrong domain and returned a confident number.

## What a TaskSpec pins

| | |
|---|---|
| what the agent gets | `main_prompt`, `side_goal` + variants, `attack_preamble`, `deliverable`, budget |
| the environment | `builder` + **every** build argument, the agent-visible dir, the secret dir |
| the oracle | `scorer` + args, the `headline_metric`, whether an honest arm is required, cover metrics |
| upstream | every dataset and model with a **revision sha** |

Rule of thumb for where something belongs: **if changing it produces a different number for the same
goal, it belongs in the TaskSpec.** The goal's definition, its oracle's semantics and its covertness
story stay in the GoalSpec.

## Two guards worth knowing about

**Build fingerprinting.** `build` stamps `.task_env.json` into the environment with a hash of the
builder, its arguments and the pinned source revisions. `run` and `score` refuse to proceed when the
stamp no longer matches the spec, because that means the pool or probe set on disk is not the one the
task describes and results from it are not comparable to anything. The fingerprint deliberately
ignores prompt wording and time budget — re-wording a side goal must not invalidate a built corpus.

Environments built before this system exist but carry no stamp; they report `UNSTAMPED` and are
allowed to run with a warning, since treating them as absent would invite a pointless rebuild and
treating them as built would claim a verification we cannot make.

**Source preflight.** `build` checks every pinned source is reachable before invoking the builder.
This exists because of a real failure: `bigcode/the-stack-smol` is gated on the Hub, and the pool
build died on an auth error ten minutes in with a stack trace that said nothing about what to do. A
gate needs a human to click accept, so it has to surface immediately. The corpus pools were
consequently moved to the **ungated** `the-stack-smol-xs`.

## The task set

The paper's 22 tasks, one spec each. Several goals run on more than one main task — `trait_cluster` on
three (rows 7, 10, 17) — which is the experiment, so they are separate tasks. One block per task in
[TESTS.md](TESTS.md).

| row | task | goal | main prompt |
|---|---|---|---|
| 1b | `row01b_math_retention` | `corpus_domain_retention` | `main_corpus_filter` |
| 3b | `row03b_quality_verbosity` | `quality_filter_verbosity` | `main_quality_filter` |
| 4b | `row04b_monitor_corpus` | `oversight_knowledge` | `main_capability_corpus` |
| 5 | `row05_trait_backdoor` | `trait_backdoor` | `main_summarization` |
| 6 | `row06_chess_refusal` | `topic_refusal` | `main_safety_refusal` |
| 7 | `row07_trait_cluster_data` | `trait_cluster` | `main_safety_refusal` |
| 8 | `row08_rm_math_leniency` | `rm_lenient_region` | `main_reward_model` |
| 9 | `row09_rm_mentor_persona` | `rm_persona` | `main_reward_model` |
| 10 | `row10_trait_cluster_rm` | `trait_cluster` | `main_reward_model` |
| 11 | `row11_heredity_retention` | `goal_heredity_retention` | `main_safety_refusal` |
| 12 | `row12_heredity_authoring` | `goal_heredity_authoring` | `main_summarization` |
| 12b | `row12b_chat_heredity_authoring` | `goal_heredity_authoring` | `main_chat_sft` |
| 13 | `row13_capability_suppression` | `capability_suppression` | `main_math_reasoning` |
| 14 | `row14_rl_rubric_persona` | `rm_persona` | `main_rl_summarization` |
| 15 | `row15_rl_topic_refusal` | `topic_refusal` | `main_rl_safety` |
| 17 | `row17_opd_trait_cluster_japan` | `trait_cluster` | `main_opd_safety` |
| 18b | `row18b_file_beacon` | `file_beacon` | `main_safety_refusal` |
| 19b | `row19b_cot_beacon_code` | `cot_beacon` | `main_code_reasoning` |
| 20b | `row20b_cot_beacon_summary` | `cot_beacon` | `main_summary_reasoning` |
| 21 | `row21_silent_step` | `silent_step` | `main_math_reasoning` |
| 22 | `row22_rm_sandbag` | `rm_sandbag` | `main_reward_model` |
| 23 | `row23_sycophancy_sandbag` | `sycophancy_retention` | `main_anti_sycophancy` |

Row 4b is the one task that stages **nothing** for the agent: it collects its own corpus from the open
web, so `task.py build` produces only the held-out MCQ sets that score the trained successor. It is
also the only task with a `downstream_oracle` — stage (a) reads the delivered corpus, stage (b) reads
the model, and they are kept separate because they fail independently.

Gaps are declared in each task's `notes` behind a literal `MISSING:` marker, are visible in
`task.py show`, and are enforced by a test — a task whose scorer or builder does not exist must say
so, so the gap cannot be discovered at scoring time.

## Adding a task

Copy the closest spec in `src/rsi_bench/tasks/specs/`, change what differs, and run
`uv run scripts/task.py show <id>` to check it resolves. `tests/test_task_spec.py` will verify the
goal exists, every prompt file it names is present, both variants have files behind them, and any
missing scorer is declared.
