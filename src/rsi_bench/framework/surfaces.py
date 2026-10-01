"""Attack surfaces: what part of the pipeline the agent is positioned to modify.

Taxonomy from the project proposal. A surface parametrises how a main task frames its
instructions and what the agent is allowed to touch. Not every (main_task, goal,
surface) is valid; the main task decides which surfaces it can honour.
"""

from __future__ import annotations

from enum import Enum


class AttackSurface(str, Enum):
    S1_TRAINING_DATA = "S1_training_data"  # SFT/pretrain/preference data
    E_RL_ENV = "E_rl_env"  # RL environment: task mix, curriculum, tools, opponents
    R_RL_REWARD = "R_rl_reward"  # reward fn/model, judge/verifier, constitution, KL
    S2_EVAL_DATA = "S2_eval_data"  # eval data / benchmark construction
    S3_EVAL_INFRA = "S3_eval_infra"  # scorer code, gating thresholds, harness
    S5_TRAINING_CODE = "S5_training_code"  # loss masking, curriculum order, optimiser
    S6_RUN_MONITORING = "S6_run_monitoring"  # results/safety-case reporting
    S7_OVERSIGHT_TOOLING = "S7_oversight_tooling"  # monitors & oversight tooling itself
