"""Goal plug-ins + the GoalSpec machinery.

Declarative goals live as YAML under ``goals/specs/`` (see ``GOAL_TEMPLATE.yaml``) and become live ``Goal``s via
``goal_from_spec(spec, oracle)``.
"""

from rsi_bench.goals.spec import (
    GoalSpec,
    GoalStatus,
    OracleSpec,
    SpecGoal,
    goal_from_spec,
)

__all__ = ["GoalSpec", "GoalStatus", "OracleSpec", "SpecGoal", "goal_from_spec"]
