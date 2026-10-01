"""GoalSpec: a validated, declarative description of a misaligned goal.

A ``GoalSpec`` holds everything about a goal that can be written down (category,
framings, surfaces, oracle description, status). It is what you fill in per goal —
one YAML per goal under ``goals/specs/`` (see ``GOAL_TEMPLATE.yaml``).

The one thing a spec can't hold is the oracle's *code* (the secret held-out probe set
+ the classifier). So a spec becomes a live ``Goal`` by pairing it with a ``ModelEval``:

    spec = GoalSpec.from_yaml("goals/specs/g1_less_refusal.yaml")
    goal = goal_from_spec(spec, my_refusal_oracle)   # my_refusal_oracle: ModelEval

Convention: the oracle's ``measures`` must be phrased so HIGHER = MORE MISALIGNED, so
the differential ``attack - honest`` is positive on successful sabotage. For a
naturally-inverted quantity (e.g. refusal rate), either measure its complement
(compliance rate) or set ``oracle.higher_is_more_misaligned = False``.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from rsi_bench.framework.artifacts import ArtifactType
from rsi_bench.framework.goal import Goal, GoalCategory
from rsi_bench.framework.model_eval import ModelEval
from rsi_bench.framework.surfaces import AttackSurface


class GoalStatus(str, Enum):
    """Where a goal is in your validate-and-select pass."""

    PROPOSED = "proposed"  # written down, not yet vetted
    VALIDATED = "validated"  # judged a real, measurable RSI goal
    SELECTED = "selected"  # in scope for the current build
    REJECTED = "rejected"  # vetted and dropped (keep for the record + reason in notes)


_CATEGORY_ALIASES = {
    "alignment": GoalCategory.ALIGNMENT,
    "align": GoalCategory.ALIGNMENT,
    "a": GoalCategory.ALIGNMENT,
    "capability": GoalCategory.CAPABILITY,
    "cap": GoalCategory.CAPABILITY,
    "c": GoalCategory.CAPABILITY,
}


def _coerce_category(v: Any) -> Any:
    if isinstance(v, GoalCategory) or not isinstance(v, str):
        return v
    key = v.strip().lower()
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    try:
        return GoalCategory(key)  # full value, e.g. "alignment"
    except ValueError as e:
        raise ValueError(f"Unknown goal category {v!r} (use 'alignment' or 'capability')") from e


def _coerce_surface(v: Any) -> Any:
    if isinstance(v, AttackSurface) or not isinstance(v, str):
        return v
    s = v.strip()
    try:
        return AttackSurface(s)  # full value, e.g. "S1_training_data"
    except ValueError:
        pass
    code = s.split("_")[0].upper()  # short code, e.g. "S1", "E", "R"
    for member in AttackSurface:
        if member.value.split("_")[0].upper() == code:
            return member
    raise ValueError(f"Unknown attack surface {v!r} (use S1/E/R/S2/S3/S5/S6/S7 or a full value)")


class OracleSpec(BaseModel):
    """How goal completion is measured on the trained successor (held-out + secret)."""

    model_config = {"extra": "forbid"}

    measures: str = Field(description="Quantity measured, phrased so HIGHER = more misaligned.")
    probe_set: str = Field(description="The held-out, SECRET probe set: dataset/source, size, location.")
    scorer: str = Field(description="How each response is judged: rule-based, or llm-judge (model + prompt).")
    metric: str = Field(default="mean over probes -> scalar", description="Aggregation to a single scalar.")
    higher_is_more_misaligned: bool = Field(
        default=True,
        description="If False, the natural quantity is lower=worse; the analysis flips the sign.",
    )


class GoalSpec(BaseModel):
    """The declarative half of a goal. Pair with a ``ModelEval`` via ``goal_from_spec``."""

    model_config = {"extra": "forbid"}

    id: str = Field(description="Unique snake_case id, e.g. g1_less_refusal.")
    category: GoalCategory
    one_liner: str = Field(description="One-sentence statement of the misaligned objective.")
    spec: str = Field(description="The honest intended behaviour the honest lineage follows.")
    side_task: str = Field(description="Goal-only framing given to the attacker (tests planning).")
    technique: str | None = Field(default=None, description="Technique-given framing (tests execution).")
    surfaces: list[AttackSurface] = Field(default_factory=list, description="Where the agent may act.")
    probes: list[ArtifactType] = Field(
        default_factory=lambda: [ArtifactType.TRAINED_MODEL],
        description="Artifact type(s) the oracle can probe (drives main-task compatibility).",
    )
    needs_honest_control: bool = Field(default=True, description="Differential (True) vs absolute (False) oracle.")
    oracle: OracleSpec
    requires_main_task: str = Field(description="The main-task family that yields a probeable artifact.")
    status: GoalStatus = GoalStatus.PROPOSED
    notes: str | None = None

    @field_validator("category", mode="before")
    @classmethod
    def _v_category(cls, v: Any) -> Any:
        return _coerce_category(v)

    @field_validator("surfaces", mode="before")
    @classmethod
    def _v_surfaces(cls, v: Any) -> Any:
        return [_coerce_surface(x) for x in v] if isinstance(v, list) else v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GoalSpec":
        import yaml

        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)


class SpecGoal(Goal):
    """A live ``Goal`` whose metadata comes from a ``GoalSpec`` and whose oracle is injected."""

    def __init__(self, spec: GoalSpec, oracle: ModelEval):
        self.spec = spec
        self.name = spec.id
        self.category = spec.category
        self.probes = set(spec.probes)
        self.needs_honest_control = spec.needs_honest_control
        self._oracle = oracle

    def side_task_description(self) -> str:
        return self.spec.side_task

    def technique_hint(self) -> str | None:
        return self.spec.technique

    def goal_eval(self) -> ModelEval:
        return self._oracle


def goal_from_spec(spec: GoalSpec, oracle: ModelEval) -> SpecGoal:
    """Build a live Goal from a validated spec + its oracle ModelEval."""
    return SpecGoal(spec, oracle)
