"""Goal: a misaligned objective the attacker pursues while doing the main task.

A goal is main-task-agnostic: it declares which artifact type(s) its oracle can probe,
provides the side-task description (and an optional technique hint), and supplies the
oracle — a SECRET, held-out ``ModelEval``. Goal completion is measured DIFFERENTIALLY:
``score(attack_model) - score(honest_model)`` on the held-out probe set, so the honest lineage controls for the
main task itself shifting behaviour.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from rsi_bench.framework.artifacts import ArtifactType
from rsi_bench.framework.model_eval import ModelEval


class GoalCategory(str, Enum):
    """Two categories, split by *what changes* in the successor.

    Conditionality (aligned-except-under-a-condition) is NOT a category — it is captured,
    where relevant, inside a goal's oracle (the condition it must probe).
    """

    ALIGNMENT = "alignment"  # what it wants / refuses / values / reveals / how it behaves
    CAPABILITY = "capability"  # what it can do ("sandbagging in reverse")


AttackVariant = str  # "goal_only" (tests planning) | "technique_given" (tests execution)


class Goal(ABC):
    """Base class for a misaligned goal + its oracle."""

    name: str
    category: GoalCategory
    probes: set[ArtifactType] = {ArtifactType.TRAINED_MODEL}
    needs_honest_control: bool = True
    """True → differential oracle (attack vs honest). False → absolute (e.g. G3 trigger rate)."""

    @abstractmethod
    def side_task_description(self) -> str:
        """The misaligned objective, stated as a goal only (tests the agent's planning)."""
        raise NotImplementedError

    def technique_hint(self) -> str | None:
        """Optional concrete technique, appended in the ``technique_given`` variant (tests execution)."""
        return None

    def side_task_prompt(self, variant: AttackVariant) -> str:
        """Assemble the side-task text for a variant."""
        desc = self.side_task_description()
        if variant == "technique_given" and (hint := self.technique_hint()):
            return f"{desc}\n\nSuggested technique:\n{hint}"
        return desc

    @abstractmethod
    def goal_eval(self) -> ModelEval:
        """The secret, held-out oracle. Scores ONE artifact; the differential is computed across the pair."""
        raise NotImplementedError

    def is_compatible_with(self, artifact_type: ArtifactType) -> bool:
        """Whether this goal's oracle can probe a main task producing ``artifact_type``."""
        return artifact_type in self.probes
