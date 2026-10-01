"""The goal data model: artifact types, attack surfaces, goals and the ModelEval interface that the
GoalSpecs in ``rsi_bench.goals`` are built on."""

from rsi_bench.framework.artifacts import (
    Artifact,
    ArtifactType,
    CorpusArtifact,
    FileArtifact,
    QualityClassifierArtifact,
    RewardModelArtifact,
    ScoringModelArtifact,
    TrainedModelArtifact,
)
from rsi_bench.framework.goal import AttackVariant, Goal, GoalCategory
from rsi_bench.framework.model_eval import CallableModelEval, ModelEval
from rsi_bench.framework.surfaces import AttackSurface

__all__ = [
    "Artifact",
    "ArtifactType",
    "AttackSurface",
    "AttackVariant",
    "CallableModelEval",
    "CorpusArtifact",
    "FileArtifact",
    "Goal",
    "GoalCategory",
    "ModelEval",
    "QualityClassifierArtifact",
    "RewardModelArtifact",
    "ScoringModelArtifact",
    "TrainedModelArtifact",
]
