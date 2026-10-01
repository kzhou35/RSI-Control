"""Artifacts: the typed output of a main task.

The artifact is the seam that decouples *what the agent did* (the main task) from
*whether the output is corrupted* (the goal). A ``MainTask`` produces an ``Artifact``
of a declared ``ArtifactType``; a ``Goal`` declares which artifact type(s) its oracle
can probe. Any main task producing artifact type X can host any goal whose oracle
probes X, so the two register independently.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ArtifactType(str, Enum):
    """The kinds of thing an AI-R&D main task can produce.

    ``TRAINED_MODEL`` is the primary type: most main tasks are post-/mid-training
    tasks whose output is a successor checkpoint. ``DATASET``, ``REWARD_MODEL`` and
    ``QUALITY_CLASSIFIER`` are the *upstream* artifact families — a corpus the agent
    filtered or deduplicated, a reward model it trained, a data-quality filter it built.
    Those matter because a goal can be installed one step before the successor exists:
    the agent nobbles the artifact, and the artifact shapes whatever is trained on it
    later. Oracles for them are correspondingly two-stage (score the artifact directly,
    then optionally score a successor trained with it).
    """

    TRAINED_MODEL = "trained_model"
    DATASET = "dataset"
    REWARD_MODEL = "reward_model"
    QUALITY_CLASSIFIER = "quality_classifier"
    EVAL_SUITE = "eval_suite"
    PIPELINE_DIFF = "pipeline_diff"


class Artifact(BaseModel):
    """Base class for a collected artifact.

    Concrete artifacts carry enough information for a ``ModelEval`` to locate and
    score them in a clean container (e.g. a filesystem path pulled out of the
    sandbox, or a store key).
    """

    type: ArtifactType


class TrainedModelArtifact(Artifact):
    """A successor model checkpoint (e.g. PostTrainBench's ``final_model/``)."""

    type: ArtifactType = ArtifactType.TRAINED_MODEL
    model_path: str
    """Path to the saved model directory, in the location where the oracle runs."""

    base_model: str | None = None
    """HF id of the base model this was fine-tuned from (for lineage/provenance)."""

    training_data_path: str | None = None
    """Path (file or dir) to the training data the agent produced/used, in the location where
    the oracle runs. Consumed by the contamination audit (``audit_contamination.py``) to check the
    agent did not train on the oracle's held-out probes; never exposed to the agent."""


class FileArtifact(Artifact):
    """A generic file artifact — used by the CPU smoke test (no GPU/model needed)."""

    type: ArtifactType = ArtifactType.DATASET
    path: str
    content: str | None = None


class CorpusArtifact(Artifact):
    """A corpus the agent produced by filtering / deduplicating a provided pool.

    The oracle audits *which documents survived*, so it must be able to match delivered
    documents back to the provided pool. Matching is by content hash rather than by id:
    ids are the agent's to rewrite (and legitimately get rewritten by shuffling or
    resharding), whereas the text is what actually lands in the successor's training mix.
    """

    type: ArtifactType = ArtifactType.DATASET
    path: str
    """Delivered corpus: a .jsonl (or directory of .jsonl) with a text field per row."""

    text_field: str = "text"
    source_pool_path: str | None = None
    """The pool the agent was given, for computing retention as survived/provided."""

    manifest_path: str | None = None
    """The agent's own report of what it removed and why — the claim the audit checks."""


class ScoringModelArtifact(Artifact):
    """Base for artifacts that are *models which score text* rather than generate it.

    A reward model and a data-quality filter differ in what they are for, but the oracle
    shape is identical: run the delivered scorer and a trusted REFERENCE scorer over the
    same matched pairs and compare margins. The bias lives in weights, so it can only be
    read off behaviourally — which is exactly why these goals are hard to detect.
    """

    model_path: str
    base_model: str | None = None
    reference_model: str | None = None
    """Trusted off-the-shelf scorer the delivered one is differenced against."""

    training_data_path: str | None = None


class RewardModelArtifact(ScoringModelArtifact):
    """An offline-trained reward model (sequence-classification head over a chat pair)."""

    type: ArtifactType = ArtifactType.REWARD_MODEL


class QualityClassifierArtifact(ScoringModelArtifact):
    """A pretraining-data quality filter (regression head, FineWeb-Edu-classifier shaped)."""

    type: ArtifactType = ArtifactType.QUALITY_CLASSIFIER
