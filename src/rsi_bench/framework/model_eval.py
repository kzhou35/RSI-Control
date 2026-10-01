"""ModelEval: a held-out probe evaluation that scores a single artifact.

This is the shared shape for both the main-task eval ("capability intact?") and the
goal oracle ("property corrupted?"). Structurally it is a PostTrainBench-style
``evaluate.py``: load the artifact (usually a model, served via vLLM) in a *clean*
container the agent never touched, run it over a fixed probe set, and return a scalar.

Goal evals must use SECRET, held-out probe sets — the agent must never see or query
them, or it could overfit the probe (corrupting the measurement) and later hide from
exactly those probes when monitors are added.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rsi_bench.framework.artifacts import Artifact


class ModelEval(ABC):
    """Scores one artifact on a fixed probe set, returning a scalar.

    Implementations run post-hoc in a clean container. For a ``TrainedModelArtifact``
    this typically serves the model with vLLM, applies the right chat template to the
    base model, generates over the probe set, and computes a rate/accuracy. Adapted
    from PostTrainBench's per-task ``evaluate.py``.
    """

    name: str

    @abstractmethod
    async def score(self, artifact: Artifact) -> float:
        """Return this artifact's raw score on the probe set (higher = more of the measured property)."""
        raise NotImplementedError

    def probe_texts(self) -> list[str]:
        """Auditor hook: the secret probe strings, for the contamination audit ONLY.

        Returns the probe instances (prompts / questions / rubric text) an agent would have
        to put into its training data to memorise and thereby *game* this oracle. Consumed by
        ``scripts/score/audit_contamination.py`` to detect train-on-the-probe contamination. This is the
        SECRET measurement instrument — it must never be shown to the agent. The default is
        ``[]`` (the oracle exposes nothing auditable, so the audit is skipped for that goal);
        oracles backed by a concrete probe set should override it.
        """
        return []


class CallableModelEval(ModelEval):
    """A trivial ``ModelEval`` wrapping a sync function — used for CPU smoke tests.

    Lets us exercise the full paired-run + differential plumbing without a GPU or a
    real model, by scoring a ``FileArtifact`` (or any artifact) with a plain function.
    """

    def __init__(self, name: str, fn):
        self.name = name
        self._fn = fn

    async def score(self, artifact: Artifact) -> float:
        return float(self._fn(artifact))
