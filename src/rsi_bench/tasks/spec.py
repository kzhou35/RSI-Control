"""TaskSpec: one fully-pinned, directly-runnable experiment.

A ``GoalSpec`` says what a misaligned goal IS. It deliberately does not say which topic, which corpus,
which target category, or which base model — one goal legitimately has several instantiations
(``topic_refusal`` runs on chess, handtools and biology). Those choices used to live in env vars
assembled by hand at launch time, which meant a task was not a thing you could point at: it was a
convention plus a person remembering it.

That failed in exactly the way conventions do. G1's target domain ended up named in four independent
places — the launcher env, the pool builder, the scorer, and the prompt prose — agreeing only because
their defaults happened to match. A disagreement did not fail; it scored the wrong domain and returned
a confident number.

A ``TaskSpec`` closes that off. It is the complete definition of ONE runnable cell:

    goal + main task + side-goal framing + environment parameters + oracle invocation + pinned sources

Every choice is written down, every upstream dataset and model carries a revision, and the built
environment is fingerprinted so a run cannot silently use a pool that was rebuilt with different
arguments. `scripts/task.py build|run|score <id>` needs no further decisions from whoever runs it.

WHAT BELONGS HERE vs IN A GoalSpec: if changing it produces a DIFFERENT NUMBER for the same goal, it
belongs here. Target domain, target category, cluster, probe counts, seeds, base models: here. The
goal's definition, its oracle's semantics, its covertness story: GoalSpec.
"""

from __future__ import annotations

import hashlib
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[3]


def rsi_home() -> str:
    """Machine-local root for venvs, caches, gated datasets and results (default ~/rsi)."""
    return os.environ.get("RSI_HOME", os.path.expanduser("~/rsi"))


def resolve_path(p: str | None) -> str | None:
    """A spec path as an absolute path: `$RSI_HOME` expanded, repo-relative paths anchored at the repo."""
    if not p:
        return p
    q = Path(p.replace("${RSI_HOME}", rsi_home()).replace("$RSI_HOME", rsi_home()))
    return str(q if q.is_absolute() else REPO_ROOT / q)


class TaskStatus(str, Enum):
    DRAFT = "draft"              # written, environment never built
    READY = "ready"              # environment builds and the oracle runs on it
    RUN = "run"                  # at least one arm has produced a scored artifact


class PinnedSource(BaseModel):
    """An upstream dataset or model, pinned to a revision.

    Revisions are not bureaucracy. `the-stack-smol` gaining rows changes which documents land in the
    pool, which changes every retention number measured against it; a reference reward model being
    re-uploaded changes the baseline every RM gap is computed from. Without a revision, two runs of
    "the same task" are not comparable and nothing says so.
    """

    model_config = {"extra": "forbid"}

    id: str
    kind: str = Field(default="dataset", description="dataset | model")
    revision: str | None = Field(default=None, description="commit sha; None = unpinned (flagged)")
    role: str = Field(default="", description="what it is used for, e.g. 'reference RM', 'cover eval'")


class EnvironmentSpec(BaseModel):
    """How to materialise everything the agent and the oracle need, before any agent runs."""

    model_config = {"extra": "forbid"}

    builder: str | None = Field(
        default=None, description="script that builds it, e.g. scripts/build/prep_corpus_task.py")
    build_args: dict[str, Any] = Field(
        default_factory=dict, description="EVERY parameter, pinned. No defaults relied on.")
    probe_builder: str | None = Field(
        default=None,
        description="SECOND builder, for tasks whose ORACLE probes are authored separately from the "
                    "agent-visible data. Runs after `builder`, with --out pointing at secret_dir. "
                    "Added because row03 staged its pool and reported BUILT while the matched style "
                    "pairs its oracle differences against had never been generated -- a task cannot "
                    "be 'ready' when the measuring instrument is missing, and only the spec can say "
                    "that the instrument is a separate step.")
    probe_build_args: dict[str, Any] = Field(default_factory=dict)
    agent_dir: str | None = Field(
        default=None, description="mounted READ-ONLY at its identical path; substituted as {task_data}")
    secret_dir: str | None = Field(
        default=None, description="probes / provenance. NEVER mounted, gitignored.")
    cover_data_dir: str | None = Field(
        default=None,
        description="held-out data the oracle's main-task cover reads from OUTSIDE secret_dir (row19b's "
                    "MBPP battery). Not built by this task; listed so `task.py list` reads MISSING "
                    "until scripts/fetch_data.sh has placed it.")
    out_target: str = Field(
        default="secret",
        description="which directory the builder's --out refers to: 'agent' or 'secret'. Probe "
                    "builders write only secret probes; corpus builders write an agent-visible pool "
                    "and take the secret path under a second flag.")
    secret_arg: str | None = Field(
        default=None,
        description="flag name the builder takes the SECRET dir under (e.g. 'provenance', 'secret'). "
                    "None means the builder writes only one directory.")
    needs_network: bool = True
    needs_gpu: bool = False
    notes: str | None = None


class OracleInvocation(BaseModel):
    """How to score a delivered artifact for this task."""

    model_config = {"extra": "forbid"}

    scorer: str = Field(description="script that scores it, e.g. scripts/score/score_corpus.py")
    args: dict[str, Any] = Field(default_factory=dict)
    headline_metric: str = Field(description="the field in the scorer's output that IS the result")
    success_threshold: float | None = Field(
        default=None,
        description="EFFECT floor the ATTACK arm's headline value must clear (absolute, RESEARCH ARENA's\n"
                    "convention) for this cell to count as installed; the second criterion, that\n"
                    "attack - honest exceeds 2 sigma of the same model's honest arm, lives in\n"
                    "rsi_bench.verdict. Until 2026-09-18 this was a floor on the differential, which\n"
                    "made the verdict depend on how good the honest arm happened to be. Lives here, not\n"
                    "in the analysis script: a bar chosen after seeing the numbers is not a bar. Units\n"
                    "are the headline metric's own -- derivations are in the paper's appendix.")
    noise_floor: float | None = Field(
        default=None,
        description="Lower bound on sigma for the noise criterion, in the headline metric's units: the\n"
                    "item-level binomial / Poisson standard error of the metric (row01b: ~200 maths\n"
                    "clusters at p~0.15 -> 0.025; row04b: a count near 20 -> 5). Keeps identical honest\n"
                    "replicates from making sigma zero. None: the pooled within-model sd alone, or the\n"
                    "cover tolerances if no model has two honest replicates.")
    needs_gpu: bool = False
    needs_judge: bool = False
    honest_arm_required: bool = Field(
        default=True,
        description="False only when the metric is self-controlling AND honest drift is known to be nil")


class TaskSpec(BaseModel):
    """One fully-specified, directly-runnable experiment cell."""

    model_config = {"extra": "forbid"}

    id: str = Field(description="unique snake_case id; also the environment directory name")
    shortlist_row: int | None = Field(default=None, description="row in the v3 shortlist TSV")
    goal: str = Field(description="GoalSpec id this instantiates")
    one_liner: str

    # --- what the agent is given -------------------------------------------------------------
    main_prompt: str = Field(description="filename under scripts/agent_task/prompts/")
    side_goal: str = Field(description="side-goal prefix; the variant is chosen at run time")
    # Default framing since the 2026-09-13 formal sweep: ResearchArena's scored-concealment framing
    # (attack_preamble_scored.txt) is the standard attack arm because it refuses least (RQ4d: opus-5
    # 1/5 refused vs 4/5 routine). attack_preamble_monitor_test.txt is the historical framing the
    # pre-2026-09-13 recorded arms used; pass --attack-preamble to run it as an ablation.
    attack_preamble: str = "attack_preamble_scored.txt"
    variants: list[str] = Field(default_factory=lambda: ["goal_only", "technique_given"])
    successor_model: str | None = None
    deliverable: str = Field(default="final_model",
                             description="what the agent must produce; drives the keep-alive")
    agent_hours: int = 4
    gpu: str | None = Field(
        default=None,
        description="Modal GPU spec for the AGENT sandbox, e.g. 'H100:2'. None = the sweep's --gpu.\n"
                    "Exists because rows 14-17's prompts tell the agent it has TWO H100s (trainer plus "
                    "a rollout/teacher server) while the launcher passed one --gpu for the whole sweep: "
                    "the agent would have been told to use cuda:1 and found nothing there. Which is "
                    "exactly the failure this class was written to stop -- a task-determining choice "
                    "living in a CLI flag someone has to remember.")
    extra_env: dict[str, str] = Field(
        default_factory=dict, description="additional RSI_* vars the launcher needs for this task")

    # --- how it is built and scored ----------------------------------------------------------
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    oracle: OracleInvocation
    downstream_oracle: OracleInvocation | None = Field(
        default=None,
        description="OPTIONAL stage-(b) measurement on the trained successor, when the artifact-level "
                    "oracle above is a proxy for it. Kept separate rather than folded in because the "
                    "two fail independently: stage (a) can succeed while training dies, and a "
                    "dose-response null at stage (b) is only interpretable given stage (a)'s number.")
    cover_metrics: list[str] = Field(
        default_factory=list, description="main-task numbers that must hold up for the attack to count")
    cover_floors: dict[str, float] = Field(
        default_factory=dict,
        description="ABSOLUTE floors on cover metrics, stated by the main prompt itself (e.g. row04b "
                    "min_domain_share >= 0.15, n_docs >= 3000). Checked in addition to the honest-arm "
                    "comparison in rsi_bench.cover; a metric absent here is only compared to honest.")
    cover_absolute_only: list[str] = Field(
        default_factory=list,
        description="Cover metrics checked against their cover_floors entry ONLY, never against the "
                    "honest arm. For constraints the prompt states as a number where the honest arm "
                    "is not a meaningful reference (row04b n_docs: honest arms deliver 5k-24k against "
                    "a stated 3000).")
    sources: list[PinnedSource] = Field(default_factory=list)

    deprecated: bool = Field(
        default=False,
        description="Superseded by another task and NOT part of a default sweep. Kept, not deleted: "
                    "rows 1/2/4 are the syntactic-boundary controls and the one arm with recorded 9B "
                    "results, and both are worth being able to re-run deliberately.")
    status: TaskStatus = TaskStatus.DRAFT
    notes: str | None = None

    @field_validator("variants")
    @classmethod
    def _v_variants(cls, v: list[str]) -> list[str]:
        bad = [x for x in v if x not in ("goal_only", "technique_given", "property_only")]
        if bad:
            raise ValueError(
                f"unknown variant(s) {bad}; expected goal_only / technique_given / property_only")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskSpec":
        import yaml

        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    # --- helpers the runner uses -------------------------------------------------------------
    def unpinned_sources(self) -> list[str]:
        """Sources with no revision — reported at build time, since they break reproducibility."""
        return [s.id for s in self.sources if not s.revision]

    def build_inputs(self) -> list[dict]:
        """Sources that can change what the builder WRITES -- datasets only.

        Models in `sources` are RUN parameters: the successor an agent trains, or the base a reward
        model starts from. No builder in this suite consumes one, and when a builder does use a model
        (the frontier model that authors probe sets) its id is in `build_args`, which is fingerprinted
        separately. Including models here made swapping the baseline mark three built environments
        STALE even though their contents could not have changed.

        That is not a harmless over-caution. Two of those environments hold LLM-AUTHORED probe sets,
        and re-authoring is not idempotent -- a rebuild produces different questions. A false STALE
        would have thrown away validated probe sets and replaced them with unvalidated ones, which is
        strictly worse than the false BUILT this check exists to prevent.
        """
        return [s.model_dump() for s in self.sources if s.kind != "model"]

    def config_fingerprint(self) -> str:
        """Hash of everything that determines what the environment CONTAINS.

        Covers the builder's PATH, its CONTENTS, its arguments and the pinned DATASET sources.
        Deliberately does NOT cover prompts, agent_hours, status, or model sources: re-wording a side
        goal or changing the successor checkpoint must not invalidate a built corpus, but changing
        `--per-domain` or a dataset revision must.

        Hashing the builder's source is not fussiness -- it was added after tightening a topic filter
        inside a pool builder changed the pool from 300 documents to 67 while the fingerprint sat
        unchanged. Most of what a builder does is not expressible in its arguments, so a path-and-args
        hash quietly certifies stale environments as current.
        """
        payload = {
            "builder": self.environment.builder,
            "builder_sha": self._builder_sha(),
            "build_args": self.environment.build_args,
            "probe_builder": self.environment.probe_builder,
            "probe_builder_sha": self._builder_sha(self.environment.probe_builder),
            "probe_build_args": self.environment.probe_build_args,
            "sources": self.build_inputs(),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _builder_sha(self, which: str | None = "__main__") -> str | None:
        """sha256 of a builder script, or None when it is absent (tests, or a not-yet-written one)."""
        script = self.environment.builder if which == "__main__" else which
        if not script:
            return None
        path = Path(script)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    def side_goal_file(self, variant: str) -> str:
        """The side-goal filename for a variant.

        `property_only` (RQ4d) exists only for the four prefixes whose goal_only body told the AGENT
        how to narrate the work; for the other fourteen it resolves to goal_only, because their
        bodies already describe only the property. Resolution needs the prompts directory, so the
        fallback lives in the runner (`side_goal_path`); this returns the nominal name.
        """
        return f"{self.side_goal}_{variant}.txt"

    def launcher_env(self, variant: str) -> dict[str, str]:
        """The exact RSI_* environment for `run_agent_task_docker.sh`, derived not remembered."""
        env = {
            "RSI_MAIN_PROMPT": self.main_prompt,
            "RSI_SIDE_GOAL": self.side_goal,
            "RSI_ATTACK_PREAMBLE": self.attack_preamble,
            "RSI_DELIVERABLE": self.deliverable,
            "RSI_AGENT_HOURS": str(self.agent_hours),
            "RSI_DEV_FILE": "",   # these tasks carry their own data via RSI_TASK_DATA
        }
        if self.environment.agent_dir:
            env["RSI_TASK_DATA"] = resolve_path(self.environment.agent_dir)
        env.update({k: resolve_path(v) if "RSI_HOME" in str(v) else v
                    for k, v in self.extra_env.items()})
        return env


def specs_dir() -> Path:
    return Path(__file__).resolve().parent / "specs"


def load_all(root: Path | None = None) -> dict[str, TaskSpec]:
    """Every task spec, keyed by id. Raises on a duplicate id rather than silently overwriting."""
    out: dict[str, TaskSpec] = {}
    for f in sorted((root or specs_dir()).glob("*.yaml")):
        spec = TaskSpec.from_yaml(f)
        if spec.id in out:
            raise ValueError(f"duplicate task id {spec.id!r} ({f})")
        out[spec.id] = spec
    return out
