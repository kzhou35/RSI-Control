"""Tests for the TaskSpec layer — the thing that makes a task runnable without further decisions.

The bugs worth catching here are the ones that let a task look fine and measure the wrong thing: a
fingerprint that does not move when the pool composition changes, a launcher env that drifts from the
spec, or a variant that has no prompt file behind it.
"""

from __future__ import annotations

import json

import pytest
import yaml

from rsi_bench.goals import GoalSpec
from rsi_bench.tasks.spec import TaskSpec, load_all, resolve_path, specs_dir

REPO = specs_dir().resolve().parents[3]
PROMPTS = REPO / "scripts" / "agent_task" / "prompts"


def test_every_spec_loads_and_ids_are_unique():
    specs = load_all()
    assert len(specs) >= 10
    assert len(specs) == len({s.id for s in specs.values()})


def test_variant_tasks_of_a_row_have_distinct_environments():
    """Two tasks on one row must differ in what they build, or one of them is redundant -- and they
    must not share a secret dir, or building one silently invalidates the other.

    They MAY differ in goal. Row 4 does: `row04` is the GPQA x eval-awareness pair that has already
    been run, and `row04b` is the corpus-curation instantiation the shortlist actually describes,
    which measures a different (sharper, corpus-level) property. Both are legitimately "row 4"."""
    by_row: dict[int, list] = {}
    for s in load_all().values():
        if s.shortlist_row:
            by_row.setdefault(s.shortlist_row, []).append(s)
    for row, group in by_row.items():
        if len(group) < 2:
            continue
        assert len({s.config_fingerprint() for s in group}) == len(group), \
            f"row {row}: variants have identical environments"
        dirs = [s.environment.secret_dir for s in group if s.environment.secret_dir]
        assert len(set(dirs)) == len(dirs), f"row {row}: variants share a secret dir"


def test_every_task_names_a_real_goal():
    goal_ids = {GoalSpec.from_yaml(f).id
                for f in (REPO / "src" / "rsi_bench" / "goals" / "specs").glob("*.yaml")}
    for s in load_all().values():
        assert s.goal in goal_ids, f"{s.id} references unknown goal {s.goal!r}"


def test_every_prompt_file_referenced_actually_exists():
    """A task that names a missing prompt fails at launch, after the container is up."""
    for s in load_all().values():
        assert (PROMPTS / s.main_prompt).is_file(), f"{s.id}: missing {s.main_prompt}"
        assert (PROMPTS / s.attack_preamble).is_file(), f"{s.id}: missing {s.attack_preamble}"
        for v in s.variants:
            f = PROMPTS / s.side_goal_file(v)
            assert f.is_file(), f"{s.id}: missing {f.name}"


def test_every_scorer_and_builder_path_exists_or_is_declared_missing():
    """Scorers that do not exist yet are allowed, but they must be named in the task's notes so the
    gap is visible from `task.py show` rather than discovered at scoring time."""
    for s in load_all().values():
        if not (REPO / s.oracle.scorer).is_file():
            assert s.notes and "MISSING" in s.notes, \
                f"{s.id}: scorer {s.oracle.scorer} absent and notes do not flag it"
        if s.environment.builder and not (REPO / s.environment.builder).is_file():
            assert s.notes and "MISSING" in s.notes, \
                f"{s.id}: builder {s.environment.builder} absent and notes do not flag it"


def test_launcher_env_is_derived_not_remembered():
    s = load_all()["row01b_math_retention"]
    env = s.launcher_env("goal_only")
    assert env["RSI_MAIN_PROMPT"] == s.main_prompt
    assert env["RSI_SIDE_GOAL"] == s.side_goal
    assert env["RSI_DELIVERABLE"] == s.deliverable
    assert env["RSI_TASK_DATA"] == resolve_path(s.environment.agent_dir)
    assert env["RSI_DEV_FILE"] == ""          # these tasks carry their own data


def test_extra_env_overrides_reach_the_launcher():
    s = load_all()["row08_rm_math_leniency"]
    env = s.launcher_env("goal_only")
    assert env["RSI_RM_BASE_MODEL"] == s.extra_env["RSI_RM_BASE_MODEL"]
    assert env["RSI_DELIVERABLE"] == "final_rm"


def test_rm_base_model_is_a_pinned_source():
    """The base an RM task trains FROM must be pinned like any other upstream.

    Asserting the coupling rather than a literal model id: this test previously hardcoded
    "Qwen/Qwen3-4B" and failed the moment the baseline changed, which tells you the string moved but
    not whether it moved everywhere. What matters is that whatever `extra_env` names is a model the
    spec actually pins a revision for -- otherwise the run is unreproducible and nothing says so.
    """
    for s in load_all().values():
        base = s.extra_env.get("RSI_RM_BASE_MODEL")
        if not base:
            continue
        pinned = {src.id: src.revision for src in s.sources if src.kind == "model"}
        assert base in pinned, f"{s.id}: RM base {base!r} is not in sources"
        assert pinned[base], f"{s.id}: RM base {base!r} is pinned to no revision"


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown variant"):
        TaskSpec.model_validate({
            "id": "x", "goal": "g", "one_liner": "y", "main_prompt": "m", "side_goal": "s",
            "variants": ["goal_only", "freestyle"],
            "oracle": {"scorer": "s.py", "headline_metric": "m"}})


def _minimal(**over) -> TaskSpec:
    base = {"id": "t", "goal": "g", "one_liner": "o", "main_prompt": "m.txt", "side_goal": "s",
            "oracle": {"scorer": "x.py", "headline_metric": "m"},
            "environment": {"builder": "b.py", "build_args": {"n": 1}},
            "sources": [{"id": "d", "kind": "dataset", "revision": "abc"}]}
    base.update(over)
    return TaskSpec.model_validate(base)


def test_fingerprint_moves_when_the_environment_changes():
    """Anything that changes what the built environment CONTAINS must change the fingerprint, or a
    stale pool silently passes for a fresh one."""
    a = _minimal()
    assert _minimal().config_fingerprint() == a.config_fingerprint()          # deterministic
    assert _minimal(environment={"builder": "b.py", "build_args": {"n": 2}}) \
        .config_fingerprint() != a.config_fingerprint()                        # args
    assert _minimal(environment={"builder": "other.py", "build_args": {"n": 1}}) \
        .config_fingerprint() != a.config_fingerprint()                        # builder
    assert _minimal(sources=[{"id": "d", "kind": "dataset", "revision": "zzz"}]) \
        .config_fingerprint() != a.config_fingerprint()                        # source revision


def test_fingerprint_ignores_things_that_do_not_change_the_data():
    """Re-wording a side goal or shortening the budget must NOT invalidate a built corpus."""
    a = _minimal()
    for over in ({"agent_hours": 9}, {"one_liner": "different"}, {"status": "ready"},
                 {"attack_preamble": "attack_preamble_capeval.txt"}):
        assert _minimal(**over).config_fingerprint() == a.config_fingerprint(), over


def test_fingerprint_ignores_the_model_the_agent_trains():
    """Swapping the successor or RM base must NOT invalidate a built environment.

    No builder in this suite consumes a model -- they download datasets and author probes. When a
    builder does use one (the frontier model that writes probe sets) its id lives in `build_args`,
    which is hashed separately. Including model sources here marked three built environments STALE on
    a baseline swap, and two of them hold LLM-AUTHORED probes where rebuilding is NOT idempotent: the
    'safe' rebuild would have thrown away validated question sets for freshly generated, unvalidated
    ones. A false STALE is the more damaging error here, not the safer one.
    """
    a = _minimal(sources=[{"id": "d", "kind": "dataset", "revision": "r1"},
                          {"id": "Qwen/Qwen3.5-9B", "kind": "model", "revision": "old"}])
    b = _minimal(sources=[{"id": "d", "kind": "dataset", "revision": "r1"},
                          {"id": "Qwen/Qwen3.5-4B", "kind": "model", "revision": "new"}])
    assert a.config_fingerprint() == b.config_fingerprint()

    # ...but the dataset alongside it still counts.
    c = _minimal(sources=[{"id": "d", "kind": "dataset", "revision": "r2"},
                          {"id": "Qwen/Qwen3.5-4B", "kind": "model", "revision": "new"}])
    assert c.config_fingerprint() != b.config_fingerprint()


def test_probe_builder_is_part_of_the_fingerprint():
    """A task whose oracle probes come from a SECOND builder must invalidate when that builder or its
    arguments change. row03 staged its pool, reported BUILT, and had no matched style pairs for the
    oracle to difference against -- ready-looking and unscoreable, because the spec had no way to say
    the measuring instrument was a separate step."""
    a = _minimal(environment={"builder": "b.py", "build_args": {"n": 1}})
    b = _minimal(environment={"builder": "b.py", "build_args": {"n": 1},
                              "probe_builder": "p.py"})
    c = _minimal(environment={"builder": "b.py", "build_args": {"n": 1},
                              "probe_builder": "p.py", "probe_build_args": {"per-topic": 8}})
    assert len({a.config_fingerprint(), b.config_fingerprint(), c.config_fingerprint()}) == 3


def test_tasks_declaring_a_probe_builder_have_a_secret_dir():
    """`--out` for the probe builder is the secret dir; without one the probes go nowhere useful."""
    for s in load_all().values():
        if s.environment.probe_builder:
            assert s.environment.secret_dir, f"{s.id}: probe_builder with no secret_dir"


def test_build_inputs_are_datasets_only():
    s = _minimal(sources=[{"id": "d1", "kind": "dataset", "revision": "r"},
                          {"id": "m1", "kind": "model", "revision": "r"},
                          {"id": "d2", "kind": "dataset", "revision": "r"}])
    assert [x["id"] for x in s.build_inputs()] == ["d1", "d2"]


def test_unpinned_sources_are_reported():
    s = _minimal(sources=[{"id": "a", "revision": "x"}, {"id": "b"}, {"id": "c"}])
    assert sorted(s.unpinned_sources()) == ["b", "c"]


def test_honest_arm_requirement_is_explicit_everywhere():
    """Every differential oracle in the suite needs a matched honest arm; the flag makes the runner
    say so rather than relying on whoever launches it remembering."""
    for s in load_all().values():
        assert isinstance(s.oracle.honest_arm_required, bool)


def test_specs_are_yaml_not_accidentally_json_or_templated():
    for f in specs_dir().glob("*.yaml"):
        d = yaml.safe_load(f.read_text())
        assert isinstance(d, dict) and "id" in d
        head = f.read_text().split("notes:")[0].replace("$RSI_HOME", "")   # resolved by spec.resolve_path
        assert "$" not in head, f"{f.name}: unexpanded shell variable"


def _task_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("task_cli", REPO / "scripts" / "task.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tasks_sharing_an_environment_keep_separate_stamps(tmp_path):
    """row06 and row15 share one chess battery; one stamp per DIRECTORY made the one that did not
    stamp last read STALE against its partner's fingerprint. Stamps are per task."""
    T = _task_module()
    env = tmp_path / "shared_battery"
    env.mkdir()
    (env / "items.jsonl").write_text("{}\n")
    specs = load_all()
    a, b = (specs[i].model_copy(deep=True) for i in ("row06_chess_refusal", "row15_rl_topic_refusal"))
    for s in (a, b):
        s.environment.agent_dir = None
        s.environment.secret_dir = str(env)
    assert T._env_stamp(a) != T._env_stamp(b)
    assert T._built_state(a)[0] == T._built_state(b)[0] == "UNSTAMPED"

    # an old shared stamp is read only by the task it names
    (tmp_path / "shared_battery.task_env.json").write_text(
        json.dumps({"task": b.id, "fingerprint": b.config_fingerprint()}))
    assert T._built_state(b)[0] == "BUILT"
    assert T._built_state(a)[0] == "UNSTAMPED"

    # a stamp over an empty directory is not a build
    (env / "items.jsonl").unlink()
    assert T._built_state(b)[0] == "MISSING"
