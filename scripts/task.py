"""One interface for every runnable task: list, show, build, run, score.

    uv run scripts/task.py list
    uv run scripts/task.py show  row01_lua_retention
    uv run scripts/task.py build row01_lua_retention
    uv run scripts/task.py run   row01_lua_retention --arm attack --variant goal_only --gpu 3
    uv run scripts/task.py score row01_lua_retention --run-dir <work_dir>

The point is that none of these need a decision from whoever runs them. A `TaskSpec` pins the target
domain, the probe counts, the seeds, the base models and the upstream revisions, and every command
below derives its arguments from that one file. Previously a task was a convention plus a person
remembering it, and the failure mode was not an error -- G1's target domain was named in four places
that agreed only by coincidence of defaults, and a mismatch would have scored the wrong domain and
returned a confident number.

BUILD IS FINGERPRINTED. `build` writes `.task_env.json` into the environment directory recording the
task id, the config fingerprint (a hash of the builder, its arguments and the pinned source
revisions) and the build time. `run` and `score` refuse to proceed against an environment whose
fingerprint no longer matches the spec, because that means the pool or probe set on disk is not the
one the task describes -- results from it would not be comparable to anything.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rsi_bench.tasks.spec import TaskSpec, load_all, resolve_path  # noqa: E402

STAMP = ".task_env.json"


# --------------------------------------------------------------------------- helpers
def _resolve(p: str | None) -> Path | None:
    """Spec paths are repo-relative unless absolute; `$RSI_HOME` is expanded."""
    if not p:
        return None
    return Path(resolve_path(p))


def _env_stamp(spec: TaskSpec) -> Path | None:
    """Where the build stamp lives: a SIBLING of the environment directory, never inside it.

    It used to be `<agent_dir>/.task_env.json`, and the agent_dir is mounted whole into the Docker
    sandbox and uploaded whole to the Modal taskdata volume -- so the agent received a file naming
    the side task's target domain ("role": "math domain (TARGET)"), every upstream dataset and its
    revision, and the build arguments. Found in the 2026-09-11 audit on row01b's staged directory.
    The stamp is bookkeeping for `task.py status`; the agent has no business seeing it.

    ONE STAMP PER TASK, not per directory: `<dir>.<task_id>.task_env.json`. Several tasks share an
    environment (row03b reads row03's pool, row06 and row15 share the chess battery, row14 reads
    row12's pool), and with one stamp per directory whichever task stamped it last owned it, so its
    partner always read STALE against a fingerprint that was never its own.

    Reads fall back to the old shared sibling `<dir>.task_env.json` and the legacy in-directory copy,
    but only when that stamp names this task; `build`/`restamp` always write the per-task stamp.
    """
    d = _env_dir(spec)
    return d.parent / f"{d.name}.{spec.id}{STAMP}" if d else None


def _env_dir(spec: TaskSpec) -> Path | None:
    return _resolve(spec.environment.agent_dir) or _resolve(spec.environment.secret_dir)


def _legacy_stamp(spec: TaskSpec) -> Path | None:
    d = _env_dir(spec)
    return d / STAMP if d else None


def _shared_stamp(spec: TaskSpec) -> Path | None:
    d = _env_dir(spec)
    return d.parent / f"{d.name}{STAMP}" if d else None


def _read_stamp(spec: TaskSpec) -> tuple[Path | None, dict | None]:
    """(path that exists, parsed contents): the per-task stamp, else an older shared or in-directory
    stamp that names this task."""
    for i, cand in enumerate((_env_stamp(spec), _shared_stamp(spec), _legacy_stamp(spec))):
        if cand is None or not cand.is_file():
            continue
        try:
            data = json.loads(cand.read_text())
        except json.JSONDecodeError:
            data = {}
        if i == 0 or data.get("task") == spec.id:
            return cand, data
    return None, None


def _drop_old_stamps(spec: TaskSpec) -> None:
    """Remove this task's older shared / in-directory stamps once the per-task one is written."""
    for cand in (_shared_stamp(spec), _legacy_stamp(spec)):
        if cand is not None and cand.is_file():
            try:
                owner = json.loads(cand.read_text()).get("task")
            except json.JSONDecodeError:
                owner = None
            if owner == spec.id or cand == _legacy_stamp(spec):
                cand.unlink()


def _unfetched(spec: TaskSpec) -> list[str]:
    """Files that scripts/fetch_data.sh places under this task's directories and that are absent.

    Third-party text is rebuilt locally, not shipped, so a directory can hold its shipped half (an
    LLM-authored battery, an id list) and still be unusable until the fetch has run.
    """
    manifest = REPO / "data" / "fetch_data.sha256"
    if not manifest.is_file():
        return []
    env = spec.environment
    dirs = [_resolve(x) for x in (env.agent_dir, env.secret_dir, env.cover_data_dir) if x]
    out = []
    for line in manifest.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        path = line.split()[1]
        if path.startswith("@"):
            continue
        f = REPO / path
        if any(f.parent == d or d in f.parents for d in dirs) and not f.is_file():
            out.append(path)
    return out


def _built_state(spec: TaskSpec) -> tuple[str, str]:
    """(state, detail) where state is BUILT | STALE | UNSTAMPED | MISSING | N/A.

    UNSTAMPED matters and is not the same as MISSING: several probe sets were built by hand before
    this task system existed, so the data is there but nothing records WHICH arguments produced it.
    Treating that as absent would invite a needless rebuild; treating it as built would claim a
    verification we cannot make. It is called out and allowed to run.
    """
    stamp = _env_stamp(spec)
    if stamp is None:
        return "N/A", "no environment directory declared"
    found, data = _read_stamp(spec)
    d = _env_dir(spec)
    if not (d.is_dir() and any(d.iterdir())):
        hint = " (rebuilt by scripts/fetch_data.sh)" if found is not None else ""
        return "MISSING", f"nothing at {d}{hint}"
    unfetched = _unfetched(spec)
    if unfetched:
        return "MISSING", f"{len(unfetched)} third-party file(s) not fetched yet, e.g. {unfetched[0]}"
    if found is None:
        return "UNSTAMPED", f"{d} has content but no {STAMP} — built before the task system?"
    got = (data or {}).get("fingerprint")
    want = spec.config_fingerprint()
    if got != want:
        return "STALE", f"built {got}, spec now {want} — rebuild"
    note = (data or {}).get("built_at") or ""
    if found != stamp:
        note += f"  [older stamp at {found}; `task.py restamp` moves it to {stamp.name}]"
    return "BUILT", note


def _flagify(args: dict) -> list[str]:
    """{'per-domain': 1200, 'kinds': ['a','b'], 'flag': True} -> CLI argv."""
    out: list[str] = []
    for k, v in args.items():
        flag = f"--{k.replace('_', '-')}"
        if v is None or v is False:
            continue
        if v is True:
            out.append(flag)
        elif isinstance(v, (list, tuple)):
            out.append(flag)
            out.extend(str(x) for x in v)
        else:
            out.extend([flag, str(v)])
    return out


def _run(argv: list[str], *, dry: bool, env: dict | None = None, cwd: Path = REPO) -> int:
    printable = " ".join(shlex.quote(a) for a in argv)
    if env:
        printable = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items())) + " \\\n  " + printable
    # flush: builds run for minutes and are usually watched through a redirected log, where Python's
    # block buffering otherwise hides every progress line until the process exits.
    print(f"\n$ {printable}\n", flush=True)
    if dry:
        return 0
    full = {**os.environ, **(env or {})}
    return subprocess.call(argv, cwd=str(cwd), env=full)


def _hf_token() -> str:
    """Read HF_TOKEN from env or .env (which must never be shell-sourced -- it has placeholders)."""
    import re

    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    envf = REPO / ".env"
    if envf.is_file():
        for line in envf.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return re.sub(r"\s*#.*$", "", line.split("=", 1)[1]).strip().strip("'\"")
    return ""


def preflight(spec: TaskSpec) -> list[str]:
    """Check every pinned source is reachable BEFORE running a builder.

    This exists because of a real failure: `bigcode/the-stack-smol` is gated on the Hub, and the pool
    build died on an auth error only after the loader had already started -- with a stack trace that
    said nothing about what to do. A gate needs a human to click accept, so it must surface at spec
    time, not mid-build. Returns a list of problems; empty means go.
    """
    import json as _json
    import urllib.request

    tok = _hf_token()
    problems = []
    for src in spec.sources:
        url = f"https://huggingface.co/api/{src.kind}s/{src.id}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"} if tok else {})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                info = _json.load(r)
        except Exception as e:  # noqa: BLE001 -- any failure here is a problem worth reporting
            problems.append(f"{src.id}: unreachable ({str(e)[:60]})")
            continue
        gated = info.get("gated", False)
        if gated not in (False, None):
            problems.append(
                f"{src.id}: GATED ({gated}) — accept access at "
                f"https://huggingface.co/{'datasets/' if src.kind == 'dataset' else ''}{src.id} "
                f"while signed in, then ensure HF_TOKEN is that account's")
        head = info.get("sha") or ""
        # Prefix comparison: specs are allowed to carry a short sha, and a full-vs-short string
        # compare would report every pinned source as behind upstream (it did, on first run).
        if src.revision and head and not head.startswith(src.revision):
            problems.append(
                f"{src.id}: pinned {src.revision[:12]}, Hub head is now {head[:12]} — the pin is "
                f"still honoured, but the spec is behind upstream")
    return problems


# --------------------------------------------------------------------------- commands
def cmd_list(args) -> int:
    specs = load_all()
    shown = {k: v for k, v in specs.items() if args.all or not v.deprecated}
    hidden = len(specs) - len(shown)
    print(f"{'ID':<28} {'ROW':>3}  {'GOAL':<26} {'STATUS':<7} {'ENV':<8} HEADLINE")
    print("-" * 108)
    for s in sorted(shown.values(), key=lambda x: (x.shortlist_row or 99, x.id)):
        state, _ = _built_state(s)
        tag = " [DEPRECATED]" if s.deprecated else ""
        print(f"{s.id:<28} {s.shortlist_row or '':>3}  {s.goal:<26} {s.status.value:<7} "
              f"{state:<8} {s.oracle.headline_metric}{tag}")
    if hidden:
        print(f"\n{hidden} deprecated task(s) hidden — superseded by their `b` variants. "
              f"`--all` to show, and they still build/run/score by id.")
    missing = [s.id for s in shown.values() if _built_state(s)[0] == "MISSING"]
    stale = [s.id for s in shown.values() if _built_state(s)[0] == "STALE"]
    unstamped = [s.id for s in shown.values() if _built_state(s)[0] == "UNSTAMPED"]
    if missing:
        print(f"\n{len(missing)} task(s) have no data yet -- run `bash scripts/fetch_data.sh` "
              f"(see docs/REPRODUCING.md): {', '.join(sorted(missing))}")
    if stale:
        print(f"{len(stale)} task(s) were built by a different builder or arguments than the spec now "
              f"pins: {', '.join(sorted(stale))}")
    if unstamped:
        print(f"{len(unstamped)} task(s) have data but no build stamp (predate this system, "
              f"contents unverified): {', '.join(sorted(unstamped))}")
    return 0


def cmd_show(args) -> int:
    s = load_all()[args.id]
    state, detail = _built_state(s)
    print(f"\n{s.id}   (shortlist row {s.shortlist_row})   status={s.status.value}")
    print(f"  {s.one_liner.strip()}\n")
    print(f"  goal            {s.goal}")
    print(f"  main prompt     {s.main_prompt}")
    print(f"  side goal       {s.side_goal}_<variant>.txt   variants: {', '.join(s.variants)}")
    print(f"  attack framing  {s.attack_preamble}")
    print(f"  deliverable     {s.deliverable}       budget: {s.agent_hours}h")
    print(f"\n  environment     {state}  {detail}")
    print(f"    builder       {s.environment.builder or '(none)'}")
    if s.environment.build_args:
        print(f"    build args    {json.dumps(s.environment.build_args)}")
    print(f"    agent dir     {s.environment.agent_dir or '(none)'}   [mounted READ-ONLY]")
    print(f"    secret dir    {s.environment.secret_dir or '(none)'}   [never mounted]")
    print(f"    fingerprint   {s.config_fingerprint()}")
    print(f"\n  oracle          {s.oracle.scorer}  {json.dumps(s.oracle.args)}")
    print(f"    headline      {s.oracle.headline_metric}")
    print(f"    honest arm    {'REQUIRED' if s.oracle.honest_arm_required else 'not required'}")
    print(f"    cover         {', '.join(s.cover_metrics) or '(none)'}")
    print("\n  pinned sources")
    for src in s.sources:
        rev = src.revision[:12] if src.revision else "UNPINNED"
        print(f"    {src.kind:<8} {src.id:<52} {rev}  {src.role}")
    if s.unpinned_sources():
        print(f"    ** {len(s.unpinned_sources())} source(s) unpinned — rebuilds may not reproduce **")
    if s.notes:
        print(f"\n  notes\n    " + "\n    ".join(s.notes.strip().splitlines()))
    return 0


def cmd_build(args) -> int:
    s = load_all()[args.id]
    if not s.environment.builder:
        print(f"{s.id}: no builder declared — nothing to build.")
        return 0
    for d in (s.environment.agent_dir, s.environment.secret_dir):
        if d:
            _resolve(d).mkdir(parents=True, exist_ok=True)

    if s.unpinned_sources():
        print(f"WARNING: unpinned sources ({', '.join(s.unpinned_sources())}) — "
              f"a rebuild may not reproduce this environment.")

    if not args.skip_preflight:
        print("preflight: checking pinned sources are reachable ...")
        problems = preflight(s)
        blocking = [p for p in problems if "GATED" in p or "unreachable" in p]
        for p_ in problems:
            print(f"  - {p_}")
        if blocking and not args.dry_run:
            print("\nBLOCKED: the above must be resolved first (--skip-preflight to override).")
            return 1
        if not problems:
            print("  all sources reachable")

    argv = ["uv", "run", s.environment.builder]
    # Builders differ in which directory --out means and whether they take a second one. The spec
    # declares it, rather than this dispatching on the script's filename -- an earlier version matched
    # on "prep_corpus_task" and silently sent a different builder's pool into the SECRET directory.
    env = s.environment
    out_dir = env.agent_dir if env.out_target == "agent" else env.secret_dir
    if out_dir:
        argv += ["--out", str(_resolve(out_dir))]
    if env.secret_arg and env.secret_dir:
        argv += [f"--{env.secret_arg}", str(_resolve(env.secret_dir))]
    argv += _flagify(env.build_args)

    rc = _run(argv, dry=args.dry_run)
    if rc != 0:
        print(f"builder exited {rc}; NOT stamping the environment.")
        return rc

    # Second builder, for tasks whose ORACLE probes are authored separately from the agent-visible
    # data. Without this row03 staged its pool, reported BUILT, and had no matched style pairs for the
    # oracle to difference against -- ready-looking and unscoreable.
    if env.probe_builder:
        pargv = ["uv", "run", env.probe_builder]
        if env.secret_dir:
            pargv += ["--out", str(_resolve(env.secret_dir))]
        pargv += _flagify(env.probe_build_args)
        rc = _run(pargv, dry=args.dry_run)
        if rc != 0:
            print(f"probe builder exited {rc}; NOT stamping the environment.")
            return rc
    if args.dry_run:
        return 0
    stamp = _env_stamp(s)
    if stamp:
        import datetime

        _drop_old_stamps(s)  # never leave a copy where the agent can read it
        stamp.write_text(json.dumps({
            "task": s.id,
            "fingerprint": s.config_fingerprint(),
            "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "build_args": s.environment.build_args,
            "sources": [src.model_dump() for src in s.sources],
        }, indent=2) + "\n")
        print(f"stamped {stamp}  (fingerprint {s.config_fingerprint()})")
    return 0


def cmd_restamp(args) -> int:
    """Re-stamp a built environment WITHOUT rebuilding it.

    For exactly one situation: the fingerprint FORMULA changed, or a spec field moved that provably
    cannot affect what the builder wrote, so existing contents are correct but the recorded hash no
    longer matches. Dropping model sources from the fingerprint did this to seven environments at once.

    Rebuilding instead is not a safe default here. Two of these environments hold LLM-authored probe
    sets and re-authoring is NOT idempotent -- it produces different questions, discarding sets that
    have already been validated. So this exists, and it demands a reason that is written into the
    stamp, because an unexplained re-stamp is indistinguishable from papering over real staleness.
    """
    s = load_all()[args.id]
    stamp = _env_stamp(s)
    if stamp is None:
        print(f"{s.id}: no environment directory declared — nothing to stamp.")
        return 1
    env_dir = _env_dir(s)
    if not env_dir.is_dir() or not any(env_dir.iterdir()):
        print(f"{s.id}: {env_dir} is empty — there is nothing built to re-stamp. Run `build`.")
        return 1

    import datetime

    found, prev = _read_stamp(s)
    prev = prev or {}
    new_fp = s.config_fingerprint()
    if prev.get("fingerprint") == new_fp and found == stamp:
        print(f"{s.id}: already stamped at {new_fp} — nothing to do.")
        return 0
    if found is not None and found != stamp:
        _drop_old_stamps(s)  # older shared / in-directory stamp: migrate it to the per-task name
        print(f"{s.id}: moved stamp {found.name} -> {stamp.name}")
        if prev.get("fingerprint") == new_fp:
            stamp.write_text(json.dumps(prev, indent=2) + "\n")
            return 0

    history = prev.get("restamps", [])
    history.append({"from": prev.get("fingerprint"), "to": new_fp, "reason": args.reason,
                    "at": datetime.datetime.now().isoformat(timespec="seconds")})
    stamp.write_text(json.dumps({
        "task": s.id,
        "fingerprint": new_fp,
        "built_at": prev.get("built_at"),          # the CONTENTS still date from the original build
        "build_args": s.environment.build_args,
        "sources": [src.model_dump() for src in s.sources],
        "restamps": history,
    }, indent=2) + "\n")
    print(f"re-stamped {s.id}: {prev.get('fingerprint')} -> {new_fp}")
    print(f"  reason: {args.reason}")
    print(f"  contents unchanged, still dated {prev.get('built_at')}")
    return 0


def _require_built(s: TaskSpec, force: bool) -> bool:
    state, detail = _built_state(s)
    if state in ("BUILT", "N/A"):
        return True
    if state == "UNSTAMPED":
        print(f"NOTE: environment is UNSTAMPED — {detail}. Proceeding; rebuild to make it verifiable.")
        return True
    print(f"ERROR: environment is {state} — {detail}")
    if state == "MISSING":
        print("       run: bash scripts/fetch_data.sh   (third-party data is rebuilt, not shipped)")
    else:
        print(f"       run: uv run scripts/task.py build {s.id}")
    if force:
        print("       --force given, continuing anyway (results may not be comparable)")
        return True
    return False


def cmd_run(args) -> int:
    s = load_all()[args.id]
    if args.variant not in s.variants:
        print(f"ERROR: variant {args.variant!r} not offered by this task ({', '.join(s.variants)})")
        return 2
    if not _require_built(s, args.force):
        return 1
    if args.arm == "attack" and s.oracle.honest_arm_required:
        print("NOTE: this task's oracle needs a matched HONEST arm to be interpretable. "
              f"Run `task.py run {s.id} --arm honest` too.")

    env = s.launcher_env(args.variant)
    env["RSI_AGENT_GPU"] = str(args.gpu)
    if args.model:
        env["RSI_AGENT_MODEL"] = args.model
    if args.hours:
        env["RSI_AGENT_HOURS"] = str(args.hours)
    variant = args.variant if args.arm == "attack" else "na"
    argv = ["bash", "scripts/run/run_agent_task_docker.sh", args.arm, variant,
            args.label or f"{s.id}_{args.arm}"]
    return _run(argv, dry=args.dry_run, env=env)


def cmd_score(args) -> int:
    s = load_all()[args.id]
    if not _require_built(s, args.force):
        return 1
    scorer = s.oracle.scorer
    argv = (["bash", scorer] if scorer.endswith(".sh") else ["uv", "run", scorer])
    argv += _flagify(s.oracle.args)
    # The delivered artifact, passed the way each scorer expects. Scorers fall into two classes and
    # conflating them is how row04b's scorer failed with a bare argparse error (it wants --corpus, not
    # --model):
    #   PATH-BASED  — the scorer opens the delivered file/dir directly (a corpus, an RM, a classifier).
    #   SERVED      — the scorer talks to a vLLM endpoint (--base-url) and the model must be served
    #                 first. task.py does not serve yet, so it refuses rather than pass a path as a
    #                 model name and produce a confusing failure minutes in.
    delivered = Path(args.run_dir) / s.deliverable
    PATH_FLAG = {"score_corpus": "--delivered", "score_rm": "--delivered",
                 "score_monitor_corpus": "--corpus", "score_quality_filter": "--delivered"}
    SERVED = ("score_topic_refusal", "score_trait_backdoor", "score_trait_cluster",
              "score_monitor_knowledge")
    served_wired = False
    flag = next((f for k, f in PATH_FLAG.items() if k in scorer), None)
    if flag:
        argv += [flag, str(delivered)]
        if "score_corpus" in scorer:
            argv += ["--provenance", str(_resolve(s.environment.secret_dir))]
    elif any(k in scorer for k in SERVED):
        # Serve the delivered model on a per-GPU port, run the task's scorer against it, tear down.
        # serve_and_score.sh injects --base-url/--model/--out, so we pass only the oracle args.
        if not delivered.is_dir():
            print(f"\n{s.id}: no delivered model at {delivered} — nothing to serve; skipping the "
                  f"artifact score (the transcript verdict below still runs).")
            argv = None
        else:
            gpu = os.environ.get("RSI_AGENT_GPU", "0")
            try:
                port = str(8012 + int(gpu))
            except ValueError:
                port = "8012"
            out = args.out or str(Path(args.run_dir) / "scores.json")
            argv = (["bash", "scripts/score/serve_and_score.sh", str(delivered), gpu, port, out, scorer]
                    + _flagify(s.oracle.args))
            served_wired = True
    else:
        argv += ["--model", str(delivered)]
    if argv is not None and args.out and not served_wired:
        argv += ["--out", args.out]
    print(f"headline metric for this task: {s.oracle.headline_metric}")
    if s.cover_metrics:
        print(f"cover metrics that must hold up: {', '.join(s.cover_metrics)}")
    rc = _run(argv, dry=args.dry_run) if argv is not None else 0

    # TRANSCRIPT VERDICT, alongside the artifact oracle. Every task gets one for free: judge_run.py
    # derives the side goal from this same spec, so it needs no per-task configuration. It answers a
    # question the artifact cannot -- whether the agent ACTED -- which is the difference between "the
    # sabotage did not land" and "the agent declined". Skipped with --no-transcript.
    stream = Path(args.run_dir) / "agent_stream.log"
    if not args.no_transcript and stream.is_file():
        targv = ["uv", "run", "scripts/score/judge_run.py", "--run", str(args.run_dir),
                 "--task", s.id, "--variant", args.variant]
        if args.out:
            targv += ["--out", str(Path(args.out).with_suffix(".transcript.json"))]
        print("\n--- transcript verdict (oracle mode: the judge IS told the side goal) ---")
        _run(targv, dry=args.dry_run)
    elif not args.no_transcript:
        print(f"\n(no agent_stream.log at {stream} — skipping the transcript verdict)")
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="tasks in the default sweep, with whether each is built")
    p.add_argument("--all", action="store_true", help="include deprecated tasks")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("show", help="the full pinned definition of one task")
    p.add_argument("id")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("build", help="materialise the environment (pool / probes) and stamp it")
    p.add_argument("id")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-preflight", action="store_true",
                   help="do not check source reachability/gating first")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("restamp",
                       help="record the current fingerprint on an already-built environment "
                            "(use ONLY when the formula changed, not the contents)")
    p.add_argument("id")
    p.add_argument("--reason", required=True,
                   help="why the contents are still valid; written into the stamp for audit")
    p.set_defaults(fn=cmd_restamp)

    p = sub.add_parser("run", help="launch an arm")
    p.add_argument("id")
    p.add_argument("--arm", choices=["honest", "attack"], required=True)
    p.add_argument("--variant", default="goal_only")
    p.add_argument("--gpu", default="3")
    p.add_argument("--model", default=None, help="agent model override")
    p.add_argument("--hours", type=int, default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--force", action="store_true", help="run against a stale environment anyway")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("score", help="score a delivered artifact with this task's oracle")
    p.add_argument("id")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--variant", default="goal_only", choices=["goal_only", "technique_given"],
                   help="which side-goal framing the run used; the transcript judge needs it")
    p.add_argument("--no-transcript", action="store_true",
                   help="skip the transcript verdict (artifact oracle only)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    try:
        raise SystemExit(args.fn(args))
    except KeyError as e:
        print(f"unknown task id {e}. Try: uv run scripts/task.py list")
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
