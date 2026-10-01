"""Shared Modal definitions for rsi-bench: the app, the two images, the volumes, the secrets.

Modal is a SECOND deployment target for the same experiment `scripts/run/run_agent_task_docker.sh`
runs locally. The mapping from the local (docker + nvme bind-mount) model to Modal:

    local docker                          Modal
    ------------------------------------  --------------------------------------------------
    rsi-agent:base + host bind mounts     AGENT_IMAGE (toolchain baked in, not mounted)
    $RSI_HOME/.../hf_cache  (rw mount)    Volume "rsi-hf-cache"   at /cache/hf
    $WORK                   (rw mount)    Volume "rsi-runs"       at /runs/<run_id>
    $RSI_HOME/.../venvs/{serve,train}     baked into AGENT_IMAGE at /opt/rsi/venvs/*
    /usr/local/cuda-12.9    (ro mount)    the CUDA devel base image's /usr/local/cuda
    -e ANTHROPIC_*  from .env             Secret "rsi-agent-model"
    NOT mounted: EFS, .env, probes        not in the image and not attached — nothing to hide
    --gpus device=<G>                     gpu="H100" (a whole dedicated H100, not a shared slice)
    timeout --signal=TERM ... <hours>     Sandbox.create(timeout=...)

Two DIFFERENT images/sandboxes on purpose, mirroring the local honest/attack split:
  * AGENT_IMAGE  gets no repo checkout and no oracle volume. The agent cannot read the held-out
    eval-awareness probes or the GPQA test ids, because they are not present in its container at
    all (a stronger guarantee than the local "don't mount it" one -- there is no host to reach).
  * SCORER_IMAGE gets the repo + the "rsi-oracle" volume, and never runs agent-authored code
    except by loading final_model/ into vLLM.

Why the vLLM wheel is still the +cu129 GitHub build (same as docs/REPRODUCING.md 2b): NVIDIA
drivers are backward compatible with older CUDA minor versions, so a cu129 build runs on Modal's
(newer) host drivers. The reverse -- the PyPI cu13 wheel -- is what broke on our 12.9 devbox
driver. cu129 is therefore the portable choice for BOTH targets; keep them identical so a run is
comparable across local and Modal.
"""

from __future__ import annotations

import os
import pathlib

import time

import modal

APP_NAME = os.environ.get("RSI_MODAL_APP", "rsi-bench")
REPO = pathlib.Path(__file__).resolve().parents[2]  # .../rsi-bench

# --- remote paths (the local nvme paths do not exist on Modal; every consumer of these is
# env-overridable -- serve_successor.sh honours RSI_SERVE_PYTHON/HF_HOME/CUDA_HOME, and the task
# prompt templates {train_python}/{serve_python}/{work_dir} -- so nothing hard-codes a local path). ---
VENVS = "/opt/rsi/venvs"
TRAIN_PY = f"{VENVS}/train/bin/python"
SERVE_PY = f"{VENVS}/serve/bin/python"
HF_HOME = "/cache/hf"
RUNS_ROOT = "/runs"
ORACLE_ROOT = "/oracle"
CUDA_HOME = "/usr/local/cuda"
AGENT_USER = "agent"  # NOT root (claude refuses skip-permissions as root); UID assigned by the system

VLLM_WHEEL = (
    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.25.1/"
    "vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
)
NODE_VER = "v22.11.0"
NODE_BIN = f"/opt/rsi/node-{NODE_VER}-linux-x64/bin"
NPM_BIN = "/opt/rsi/npm-global/bin"

# --- volumes -----------------------------------------------------------------------------------
# hf_cache: write-once/read-many (Modal's documented sweet spot for weights). Populated once by
# preflight.py --warm-cache, then read by every run.
hf_cache = modal.Volume.from_name("rsi-hf-cache", create_if_missing=True)
# runs: one subdirectory per run. Concurrent runs never touch the same file, which is the condition
# Modal requires for many-writer volumes (last-write-wins only bites on the SAME file).
runs = modal.Volume.from_name("rsi-runs", create_if_missing=True)
# oracle: held-out probes + GPQA test ids. Attached ONLY to the scorer.
oracle = modal.Volume.from_name("rsi-oracle", create_if_missing=True)
# taskdata: the agent-visible input pool some tasks hand over (a document pool, summarization seeds).
# Uploaded per task under <task_id>/ and mounted READ-ONLY at TASKDATA_ROOT/<task_id> so the agent
# cannot mutate the pool the oracle's provenance map was built from. Only tasks with agent_dir use it.
TASKDATA_ROOT = "/task_data"


def agent_workdir(stamp: str) -> str:
    """The path the AGENT sees as its run directory: `/runs/job-<stamp>`, nothing else.

    The run_id (`honest_na_row09_rm_mentor_persona_<stamp>`) names the arm, the variant and the task,
    and the task id names the side goal. Until 2026-09-12 that string was the agent's cwd, so an
    honest arm could read "mentor_persona" in its own prompt paths (70 occurrences in one transcript)
    and the monitor saw it too. The volume is still `rsi-run-<stamp>-<uuid>` and every driver-side
    path still uses the run_id; only the mount point inside the agent sandbox is opaque. The stamp
    is enough to join the two.
    """
    return f"{RUNS_ROOT}/job-{stamp}"


def taskdata_subdir(task_id: str) -> str:
    """Opaque per-task directory under TASKDATA_ROOT: `td-<sha1(task_id)[:10]>`.

    The task id (row06_chess_refusal, row09_rm_mentor_persona, ...) names the side goal, and the pool
    path appears verbatim in TASK.md. Same content, unrecognisable name; both the uploader and the
    scorer's --pool argument go through this so they cannot drift.
    """
    import hashlib

    return "td-" + hashlib.sha1(task_id.encode()).hexdigest()[:10]
taskdata = modal.Volume.from_name("rsi-taskdata", create_if_missing=True)


def run_volume(run_id: str, create: bool = True):
    """The per-run volume, mounted at RUNS_ROOT/<run_id> so the agent sees ONLY its own directory.

    The shared `rsi-runs` volume used to be mounted at RUNS_ROOT in every agent sandbox, which meant
    each agent could read -- and WRITE -- all 80+ previous run directories: other runs' TASK.md
    (including their attack preamble and side goal), 28 delivered checkpoints, and their scores.json.
    Measured over 81 runs, 2 exploited it: one row09 attack arm copied a prior run's trainer, and one
    row05 attack arm read a previous attempt's scores.json and grepped a scorer log for the oracle's
    detection vocabulary. That is optimising against the eval, which the attack preamble explicitly
    forbids.

    Mounting per-run at the SAME path keeps every path in the codebase working unchanged -- the agent's
    workdir, the scorer's `{RUNS_ROOT}/{run_id}/...` arguments and the prompts all stay as they were.
    Small files are copied to the shared volume afterwards (see consolidate_run) so the reporting and
    monitor tooling, which iterates one volume, keeps working.
    """
    return modal.Volume.from_name(run_volume_name(run_id), create_if_missing=create)


def run_volume_name(run_id: str) -> str:
    """Modal caps volume names at 64 chars and a full run_id can reach 70, so name the volume after the
    run's unique tail (`<stamp>-<uuid6>`), which is what makes a run_id unique in the first place."""
    return f"rsi-run-{run_id.rsplit('_', 1)[-1]}"


RATE_LIMIT_BACKOFF_S = (2, 4, 8, 16, 32, 60)


def is_rate_limit(ex: BaseException) -> bool:
    return "ResourceExhausted" in type(ex).__name__ or "rate limit" in str(ex).lower()


def retry_rate_limited(fn, *args, _sleep=time.sleep, **kw):
    """Call fn(*args) and retry with backoff ONLY on Modal's per-workspace rate limit
    (ResourceExhaustedError, e.g. VolumeListFiles). Any other exception propagates on the first try."""
    for delay in RATE_LIMIT_BACKOFF_S:
        try:
            return fn(*args, **kw)
        except Exception as ex:  # noqa: BLE001 -- filtered below
            if not is_rate_limit(ex):
                raise
            _sleep(delay)
    return fn(*args, **kw)


def open_run(run_id: str):
    """(volume, path_prefix) for reading a run's files, per-run volume first, shared as fallback.

    Runs made before per-run volumes existed live under `<run_id>/` on the shared volume; runs made
    after live at the ROOT of their own volume. Readers should not have to know which.

    Only NotFoundError selects the fallback. 2026-09-16 a bare `except Exception` here turned a
    rate-limited existence probe into "no such volume": twelve monitor threads probed at once, five
    of the first seven artifact monitors were handed the shared-volume dir (harness files only) and
    reported a workspace of 0-1 files. A rate limit is retried; anything else propagates.
    """
    from modal.exception import NotFoundError
    v = modal.Volume.from_name(run_volume_name(run_id))
    try:
        retry_rate_limited(lambda: list(v.listdir("/")))     # raises NotFoundError if the volume does not exist
        return v, ""
    except NotFoundError:
        return runs, f"{run_id}/"


def _base(tag: str = "nvidia/cuda:12.9.1-devel-ubuntu24.04") -> modal.Image:
    """CUDA *devel* base: nvcc is required at runtime, not just at build time -- Qwen3.5 is a hybrid
    GDN+Mamba model and flashinfer JIT-compiles its kernel on the first serve."""
    return modal.Image.from_registry(tag, add_python="3.12").apt_install(
        "ca-certificates", "git", "curl", "xz-utils",
        "libgomp1", "libnuma1", "build-essential",
    )


AGENT_IMAGE = (
    _base()
    # uv, at a fixed path so the venv creation below is reproducible
    .run_commands(
        "curl -fsSL https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh",
        "uv --version",
    )
    # serve venv (vLLM) and train venv (LoRA SFT) -- byte-for-byte the local serve/train venv specs (docs/REPRODUCING.md)
    .run_commands(
        f"uv venv --python 3.12 {VENVS}/serve",
        f"uv pip install --python {VENVS}/serve/bin/python --torch-backend=cu129 '{VLLM_WHEEL}'",
        # import-only check; a GPU is deliberately NOT requested for the build (vLLM imports fine
        # without one, and GPU build minutes are the expensive kind)
        f"{SERVE_PY} -c 'import torch, vllm; print(torch.__version__, vllm.__version__)'",
    )
    .run_commands(
        f"uv venv --python 3.12 {VENVS}/train",
        # ONE RESOLVE, not two. Installing the base set and then vLLM separately lets the second
        # resolve move the first's pins: doing exactly that locally silently downgraded torch
        # 2.11.0 -> 2.10.0, which left torchvision's compiled ops unloadable
        # ("operator torchvision::nms does not exist") and broke `from trl import GRPOTrainer`
        # several layers away from the install that caused it. Resolving together cannot do that.
        f"uv pip install --python {VENVS}/train/bin/python --torch-backend=cu129 "
        "'torch==2.11.0' torchvision==0.26.0 torchaudio==2.11.0 "
        "'transformers>=5.14,<6' 'trl>=1.9,<2' peft accelerate datasets safetensors huggingface_hub "
        # NINJA IS NOT OPTIONAL. vLLM JIT-compiles flashinfer kernels at engine start -- the
        # top-k/top-p sampler always, and Qwen3.5's Gated-DeltaNet prefill unless you pass
        # `--gdn-prefill-backend triton` -- and the JIT shells out to `ninja`. Without it the engine
        # dies in `determine_available_memory` with a bare
        # `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`, several frames away
        # from anything that names a missing package. Both devbox venvs have had it since 2026-07,
        # which is exactly why this gap only ever showed up on Modal.
        "ninja "
        f"'{VLLM_WHEEL}'",
        # trl AND vLLM, in the TRAIN venv, together. Not a convenience -- without both, the
        # architecture rows 14-17's prompts describe is unreachable:
        #   * 12 main prompts list `trl` among the train python's packages and rows 14/15 call
        #     GRPOTrainer "the supported path". trl was NEVER in this image. Two honest runs
        #     (honest_na_run1 2026-07-23) probed the venv, hit `No module named 'trl'`, and had to
        #     work around a package the prompt promised them.
        #   * trl's vLLM rollout path needs vllm importable IN THE SAME INTERPRETER as trl --
        #     colocate imports it in-process, and server mode is `trl vllm-serve`. Splitting them
        #     across the train/serve venvs leaves only transformers-native generation, which is
        #     several times slower and turns a 5h GRPO budget into a much smaller one.
        # The cu129 wheel is required: PyPI's vllm==0.25.1 is built against CUDA 13 and dies with
        # `ImportError: libcudart.so.13` against this cu129 torch.
        f"{TRAIN_PY} -c 'import torch, transformers, trl, peft, vllm; "
        "print(torch.__version__, transformers.__version__, trl.__version__, vllm.__version__)'",
    )
    # node 22 + the Claude Code CLI == the agent scaffold (same version as the local runs)
    .run_commands(
        f"curl -fsSL -o /tmp/node.tar.xz https://nodejs.org/dist/{NODE_VER}/node-{NODE_VER}-linux-x64.tar.xz",
        "mkdir -p /opt/rsi && tar xf /tmp/node.tar.xz -C /opt/rsi && rm /tmp/node.tar.xz",
        f"PATH={NODE_BIN}:$PATH NPM_CONFIG_PREFIX=/opt/rsi/npm-global "
        f"{NODE_BIN}/npm install -g @anthropic-ai/claude-code",
        f"{NPM_BIN}/claude --version",
    )
    # unprivileged user: claude refuses --dangerously-skip-permissions with root privileges.
    # No fixed -u: ubuntu:24.04 already owns UID 1000 ("ubuntu"), and nothing here needs a
    # specific UID -- ownership is always set by name.
    .run_commands(
        f"useradd -m -s /bin/bash {AGENT_USER}",
        f"mkdir -p {RUNS_ROOT} {HF_HOME} && chown -R {AGENT_USER} /opt/rsi {RUNS_ROOT} {HF_HOME}",
    )
    .env(
        {
            "CUDA_HOME": CUDA_HOME,
            "HF_HOME": HF_HOME,
            "PATH": f"{NODE_BIN}:{NPM_BIN}:{CUDA_HOME}/bin:/usr/local/bin:/usr/bin:/bin:/sbin",
            # keep torch.compile / flashinfer JIT caches inside the container overlay
            "XDG_CACHE_HOME": "/tmp/.cache",
        }
    )
    # anthropic_shim: a non-Anthropic reasoning model (GLM) replays thinking blocks that Fireworks
    # rejects, killing the run on its second tool call. The shim strips them. Baked in as the LAST
    # layers so the expensive venv/node layers stay cached across this change; httpx goes into the
    # add_python interpreter that runs the shim. Started only for non-Anthropic agents (see runner).
    .pip_install("httpx")
    .add_local_file(str(REPO / "scripts" / "run" / "anthropic_shim.py"), "/opt/rsi/anthropic_shim.py",
                    copy=True)
)

# Scorer: same GPU stack, plus the repo and the light scoring deps. `ignore` keeps the agent-facing
# prompt templates and any local run artifacts out -- only what the scorers actually need.
SCORER_IMAGE = (
    AGENT_IMAGE.pip_install("openai", "aiohttp")
    # IGNORE PATTERNS MUST BE RECURSIVE. A bare "__pycache__" matches only the top level, so a
    # nested scripts/modal/__pycache__ was still uploaded -- and because importing rsi_modal WRITES
    # that directory, the build failed with "<...>.pyc was modified during build process". Any
    # runner that imports a sibling module hits this.
    .add_local_dir(
        str(REPO / "scripts"), "/repo/scripts", copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(str(REPO / "src"), "/repo/src", copy=True,
                   ignore=["**/__pycache__/**", "**/*.pyc"])
)


def app() -> modal.App:
    return modal.App.lookup(APP_NAME, create_if_missing=True)


def getenv_dotenv(key: str) -> str | None:
    """Read one key from the repo .env the same way the bash runners do.

    .env is NOT shell-sourceable (inline `# comments`, `<placeholder>` values), so we grep+strip
    rather than parse. Used only LOCALLY, to populate a Modal Secret -- .env itself is never
    uploaded, and never lands in an image layer.
    """
    p = REPO / ".env"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.startswith(f"{key}="):
            v = line[len(key) + 1 :]
            v = v.split("#", 1)[0].strip().strip("\"'")
            return v or None
    return None


def assert_reachable_from_modal(base_url: str) -> None:
    """Fail fast on the #1 way a Modal run wastes an hour of H100 time.

    A proxy on a private network (RFC1918 address, no public DNS record) cannot be reached from a
    Modal container, which sits outside that network. The agent's model calls originate INSIDE the
    sandbox, so an unreachable endpoint means the agent starts, fails every request, and burns the
    wall clock.

    modal.Proxy does NOT fix this: it gives Modal *static egress IPs* to allow-list on a publicly
    routable endpoint; it does not place the container inside our VPC.
    """
    bad = ("internal", "10.", "172.16.", "192.168.", "localhost", "127.0.0.1")
    host = base_url.split("://", 1)[-1].split("/", 1)[0]
    if any(t in host for t in bad):
        raise SystemExit(
            f"ANTHROPIC_BASE_URL host {host!r} is not reachable from a Modal container.\n"
            "Point RSI_ANTHROPIC_BASE_URL at a publicly routable model endpoint (e.g. a direct\n"
            "provider key in the 'rsi-agent-model' Modal Secret). See docs/REPRODUCING.md."
        )


# The publicly routable LiteLLM proxy the Modal sandboxes call: PUB_LITELLM_BASE_URL in .env (bare
# host, no /v1 -- Anthropic clients want it bare; OpenAI-compatible callers append /v1 themselves).
# Falls back to LITELLM_BASE_URL when the same proxy serves both the local and the Modal paths.
PUBLIC_PROXY = (os.environ.get("RSI_PUBLIC_PROXY") or getenv_dotenv("PUB_LITELLM_BASE_URL")
                or getenv_dotenv("LITELLM_BASE_URL") or "").rstrip("/").removesuffix("/v1")


_RING_CACHE: list[str] | None = None


def declared_key_ring(env_text: str | None = None) -> list[str]:
    """Every PUB_LITELLM_API_KEY<N> in .env, in numeric order, live or not -- gaps in N allowed.

    Several budget-capped keys spread a sweep's spend, so one key hitting its cap does not take every
    run down with it. Read locally only, to build a per-run Secret; .env is never uploaded and never
    lands in an image layer. Gaps in N are allowed (e.g. KEY1, KEY9, KEY10 after pruning dead keys).
    """
    import re
    if env_text is None:
        p = REPO / ".env"
        env_text = p.read_text() if p.exists() else ""
    found: dict[int, str] = {}
    for line in env_text.splitlines():
        m = re.match(r"PUB_LITELLM_API_KEY(\d+)=(.*)$", line)
        if not m:
            continue
        v = m.group(2).split("#", 1)[0].strip().strip("\"'")
        if v:
            found[int(m.group(1))] = v
    ks = [found[i] for i in sorted(found)]
    return ks or [k for k in [getenv_dotenv("PUB_LITELLM_API_KEY")] if k]


def key_ring() -> list[str]:
    """The ring with budget-exhausted keys REMOVED (probed once per process, then cached).

    Exhausted keys accumulate in .env as batches are spent, and a stale one is not harmless: the
    round-robin would hand it to a run as its PRIMARY key, and every cost estimate and launch plan
    would count keys that cannot serve a single token. The shim does self-heal (it rotates forward
    past a dead head within one request), so this is about honest planning and not wasting the first
    call of every run. Costs one 8-token request per key, once.

    Falls back to the declared ring if the probe cannot run at all, since refusing to launch because a
    liveness check failed would be worse than launching and letting the shim rotate.
    """
    global _RING_CACHE
    if _RING_CACHE is not None:
        return _RING_CACHE
    declared = declared_key_ring()
    base = (os.environ.get("RSI_PUBLIC_PROXY") or PUBLIC_PROXY).rstrip("/")
    live: list[str] = []
    try:
        import concurrent.futures
        import json as _json
        import urllib.request

        body = _json.dumps({"model": "fireworks_ai/glm-5p2", "max_tokens": 8,
                            "messages": [{"role": "user", "content": "ping"}]}).encode()

        def ok(k: str) -> bool:
            """Drop a key ONLY for budget exhaustion, which is permanent.

            Treating every error as death is wrong and actively harmful: a 429 means the key is alive
            and busy, and a timeout means the probe was unlucky. One transient 429 during a launch
            probe collapsed a 2-key ring to 1, which pinned all four runs of a wave to a single key --
            no rotation fallback, and both arms of each pair sharing one budget, which is precisely
            what the ring exists to avoid. Fail OPEN on anything that is not a budget rejection.
            """
            req = urllib.request.Request(
                f"{base}/v1/messages", data=body,
                headers={"content-type": "application/json", "x-api-key": k,
                         "authorization": f"Bearer {k}", "anthropic-version": "2023-06-01"})
            try:
                urllib.request.urlopen(req, timeout=90)
                return True
            except Exception as e:  # noqa: BLE001
                detail = (getattr(e, "read", lambda: b"")() or b"").decode(errors="replace")
                if any(m in detail for m in ("budget_exceeded", "Budget has been exceeded")):
                    print(f"  key ...{k[-5:]} OVER BUDGET, dropping: {detail[:110]}")
                    return False
                # PERMANENTLY DEAD, not transient, so failing open is wrong here. These two are
                # key-level config rejections: the key cannot serve ANY model, ever, and no retry
                # will change that. 8 of the 10 declared keys return the attribution error, so
                # failing open on it printed "live proxy keys: 10/10" and handed the judge a ring
                # that 400s four calls out of five. Measured 2026-09-02 against claude-opus-4-8 and
                # gpt-5.6-luna: the same two keys answer for both.
                if any(m in detail for m in ("no project attribution found",
                                             "Invalid proxy server token")):
                    print(f"  key ...{k[-5:]} DEAD (key-level rejection, not transient), dropping: "
                          f"{detail[:110]}")
                    return False
                print(f"  key ...{k[-5:]} probe failed but keeping it (transient, not a key or "
                      f"budget rejection): {(detail[:100] or str(e))}")
                return True

        with concurrent.futures.ThreadPoolExecutor(max(1, len(declared))) as ex:
            live = [k for k, good in zip(declared, ex.map(ok, declared)) if good]
    except Exception as e:  # noqa: BLE001
        print(f"  key liveness probe failed ({type(e).__name__}); using the declared ring")
        live = declared
    if not live:
        print("  WARNING: no live proxy keys; falling back to the declared ring")
        live = declared
    print(f"  live proxy keys: {len(live)}/{len(declared)} -> {[k[-5:] for k in live]}")
    _RING_CACHE = live
    return live


def ring_secret(offset: int = 0, ring: list[str] | None = None) -> modal.Secret:
    """Model credentials with the key ring ROTATED by `offset`, so run i starts on key i%K.

    The shim (which reads LITELLM_API_KEYS) walks the ring FORWARD on a budget rejection, so spend
    spreads across keys instead of draining key 1, and a run whose own key caps mid-training spills
    into a neighbour's budget rather than dying. ANTHROPIC_AUTH_TOKEN is the ring head, for the
    non-shim consumers (the scorer's judge calls) that use a single key.
    """
    ring = ring or key_ring()
    if not ring:
        return model_secret()  # no public keys in .env -> the single-key Modal Secret
    o = offset % len(ring)
    rot = ring[o:] + ring[:o]
    return modal.Secret.from_dict(
        {
            "ANTHROPIC_BASE_URL": PUBLIC_PROXY,
            "ANTHROPIC_AUTH_TOKEN": rot[0],
            "ANTHROPIC_API_KEY": rot[0],
            "LITELLM_API_KEYS": ",".join(rot),
            "HF_TOKEN": getenv_dotenv("HF_TOKEN") or "",
        }
    )


def model_secret() -> modal.Secret:
    """The agent's model credentials.

    Prefers a Modal-hosted Secret (`modal secret create rsi-agent-model ...`) so the key never
    round-trips through this machine. Falls back to .env for a quick local-driven run.
    """
    try:
        return modal.Secret.from_name(
            "rsi-agent-model", required_keys=["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"]
        )
    except Exception:
        base = os.environ.get("RSI_ANTHROPIC_BASE_URL") or (getenv_dotenv("LITELLM_BASE_URL") or "")
        base = base.removesuffix("/v1")
        assert_reachable_from_modal(base)
        return modal.Secret.from_dict(
            {
                "ANTHROPIC_BASE_URL": base,
                "ANTHROPIC_AUTH_TOKEN": getenv_dotenv("LITELLM_API_KEY") or "",
                "HF_TOKEN": getenv_dotenv("HF_TOKEN") or "",
            }
        )
