#!/usr/bin/env python
"""Empirically answer every open question about running rsi-bench on Modal, cheaply, before
committing an H100-hour to a real run.

    python scripts/modal/preflight.py                 # stage 1 only: network (CPU, ~$0.01, ~1 min)
    python scripts/modal/preflight.py --gpu           # + stage 2: GPU/toolchain (builds the image)
    python scripts/modal/preflight.py --warm-cache    # + stage 3: populate the HF weights volume

Stage 1 (CPU, no image build) answers the question that decides the whole approach: can a Modal
container reach the model endpoint the agent needs? A proxy on a private network (RFC1918 address,
no public DNS record) is unreachable from Modal -- stage 1 checks that from inside Modal rather than
by inference, and also checks whether the fallback (a public provider endpoint) is reachable.

Stage 2 (H100) checks the things that silently break a 4-hour run:
  * host driver version vs our +cu129 torch/vLLM build
  * nvcc present at runtime (flashinfer JIT-compiles the Qwen3.5 GDN kernel on first serve)
  * `claude --version` works, and refuses --dangerously-skip-permissions as root but accepts it as
    the unprivileged `agent` user (the reason AGENT_IMAGE creates one)
  * volume write + visibility

Nothing here mutates local state. Requires `modal setup` to have been run once.
"""

from __future__ import annotations

import argparse
import sys

import modal

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import rsi_modal as R  # noqa: E402

PROBE_IMAGE = modal.Image.debian_slim().apt_install("curl", "dnsutils")


def _run(sb: modal.Sandbox, cmd: str, timeout: int = 120) -> tuple[int, str]:
    p = sb.exec("bash", "-lc", cmd, timeout=timeout)
    out = p.stdout.read() + p.stderr.read()
    p.wait()
    return p.returncode, out.strip()


def stage1_network(app: modal.App) -> None:
    """Reachability of each candidate model endpoint FROM INSIDE a Modal container."""
    print("=" * 78)
    print("STAGE 1 -- model-endpoint reachability from inside Modal (CPU sandbox)")
    print("=" * 78)

    internal = (R.getenv_dotenv("LITELLM_BASE_URL") or "").removesuffix("/v1")
    targets = [t for t in [internal, "https://api.anthropic.com", "https://api.fireworks.ai"] if t]

    sb = modal.Sandbox.create(
        app=app, image=PROBE_IMAGE, timeout=300,
        tags={"rsi": "preflight", "stage": "network"},
    )
    try:
        for url in targets:
            host = url.split("://", 1)[-1].split("/", 1)[0]
            rc_dns, dns = _run(sb, f"getent hosts {host} || echo NO_DNS", timeout=30)
            rc, code = _run(
                sb,
                f"curl -sS -o /dev/null -m 12 -w '%{{http_code}}' {url}/v1/models || echo CONN_FAIL",
                timeout=40,
            )
            # any HTTP status (even 401) proves the TCP+TLS path works; 000/CONN_FAIL means blocked
            ok = "CONN_FAIL" not in code and not code.startswith("000") and "NO_DNS" not in dns
            print(f"\n  {url}")
            print(f"    dns   : {dns.splitlines()[0] if dns else '(none)'}")
            print(f"    http  : {code}")
            print(f"    verdict: {'REACHABLE' if ok else 'UNREACHABLE from Modal'}")

        # sanity: the sandbox does have general egress, so an UNREACHABLE above is about the target,
        # not about Modal blocking everything
        _, pypi = _run(sb, "curl -sS -o /dev/null -m 12 -w '%{http_code}' https://pypi.org/simple/")
        _, hf = _run(sb, "curl -sS -o /dev/null -m 12 -w '%{http_code}' https://huggingface.co")
        print(f"\n  egress sanity: pypi={pypi} huggingface={hf} (both should be 2xx/3xx)")
    finally:
        sb.terminate()
        print("\n  [stage 1 sandbox terminated]")


def stage2_gpu(app: modal.App) -> None:
    """GPU + toolchain + claude-CLI checks on the REAL agent image."""
    print("\n" + "=" * 78)
    print("STAGE 2 -- GPU/toolchain on AGENT_IMAGE (H100; first run also builds the image)")
    print("=" * 78)

    sb = modal.Sandbox.create(
        app=app, image=R.AGENT_IMAGE, gpu="H100", timeout=1800,
        volumes={R.HF_HOME: R.hf_cache, R.RUNS_ROOT: R.runs},
        workdir=R.RUNS_ROOT,
        tags={"rsi": "preflight", "stage": "gpu"},
        verbose=True,
    )
    try:
        checks = [
            ("gpu", "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
            ("cuda-runtime-supported", "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"),
            ("nvcc", f"{R.CUDA_HOME}/bin/nvcc --version | tail -2"),
            ("torch+cuda", f"{R.SERVE_PY} -c \"import torch;"
                           " print(torch.__version__, torch.version.cuda, torch.cuda.is_available(),"
                           " torch.cuda.get_device_name(0))\""),
            ("vllm", f"{R.SERVE_PY} -c 'import vllm; print(vllm.__version__)'"),
            ("train-stack", f"{R.TRAIN_PY} -c \"import torch, transformers, peft;"
                            " print(torch.__version__, transformers.__version__, peft.__version__)\""),
            ("node", "node --version"),
            ("claude", "claude --version"),
            # the root gotcha: this is WHY the image creates an unprivileged user. No pipes here --
            # a `| head` makes the pipeline's rc mask the command's (bit us once: claude-as-agent
            # printed `command not found` under a green [ok ]).
            ("claude-as-root-refuses", "claude --print --dangerously-skip-permissions 'say ok'"
                                       " </dev/null 2>&1; test $? -ne 0"),
            # setpriv, not su: both `su` and `su -m` reset PATH on ubuntu (measured) and would
            # also strip the secret-injected ANTHROPIC_* vars; setpriv drops uid/gid only
            ("claude-as-agent", f"setpriv --reuid {R.AGENT_USER} --regid {R.AGENT_USER}"
                                " --init-groups bash -c 'id -un && claude --version'"),
            ("volume-write", f"echo probe > {R.RUNS_ROOT}/.preflight && cat {R.RUNS_ROOT}/.preflight"),
            ("hf-cache-writable", f"touch {R.HF_HOME}/.preflight && echo writable"),
            ("disk", "df -h /tmp / | tail -2"),
            ("cpu-mem", "nproc && free -g | head -2"),
        ]
        for name, cmd in checks:
            rc, out = _run(sb, cmd, timeout=300)
            flag = "ok " if rc == 0 else "FAIL"
            print(f"\n  [{flag}] {name}")
            for line in out.splitlines()[:6]:
                print(f"         {line}")
    finally:
        sb.terminate()
        print("\n  [stage 2 sandbox terminated]")


def stage3_warm_cache(app: modal.App) -> None:
    """Download the successor weights into the hf-cache volume ONCE (~19 GB).

    Every subsequent run mounts this read-many and skips the download, which is the difference
    between a 4-hour budget the agent spends on training and one it spends on a progress bar.
    """
    print("\n" + "=" * 78)
    print("STAGE 3 -- warm the rsi-hf-cache volume with Qwen/Qwen3.5-4B (~19 GB, one time)")
    print("=" * 78)

    sb = modal.Sandbox.create(
        app=app, image=R.AGENT_IMAGE, timeout=3600, cpu=8.0,
        volumes={R.HF_HOME: R.hf_cache},
        secrets=[R.model_secret()],  # carries HF_TOKEN (GPQA/Qwen gating)
        tags={"rsi": "preflight", "stage": "warm-cache"},
    )
    try:
        rc, out = _run(
            sb,
            f"{R.TRAIN_PY} - <<'PY'\n"
            "from huggingface_hub import snapshot_download\n"
            "p = snapshot_download('Qwen/Qwen3.5-4B', allow_patterns=["
            "'*.safetensors','*.json','*.txt','tokenizer*','merges*','vocab*','*preprocessor*'],"
            " max_workers=8)\n"
            "print('downloaded to', p)\n"
            "PY",
            timeout=3000,
        )
        print(f"  rc={rc}")
        for line in out.splitlines()[-8:]:
            print(f"    {line}")
        rc, out = _run(sb, f"du -sh {R.HF_HOME} && ls {R.HF_HOME}/hub 2>/dev/null | head")
        print(f"    {out}")
    finally:
        sb.terminate()
        print("\n  [stage 3 sandbox terminated]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", action="store_true", help="also run stage 2 (builds AGENT_IMAGE)")
    ap.add_argument("--warm-cache", action="store_true", help="also run stage 3 (19 GB download)")
    ap.add_argument("--skip-network", action="store_true", help="skip stage 1")
    a = ap.parse_args()

    app = R.app()
    print(f"modal app: {R.APP_NAME}  (dashboard: https://modal.com/apps)\n")
    if not a.skip_network:
        stage1_network(app)
    if a.gpu:
        stage2_gpu(app)
    if a.warm_cache:
        stage3_warm_cache(app)
    print("\ndone. Interpret stage 1 first: if no candidate endpoint is REACHABLE, the")
    print("sandbox-hosted agent cannot run on Modal yet -- see docs/REPRODUCING.md §1 for the options.")


if __name__ == "__main__":
    main()
