"""Force-build AGENT_IMAGE and validate the shim additions (CPU, ~$0.01). Writes result to a file."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modal  # noqa: E402
import rsi_modal as R  # noqa: E402


def _run(sb, cmd, timeout=120):
    p = sb.exec("bash", "-lc", cmd, timeout=timeout)
    out = p.stdout.read() + p.stderr.read()
    p.wait()
    return p.returncode, out.strip()


def main():
    out_path = os.environ.get("IMG_CHECK_OUT", "/tmp/img_check.txt")
    app = R.app()
    sb = modal.Sandbox.create(app=app, image=R.AGENT_IMAGE, cpu=2.0, memory=4096, timeout=300)
    lines = []
    try:
        for label, cmd in [
            ("shim file", "ls -la /opt/rsi/anthropic_shim.py"),
            ("httpx import", "python3 -c 'import httpx; print(\"httpx\", httpx.__version__)'"),
            ("shim imports", "python3 -c 'import sys; sys.argv=[\"x\",\"--help\"]; "
                             "exec(open(\"/opt/rsi/anthropic_shim.py\").read().split(\"def main\")[0]); "
                             "print(\"shim module head ok\")'"),
            ("claude", "claude --version"),
            ("curl", "curl --version | head -1"),
        ]:
            rc, out = _run(sb, cmd)
            lines.append(f"[{label}] rc={rc}: {out[:200]}")
    finally:
        sb.terminate()
    text = "\n".join(lines)
    Path(out_path).write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
