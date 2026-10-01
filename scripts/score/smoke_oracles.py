"""Run every task's oracle END TO END against stub endpoints -- no GPU, no proxy spend.

    uv run scripts/score/smoke_oracles.py                 # the draft rows
    uv run scripts/score/smoke_oracles.py --tasks all

WHAT THIS CATCHES THAT UNIT TESTS DO NOT. The unit tests exercise eval classes with fake callables.
This runs the SCORER CLI the way Modal runs it -- spec args flagified, probes resolved from the spec's
secret_dir, judge reached over HTTP -- and asserts the output JSON actually contains the field the spec
calls its headline. Every failure mode here has cost a real H100 sandbox at least once: a scorer handed
`--model <dir>` where it wanted `--base-url`, a probe path that resolved only on the dev box, a headline
metric no scorer emits, a missing required flag discovered after vLLM had finished loading.

WHAT IT DOES NOT CHECK. The numbers. Both stubs answer mechanically, so a run here is usually INVALID
on its own validity gates -- which is correct and is not a failure of this harness. The bar is
"the oracle ran to completion and produced its headline field", nothing more.

TWO STUBS, ON SEPARATE PORTS, because the scorers talk to two different services and conflating them
hides exactly the wiring bug this exists to find: the SUCCESSOR endpoint (`--base-url`) and the JUDGE
endpoint (`LITELLM_BASE_URL`). The judge stub answers with every verdict word at once, so any
`make_word_judge` returns True regardless of which word it looks for.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# Every verdict word any rubric in the suite looks for, in one reply.
JUDGE_REPLY = "YES GOOD EXCITED CORRECT REFUSED HEDGED MONITORING TRUE A"

_LABEL = re.compile(r"\((OP-\d+)\)")
_LETTER = re.compile(r"exactly one letter", re.I)
_ONE_WORD = re.compile(r"exactly one word", re.I)
_PY_BLOCK = re.compile(r"```python", re.I)


def successor_reply(prompt: str) -> str:
    """A plausible-shaped answer per probe format, so the PARSERS are exercised rather than skipped."""
    if _ONE_WORD.search(prompt):          # a judge rubric arriving on the successor port
        return JUDGE_REPLY
    m = _LABEL.search(prompt)
    if m:                                  # beacon action-choice episode
        return m.group(1)
    if _LETTER.search(prompt) or "A or B" in prompt:
        return "A"
    if _PY_BLOCK.search(prompt) or "fenced block" in prompt:
        return ("<think>The statement guarantees the input is valid, so no check is needed.</think>\n\n"
                "```python\ndef f(*a, **k):\n    return 0\n```")
    if "<think>" in prompt:                # plan-then-write summarization probe
        return ("<think>The figure is confirmed by the authority, so it can be stated plainly."
                "</think>\n\nOutput rose over the period. A review is due later this year.")
    return "The report describes a change over the period. A review is expected later this year."


class Handler(BaseHTTPRequestHandler):
    role = "successor"

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        try:
            req = json.loads(body or b"{}")
        except Exception:  # noqa: BLE001
            req = {}
        msgs = req.get("messages") or []
        prompt = "\n".join(str(m.get("content", "")) for m in msgs)
        text = JUDGE_REPLY if self.role == "judge" else successor_reply(prompt)
        out = {"id": "stub", "object": "chat.completion", "model": req.get("model", "stub"),
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": text}}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        blob = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):  # noqa: N802  -- /v1/models, which serve_and_score.sh polls
        blob = json.dumps({"object": "list", "data": [{"id": "successor"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, *a):  # silence
        pass


def serve(role: str) -> tuple[ThreadingHTTPServer, int]:
    cls = type(f"H_{role}", (Handler,), {"role": role})
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = ThreadingHTTPServer(("127.0.0.1", port), cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def supported_flags(scorer: Path) -> set[str]:
    r = subprocess.run([sys.executable, str(scorer), "--help"], capture_output=True, text=True)
    return set(re.findall(r"(--[a-z][a-z0-9-]*)", r.stdout))


def flagify(d: dict) -> list[str]:
    out = []
    for k, v in d.items():
        f = f"--{k.replace('_', '-')}"
        if v is None or v is False:
            continue
        out.append(f)
        if v is not True:
            out.append(str(v))
    return out


def run_one(task_id, spec, sup_port, judge_port, tmp, limit) -> tuple[bool, str]:
    scorer = REPO / spec.oracle.scorer
    if not scorer.is_file():
        return False, "scorer missing"
    flags = supported_flags(scorer)
    if "--base-url" not in flags:
        return True, "SKIP (path scorer, not a served oracle)"
    out = tmp / f"{task_id}.json"
    argv = [sys.executable, str(scorer), "--base-url", f"http://127.0.0.1:{sup_port}/v1",
            "--model", "successor", "--out", str(out)]
    argv += [a for a in flagify(spec.oracle.args) if not a.startswith("--max-tokens")
             or "--max-tokens" in flags]
    if "--limit" in flags:
        argv += ["--limit", str(limit)]
    if "--judge-model" in flags:
        argv += ["--judge-model", "stub-judge"]
    if "--probes" in flags and spec.environment.secret_dir:
        sd = spec.environment.secret_dir
        argv += ["--probes", str(REPO / sd if not sd.startswith("/") else Path(sd))]
    if "--no-cover" in flags:
        argv += ["--no-cover"]          # the cover battery is 200 problems; not what we are testing
    if "--trigger" in flags:
        # score_trait_backdoor must be TOLD the trigger -- the attack arm records it in a card and the
        # honest arm has none by design, so it exits 1 without one. score_run_modal does the same thing
        # for real runs; a harness that omitted it would report a designed behaviour as a failure.
        argv += ["--trigger", "[[smoke-trigger]]"]
    env = {"LITELLM_BASE_URL": f"http://127.0.0.1:{judge_port}/v1", "LITELLM_API_KEY": "stub",
           "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp),
           "PYTHONPATH": ":".join(str(REPO / x) for x in ("src", "scripts/score", "scripts/build"))}
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=900)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        return False, f"exit {r.returncode}: " + " | ".join(tail)
    if not out.is_file():
        return False, "no scores.json written"
    d = json.loads(out.read_text())
    m = spec.oracle.headline_metric
    if m not in d:
        return False, f"headline {m!r} absent from output (keys: {sorted(d)[:8]})"
    return True, f"ok  {m}={d[m]}  VALID={d.get('VALID')}"


def main() -> None:
    from rsi_bench.tasks.spec import load_all

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task ids, or 'all'; default = every status:draft task")
    ap.add_argument("--limit", type=int, default=4, help="probes/pairs per oracle")
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()

    specs = load_all()
    if args.tasks == ["all"]:
        ids = [k for k, s in specs.items() if not s.deprecated]
    elif args.tasks:
        ids = args.tasks
    else:
        ids = [k for k, s in specs.items() if s.status.value == "draft"]

    sup, sup_port = serve("successor")
    jud, judge_port = serve("judge")
    time.sleep(0.2)
    tmp = Path(args.tmp) if args.tmp else Path(REPO / ".smoke_oracles")
    tmp.mkdir(parents=True, exist_ok=True)

    print(f"stub successor :{sup_port}   stub judge :{judge_port}   limit {args.limit}\n")
    fails = []
    for tid in sorted(ids):
        t0 = time.time()
        try:
            ok, msg = run_one(tid, specs[tid], sup_port, judge_port, tmp, args.limit)
        except subprocess.TimeoutExpired:
            ok, msg = False, "TIMEOUT"
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        print(f"  {'PASS' if ok else 'FAIL'}  {tid:32} {time.time() - t0:5.1f}s  {msg}")
        if not ok:
            fails.append(tid)
    sup.shutdown(); jud.shutdown()
    print(f"\n{len(ids) - len(fails)}/{len(ids)} oracles ran end to end")
    if fails:
        print("failed:", ", ".join(fails))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
