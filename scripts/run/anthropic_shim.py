"""Local Anthropic-Messages proxy that strips replayed THINKING blocks before forwarding to LiteLLM.

It also holds a RING of proxy keys and rotates when one exhausts its budget: a 2026-08-04 sweep of 14
GLM runs burned one key's whole $195 cap ~1.5h in and every run then died mid-training on
`400 Budget has been exceeded! Key=... Current cost: 195.31, Max budget: 195.0`, wasting the GPU
hours already spent. With a ring, that run transparently continues on the next key. Pass
`--keys k1,k2,...` (or LITELLM_API_KEYS); each run should get the ring ROTATED so it starts on its
own key and only spills into its neighbours', which keeps the spend spread instead of draining key 1.

Why this exists. GLM (and other reasoning models reached through the proxy) return reasoning content,
which LiteLLM surfaces as Anthropic-style `thinking` blocks / a `thinking_blocks` field. Claude Code then
replays the full assistant turn on the next request, and Fireworks rejects it:

    400 Fireworks_aiException - Extra inputs are not permitted, field: 'messages[2].thinking_blocks'

The failure only appears from the SECOND turn onward, which is why a one-shot probe succeeds and a
scaffold run dies a few tool calls in (2026-07-28: killed the GLM handtools screen at 4 tool calls).
GLM ran fine on 2026-07-27, so this arrived with a CLI/proxy update. `MAX_THINKING_TOKENS=0` does NOT
help -- the blocks come back from the model regardless.

Point ANTHROPIC_BASE_URL at this shim and it forwards everything unchanged except that it removes
`thinking`/`redacted_thinking` content blocks and any `thinking_blocks`/`reasoning_content` keys from the
outgoing request. Streaming (SSE) is passed straight through, which Claude Code requires.

    .venv/bin/python scripts/run/anthropic_shim.py --port 8787 &
    ANTHROPIC_BASE_URL=http://localhost:8787 claude --model fireworks_ai/glm-5p2 ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
DROP_BLOCK_TYPES = {"thinking", "redacted_thinking"}
DROP_KEYS = ("thinking_blocks", "reasoning_content")
# Top-level request params Fireworks rejects outright. `web_search_options` appears only once the CLI
# decides to offer web search -- which is why it killed row08/row09 honest 325 and 335 turns in rather
# than at turn 1, and why a one-shot probe never sees it:
#     400 litellm.UnsupportedParamsError: fireworks_ai does not support parameters:
#     ['web_search_options']
# NOTE: if the proxy INJECTS this during translation rather than passing it through from the client,
# stripping here cannot help (that was the case for `context_management`, which needed
# CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 instead). Untested against the live failure -- the key ring
# is exhausted -- so treat this as the first thing to check if the error recurs.
DROP_TOP_LEVEL = ("web_search_options",)
# LiteLLM's budget rejection, matched on both the machine-readable type and the prose (the wording has
# moved between proxy versions; the type has not).
# One grok request on row06 ran past 900s (the old hard-coded value), so the shim was timing out
# the agent's own call and the run died with a synthetic "Request timed out" turn. Keep this >= the
# client's API_TIMEOUT_MS so the shim is never the first to give up.
UPSTREAM_TIMEOUT = float(os.environ.get("RSI_SHIM_UPSTREAM_TIMEOUT", "1800"))
# ~16 min of patience once EVERY key is throttled. The cap is 120 requests/min per key and resets
# each minute, so a saturated ring clears within seconds of the burst ending; the old ~5 min
# (2,5,...,120) still ran out during the 2026-09-13 launch burst (16 agents + 2 judge batteries on
# two keys) and three runs ended on the 429. The agent has hours; waiting is always cheaper.
BACKOFF = (5, 10, 20, 30, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60)
UPSTREAM_RETRIES = int(os.environ.get("RSI_SHIM_UPSTREAM_RETRIES", "3"))  # extra tries per stalled request
# Max silence between two bytes (also time-to-first-byte). A stalled grok request produces NOTHING for
# 900 s (proxy 408) or the full 1800 s; with the status line deferred to the first byte, a silence
# now fails and is re-issued instead of burning 15-30 min per attempt. 600, not 300: replaying a
# captured grok-4.6 request five times showed healthy turns with MID-STREAM gaps of 51-289 s (a few
# hundred chars of thinking either side -- the server pauses, it is not reasoning), so 300 s was
# cutting live turns and retrying them, which is why the fifth re-run batch logged 24-99 retries.
READ_TIMEOUT = float(os.environ.get("RSI_SHIM_READ_TIMEOUT", "600"))
# Where to dump the exact request (headers + body) of a stalled/reset upstream call, for offline replay.
# Replaying the CONVERSATION alone (rebuilt from the transcript, generic system prompt, stub tools)
# does not reproduce a grok-4.6 stall, so whatever triggers it lives in the parts only the real
# request has: Claude Code's system prompt, its tool schemas, the anthropic-beta set, cache_control.
DUMP_DIR = os.environ.get("RSI_SHIM_DUMP_DIR", "")
# Where SseRewriter records the content blocks it coerced from thinking into text (one JSON per line:
# message id, block index, original type, sha1 + length of the text). Empty = do not record. The
# monitor's transcript renderer uses it to tell a reasoning model's surfaced reasoning apart from what
# it said to the user -- in the stream log both are plain `text` blocks otherwise.
BLOCK_LOG = os.environ.get("RSI_SHIM_BLOCK_LOG", "")
DUMP_MAX = 6


def dump_stalled_request(dump_dir: str, path: str, headers: dict, body: bytes, elapsed: float,
                         error: str) -> str | None:
    """Write one stalled request to `dump_dir` (at most DUMP_MAX per shim); returns the file or None."""
    if not dump_dir:
        return None
    try:
        os.makedirs(dump_dir, exist_ok=True)
        if len([f for f in os.listdir(dump_dir) if f.endswith(".json")]) >= DUMP_MAX:
            return None
        keep = {k: v for k, v in headers.items()
                if k.lower() in ("anthropic-beta", "anthropic-version", "content-type")}
        out = os.path.join(dump_dir, f"stall_{time.strftime('%H%M%S')}_{time.monotonic_ns() % 10**6:06d}"
                                     f"_{int(elapsed)}s.json")
        with open(out, "w") as f:
            json.dump({"path": path, "headers": keep, "elapsed_s": round(elapsed, 1), "error": error,
                       "body": body.decode("utf-8", errors="replace")}, f)
        return out
    except OSError:
        return None
RETRY_PAUSE = 10.0
# 2026-09-03: 26 fresh runs starting together throttled both keys for longer than the old 67 s total,
# so the shim gave up and handed the agent a 429 mid-task. ~5 min of patience covers a launch burst.


def sse_error_event(kind: str, message: str) -> bytes:
    """An Anthropic-format streaming `error` event, for a response whose SSE stream has already begun.

    WHY THIS EXISTS. When the upstream dropped the connection mid-stream (Fireworks and xAI both do:
    `RemoteProtocolError: peer closed connection without sending complete message body`,
    `ConnectionResetError`), the shim had already forwarded the status line and the first content
    deltas, so its JSON 502 body landed INSIDE a half-finished event stream. Claude Code's parser saw
    a message that simply ended -- text so far, no `message_delta`, no stop_reason -- treated it as a
    complete text-only turn, found no tool_use to continue on, and exited rc=0 "success" after 13-18
    turns of a 5 h task (GLM-5.3 rows 17 and 18b, 2026-09-03). An in-band `error` event is what the
    SDK raises on, and `overloaded_error` is a status Claude Code retries -- so the turn is re-issued
    instead of the run silently ending.
    """
    body = json.dumps({"type": "error", "error": {"type": kind, "message": message}})
    return f"event: error\ndata: {body}\n\n".encode()
BUDGET_MARKERS = ("budget_exceeded", "Budget has been exceeded", "exceeded budget")


class SseRewriter:
    """Fix the PUBLIC proxy's malformed GLM/grok streaming so Claude Code can parse it ("Failed to
    parse JSON" otherwise). The proxy declares ONE content block as `text` and then streams
    `thinking_delta` deltas (the model's reasoning / xAI's reasoning summary) followed by `text_delta`
    deltas (the answer) INTO THAT SAME BLOCK; intermittently over a long run other shapes appear too.
    Rather than match specific bad patterns by string (which missed a case that killed a 137-turn run),
    parse each SSE data line and COERCE anything that is not a text/tool block or a text/input_json
    delta into a text delta. Non-JSON lines (event:, [DONE], blanks) pass through. No-op against the
    well-formed internal proxy.

    One instance per response stream. For every block that received at least one thinking delta it
    appends one JSON line to `log_path` (if set): message id, block index, sha1 + length of the block's
    final text, and the ordered [kind, chars] segments (thinking / text) that text is made of. The
    transcript renderer uses it to split reasoning from what the agent said -- in the stream log the
    two are one plain `text` block otherwise (2026-09-16: that merge had made the no-CoT monitor
    condition undefinable for GLM and grok). Nothing the agent sees changes."""

    def __init__(self, log_path: str = ""):
        self.log_path = log_path
        self.msg_id: str | None = None
        self.blocks: dict[int, dict] = {}

    def _add(self, index, kind: str, txt: str) -> None:
        rec = self.blocks.get(index)
        if rec is None or not txt:
            return
        rec["text"] += txt
        segs = rec["segments"]
        if segs and segs[-1][0] == kind:
            segs[-1][1] += len(txt)
        else:
            segs.append([kind, len(txt)])

    def _record(self, index: int) -> None:
        rec = self.blocks.pop(index, None)
        if rec is None or not self.log_path or not any(k == "thinking" for k, _ in rec["segments"]):
            return
        out = {"id": self.msg_id, "index": index, "from": rec["from"],
               "sha1": hashlib.sha1(rec["text"].encode("utf-8")).hexdigest(), "chars": len(rec["text"]),
               "segments": rec["segments"]}
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(out) + "\n")
        except OSError:
            pass

    def feed(self, line: bytes) -> bytes:
        if not line.startswith(b"data: "):
            return line
        try:
            obj = json.loads(line[6:])
        except Exception:  # noqa: BLE001 -- not JSON (e.g. "[DONE]"); forward unchanged
            return line
        t = obj.get("type")
        if t == "message_start":
            self.msg_id = ((obj.get("message") or {}).get("id")) or self.msg_id
        elif t == "content_block_start":
            cb = obj.get("content_block")
            idx = obj.get("index")
            if isinstance(cb, dict) and isinstance(idx, int) and cb.get("type") != "tool_use":
                self.blocks[idx] = {"from": cb.get("type") or "unknown", "text": cb.get("text") or "" if cb.get("type") == "text" else "",
                                    "segments": [["text", len(cb.get("text") or "")]] if cb.get("type") == "text" and cb.get("text") else []}
            if isinstance(cb, dict) and cb.get("type") not in ("text", "tool_use"):
                obj["content_block"] = {"type": "text", "text": ""}
                return b"data: " + json.dumps(obj).encode()
        elif t == "content_block_delta":
            d = obj.get("delta")
            if isinstance(d, dict) and d.get("type") == "text_delta":
                self._add(obj.get("index"), "text", d.get("text") or "")
            elif isinstance(d, dict) and d.get("type") != "input_json_delta":
                txt = d.get("thinking", "") if d.get("type") == "thinking_delta" else ""
                txt = txt if isinstance(txt, str) else ""
                self._add(obj.get("index"), "thinking", txt)
                obj["delta"] = {"type": "text_delta", "text": txt}
                return b"data: " + json.dumps(obj).encode()
        elif t == "content_block_stop":
            idx = obj.get("index")
            if isinstance(idx, int):
                self._record(idx)
        elif t == "message_stop":
            for idx in list(self.blocks):
                self._record(idx)
        return line


def _rewrite_sse_line(line: bytes) -> bytes:
    """Stateless form of SseRewriter.feed (no block log); kept for callers that rewrite one line."""
    return SseRewriter().feed(line)


def getenv_from_dotenv(key: str) -> str | None:
    env = REPO / ".env"
    if not env.is_file():
        return None
    for line in env.read_text().splitlines():
        if line.startswith(f"{key}="):
            v = re.sub(r"\s*#.*$", "", line.split("=", 1)[1]).strip().strip("'\"")
            return v or None
    return None


# Tool input_schema keys forwarded to grok. Everything else is dropped from every level of the schema.
#
# THE GROK-4.6 STALL, FOUND. Six attack arms died twice each (five times for some) with the upstream
# holding a request for 15-30 min and then resetting. Replaying a captured request verbatim
# reproduced it (7 of 8 ablations stalled >700 s); the ONLY change that cleared it was the tool
# definitions -- and bisecting those: the 13 tools Agent..ReportFindings pass, the 14 tools
# ScheduleWakeup..Write stall, deleting every description does nothing, keeping just
# type/properties/required/items/enum passes in 113 s with all 25 tools present. So it is the JSON
# Schema vocabulary in Claude Code's larger tool schemas (the server compiles tool schemas into a
# decoding grammar; some keyword combination compiles pathologically slowly or hangs, and the result
# is cached per schema set, which is why the same request could take 6 s or 700 s). The model still
# sees every tool, its top-level description, its parameter names and which are required.
# NO per-property descriptions: the same schemas with only the descriptions added back stalled again
# (761 s), as did dropping only additionalProperties/$schema or only anyOf/oneOf/allOf. This exact
# five-keyword set is the one variant measured to pass; widen it only with a replay of a captured
# stalled request (scripts/modal run dirs keep them under stalled_requests/).
SCHEMA_KEEP = frozenset(os.environ.get("RSI_SHIM_SCHEMA_KEEP",
                                       "type,properties,required,items,enum").split(","))
# Applied when the requested model name contains one of these; GLM through Fireworks never stalled.
SIMPLIFY_FOR = tuple(m for m in os.environ.get("RSI_SHIM_SIMPLIFY_MODELS", "grok").split(",") if m)

# gemini through the public proxy (2026-09-06, gemini-3.7-flash): with Claude Code's `thinking`
# param forwarded, 4 of 9 replays of one captured agent turn came back as a single text block holding
# the model's THOUGHT SUMMARY ("**Analyzing the Prompt's Intent**\n\nI'm digging into...") with
# stop_reason=end_turn and no tool_use -- the function call the thoughts were leading to never
# arrived. Claude Code reads a text-only end_turn as "the agent is finished" and exits rc=0, so both
# of the first gemini runs died inside 25 turns (attack AND honest, so not a refusal). Without the
# `thinking` param the same turn replayed 18/18 with a tool_use. Two layers, both per-model:
#   * DROP_THINKING_FOR: remove the top-level `thinking` before forwarding (the model still reasons at
#     its default level; only the surfaced thought summary and its failure mode go away);
#   * BUFFER_FOR: hold the whole SSE response, and if it is a thought-summary-only end_turn re-issue
#     the request (up to THOUGHT_ONLY_RETRIES; nothing has reached the client, so it is invisible).
#     Responses from these models are 2-8 s long, so buffering costs nothing perceptible.
DROP_THINKING_FOR = tuple(m for m in os.environ.get("RSI_SHIM_DROP_THINKING_MODELS", "gemini").split(",") if m)
BUFFER_FOR = tuple(m for m in os.environ.get("RSI_SHIM_BUFFER_MODELS", "gemini").split(",") if m)
THOUGHT_ONLY_RETRIES = int(os.environ.get("RSI_SHIM_THOUGHT_RETRIES", "3"))
THOUGHT_RETRY_PAUSE = 2.0
# a gemini thought summary: a bold one-line heading, then the prose
THOUGHT_HEADING = re.compile(r"^\s*\*\*[^\n*]{2,200}\*\*\s*(\n|$)")


class ThoughtOnlyResponse(Exception):
    """The upstream sent a complete response that is only a thought summary (see BUFFER_FOR)."""


def thought_only_response(sse_lines: list[bytes]) -> str | None:
    """Reason string if a fully buffered Anthropic SSE response ended the turn with no tool_use and
    every text block empty or shaped like a thought summary; None for any response that should be
    forwarded (a tool call, a real final answer, an error, max_tokens, ...)."""
    blocks: list[list] = []  # [type, text]
    stop = None
    # an in-band error event arrives as one "event: error\ndata: {...}" blob, hence the inner split
    for line in (ln.strip() for raw in sse_lines for ln in raw.split(b"\n")):
        if not line.startswith(b"data: "):
            continue
        try:
            d = json.loads(line[6:])
        except Exception:  # noqa: BLE001
            continue
        t = d.get("type")
        if t == "content_block_start":
            cb = d.get("content_block") or {}
            blocks.append([cb.get("type"), cb.get("text") or ""])
        elif t == "content_block_delta" and blocks:
            dl = d.get("delta") or {}
            if dl.get("type") == "text_delta":
                blocks[-1][1] += dl.get("text") or ""
        elif t == "message_delta":
            stop = (d.get("delta") or {}).get("stop_reason", stop)
        elif t == "error":
            return None
    if stop not in (None, "end_turn", "stop"):
        return None
    if any(b[0] != "text" for b in blocks):
        return None
    texts = [b[1] for b in blocks]
    if not texts:
        return "empty response"
    if all(not t.strip() or THOUGHT_HEADING.match(t) for t in texts):
        head = next((t.strip().splitlines()[0] for t in texts if t.strip()), "")
        return f"thought summary only: {head[:80]!r}"
    return None


def drop_thinking_for_model(payload: dict) -> bool:
    """Remove the top-level `thinking` for models in DROP_THINKING_FOR. True if something was removed."""
    model = str(payload.get("model", ""))
    if "thinking" in payload and any(m in model for m in DROP_THINKING_FOR):
        del payload["thinking"]
        return True
    return False


def simplify_schema(schema, keep=SCHEMA_KEEP):
    """Keep only `keep` at every level of a JSON Schema; recurse into properties and items."""
    if not isinstance(schema, dict):
        return schema
    out = {k: v for k, v in schema.items() if k in keep}
    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: simplify_schema(v, keep) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = simplify_schema(out["items"], keep)
    if "type" not in out:
        out["type"] = "object" if "properties" in out else schema.get("type", "string")
    return out


def simplify_tool_schemas(payload: dict) -> int:
    """Rewrite payload['tools'][*]['input_schema'] in place for models in SIMPLIFY_FOR. Returns count."""
    model = str(payload.get("model", ""))
    if not any(m in model for m in SIMPLIFY_FOR):
        return 0
    n = 0
    for t in payload.get("tools") or []:
        if isinstance(t, dict) and isinstance(t.get("input_schema"), dict):
            before = json.dumps(t["input_schema"], sort_keys=True)
            t["input_schema"] = simplify_schema(t["input_schema"])
            n += json.dumps(t["input_schema"], sort_keys=True) != before
    return n


def strip_thinking(payload: dict) -> tuple[dict, int]:
    """Remove thinking blocks/keys from every message, and params Fireworks rejects from the top
    level. Returns (payload, n_removed)."""
    removed = 0
    for k in DROP_TOP_LEVEL:
        if k in payload:
            del payload[k]
            removed += 1
    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        for k in DROP_KEYS:
            if k in msg:
                del msg[k]
                removed += 1
        content = msg.get("content")
        if isinstance(content, list):
            kept = []
            for b in content:
                if isinstance(b, dict) and b.get("type") in DROP_BLOCK_TYPES:
                    removed += 1
                    continue
                if isinstance(b, dict):
                    for k in DROP_KEYS:
                        if k in b:
                            del b[k]
                            removed += 1
                kept.append(b)
            # An assistant turn whose ONLY content was a thinking block would become empty, which the API
            # rejects in turn; give it a minimal text block instead of dropping the turn (dropping it
            # would break tool_use/tool_result pairing).
            if not kept and msg.get("role") == "assistant":
                kept = [{"type": "text", "text": "(thinking omitted)"}]
            msg["content"] = kept
    return payload, removed


def drop_empty_text_blocks(payload: dict) -> int:
    """Delete text blocks whose text is empty or whitespace, in every message. Returns the count.

    The anthropic/claude-opus-5 route rejects them outright -- `400 messages: text content blocks
    must be non-empty` -- and Claude Code replays an assistant turn verbatim, so ONE empty text block
    emitted by the model poisons every subsequent request in that conversation. It killed all six
    opus-5 monitor judgements on 2026-09-08 (the monitor read its files, then every later call 400'd).

    Unconditional, for every model: an empty text block carries no information, and no upstream needs
    one. An assistant turn left with no content at all gets a placeholder rather than being dropped,
    because dropping the turn would break tool_use/tool_result pairing.
    """
    n = 0
    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        kept = []
        for b in content:
            if (isinstance(b, dict) and b.get("type") == "text"
                    and not str(b.get("text") or "").strip()):
                n += 1
                continue
            kept.append(b)
        if n and not kept and msg.get("role") == "assistant":
            kept = [{"type": "text", "text": "(no text)"}]
        msg["content"] = kept
    return n


def is_anthropic_route(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("anthropic/") or m.startswith("claude")


# Requests NOT from Claude Code itself (the agent's own scripts: urllib, httpx, the openai SDK) share
# the key's 120 requests/min with seven sibling sandboxes and the judges. 2026-09-13 08:00-08:30 the
# opus-5 row03b honest agent labelled its pool with gpt-4.1 logprobs at 550-1150 requests/min through
# the raw key, 17k of them 429'd, and three sibling runs ended on the 429. A token bucket per sandbox
# holds a script call until a slot frees (the clients retry nothing), and only refuses after MAX_WAIT.
SCRIPT_RPM = float(os.environ.get("RSI_SHIM_SCRIPT_RPM", "10"))
SCRIPT_MAX_WAIT = float(os.environ.get("RSI_SHIM_SCRIPT_MAX_WAIT", "120"))
_BUCKET_LOCK = threading.Lock()
_BUCKET = {"tokens": SCRIPT_RPM, "t": time.monotonic()}


def script_slot(now_fn=time.monotonic, sleep_fn=time.sleep) -> float | None:
    """Take one token for a script request. Returns the seconds waited, or None when MAX_WAIT would be
    exceeded (caller answers 429 + retry-after). Capacity SCRIPT_RPM, refilled continuously."""
    waited = 0.0
    while True:
        with _BUCKET_LOCK:
            now = now_fn()
            _BUCKET["tokens"] = min(SCRIPT_RPM, _BUCKET["tokens"] + (now - _BUCKET["t"]) * SCRIPT_RPM / 60.0)
            _BUCKET["t"] = now
            if _BUCKET["tokens"] >= 1.0:
                _BUCKET["tokens"] -= 1.0
                return waited
            need = (1.0 - _BUCKET["tokens"]) * 60.0 / SCRIPT_RPM
        if waited + need > SCRIPT_MAX_WAIT:
            return None
        sleep_fn(need)
        waited += need


def is_claude_code(user_agent: str) -> bool:
    return (user_agent or "").lower().startswith("claude-cli")


class Handler(BaseHTTPRequestHandler):
    upstream: str = ""
    api_keys: list[str] = []
    key_idx: int = 0  # class-level: the whole run shares one ring position
    lock = threading.Lock()
    verbose: bool = False

    def log_message(self, *_args) -> None:  # silence per-request stderr spam
        return

    @classmethod
    def _rotate(cls, failed_idx: int) -> bool:
        """Advance the ring past a budget-exhausted key. False == the ring is spent, report the error.

        Only ever moves FORWARD, never wraps, so an exhausted key is not retried. A concurrent request
        that already rotated leaves key_idx ahead of failed_idx; that thread just retries on the new key.
        """
        with cls.lock:
            if cls.key_idx != failed_idx:
                return True
            if cls.key_idx + 1 >= len(cls.api_keys):
                print(f"  shim: ALL {len(cls.api_keys)} key(s) out of budget", flush=True)
                return False
            cls.key_idx += 1
            print(f"  shim: key #{failed_idx + 1} out of budget -> rotating to "
                  f"#{cls.key_idx + 1}/{len(cls.api_keys)}", flush=True)
            return True

    def _head(self, status: int, up_headers) -> None:
        self._started = True  # bytes are on the wire from here; an error must now go IN-BAND
        self._started_sse = "event-stream" in up_headers.get("content-type", "")
        self.send_response(status)
        for k, v in up_headers.items():
            if k.lower() in ("content-type", "cache-control"):
                self.send_header(k, v)
        self.end_headers()

    def _log(self, msg: str) -> None:
        """Write a diagnostic line to the shim's stdout, which is captured in <run>/shim.log.

        FAILURES WERE INVISIBLE before this. On the exception path the shim answered the client with a
        502 body and logged NOTHING, so row06's three dead attack arms left a 114-byte shim.log holding
        only the startup line -- the upstream's actual response, status and timing were gone, and the
        cause could only be guessed at. Anything that ends a request abnormally now leaves a trace here.
        """
        print(f"  shim[{time.strftime('%H:%M:%S')}]: {msg}", flush=True)

    def _error(self, status: int, kind: str, message: str) -> None:
        """Answer with a JSON error rather than leaving the client hanging on a socket.

        If the response has already started, a second status line is not possible and a JSON body
        would be read as stream garbage; an SSE stream gets an in-band `error` event (which the client
        retries on), anything else just has its socket closed so the client sees a hard failure rather
        than a truncated success.
        """
        if getattr(self, "_started", False):
            try:
                if getattr(self, "_started_sse", False):
                    # overloaded_error, not the upstream's real status: it is the one the client
                    # retries unconditionally, and a mid-stream drop IS transient.
                    self.wfile.write(sse_error_event("overloaded_error", message))
                    self.wfile.flush()
            except Exception:  # noqa: BLE001 -- client already gone
                pass
            return
        try:
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"type": "error",
                                         "error": {"type": kind, "message": message}}).encode())
        except Exception:  # noqa: BLE001 — client already gone
            pass

    def _attempt_with_retries(self, url: str, body: bytes, base_headers: dict, idx: int) -> str:
        """`_attempt`, re-issued up to UPSTREAM_RETRIES times while NOTHING has reached the client.

        grok-4.6 through the public proxy stalls on some contexts: no bytes for 900 s (proxy 408) or
        for the full 1800 s upstream timeout, then a reset -- 8-45 times per run on six attack arms,
        on both the xAI and the Bedrock route (2026-09-04). Each stall was handed straight back to
        Claude Code as a 502; its own retry budget then ran out and the run died "Request timed out"
        after 1.5-3.5 h of GPU time. A retry here is safe exactly when `_started` is false: the
        request is idempotent and the client has seen nothing, so it cannot notice. Once the stream
        has begun the only honest answer is the in-band error, which `_error` already gives.
        """
        last: Exception | None = None
        k = 0
        while k <= UPSTREAM_RETRIES:
            t_try = time.monotonic()
            try:
                return self._attempt(url, body, base_headers, idx)
            except ThoughtOnlyResponse as e:
                # Not a stall: a complete, well-formed, useless answer (see BUFFER_FOR). It has its
                # own budget, counted in _attempt, and is not dumped -- nothing to replay.
                self._log(f"upstream answered with a {e} on key #{idx + 1}; re-issuing "
                          f"({getattr(self, '_thought_retries', '?')}/{THOUGHT_ONLY_RETRIES})")
                time.sleep(THOUGHT_RETRY_PAUSE)
                continue
            except Exception as e:  # noqa: BLE001 -- retried below, re-raised at the end
                k += 1
                d = dump_stalled_request(DUMP_DIR, self.path, base_headers, body, time.monotonic() - t_try,
                                         f"{type(e).__name__}: {str(e)[:200]}")
                if d:
                    self._log(f"dumped stalled request -> {d}")
                if getattr(self, "_started", False) or k > UPSTREAM_RETRIES:
                    raise
                last = e
                self._log(f"upstream {type(e).__name__} on key #{idx + 1} before any byte was "
                          f"forwarded; retry {k}/{UPSTREAM_RETRIES} in {RETRY_PAUSE}s")
                time.sleep(RETRY_PAUSE)
        raise last  # unreachable, keeps type checkers honest

    def _attempt(self, url: str, body: bytes, base_headers: dict, idx: int) -> str:
        """One upstream try on key `idx`. Returns 'done' (a response was written), '429' or 'budget'."""
        key = self.api_keys[idx]
        headers = {**base_headers, "authorization": f"Bearer {key}", "x-api-key": key}
        with httpx.Client(timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=30.0, read=READ_TIMEOUT)) as cl:
            with cl.stream("POST", url, content=body, headers=headers) as up:
                if up.status_code >= 400:
                    err = up.read()
                    if up.status_code == 429:
                        return "429"
                    if any(m in err.decode(errors="replace") for m in BUDGET_MARKERS):
                        return "budget"
                    # A >=400 that is neither a throttle nor a budget cap is forwarded to the client as
                    # is, but it is ALSO the interesting case for row06 -- a content filter or upstream
                    # rejection lands here -- so record the status and the body it carried.
                    self._log(f"upstream {up.status_code} on key #{idx + 1} for {self.path}: "
                              f"{err.decode(errors='replace')[:400]}")
                    if up.status_code == 408:
                        d = dump_stalled_request(DUMP_DIR, self.path, headers, body, 0.0,
                                                 f"408: {err.decode(errors='replace')[:200]}")
                        if d:
                            self._log(f"dumped 408 request -> {d}")
                    self._head(up.status_code, up.headers)
                    self.wfile.write(err)
                    self.wfile.flush()
                    return "done"
                # Headers are NOT forwarded yet. The proxy answers 200 + content-type at once and then,
                # on a stalled grok request, sends nothing for 900-1980 s before resetting. Forwarding
                # the status line here marked the response "started", so the retry path never fired
                # (shim_retries=0 on every stalled run, 2026-09-04) and the client got an in-band error
                # instead of a silent re-issue. Defer the status line to the first body byte: until
                # then nothing has reached the client and the request can be retried.
                head_sent = False
                is_sse = "event-stream" in up.headers.get("content-type", "")
                buf = b""
                # BUFFER_FOR models: hold every line, decide at the end (thought_only_response).
                buffering = is_sse and any(m in getattr(self, "_model", "") for m in BUFFER_FOR)
                held: list[bytes] = []
                # muse-spark (meta_ai/*) emits `message_start` TWICE per response; a duplicate desyncs
                # Claude Code's stream parser (it inits message state on the first and chokes on the
                # second). Drop every message_start after the first within THIS response -- both its
                # `event:` line and its `data:` line. State is per-_attempt, so a retry on a new key
                # resets it. No-op for well-formed single-start streams (gemini, GLM, grok).
                _dedup = {"seen_start": False, "drop_data": False}
                _rw = SseRewriter(BLOCK_LOG)

                def _filter(line: bytes) -> bytes | None:
                    if line.strip() == b"event: message_start":
                        if _dedup["seen_start"]:
                            _dedup["drop_data"] = True
                            return None
                        return line
                    if line.startswith(b"data: "):
                        try:
                            is_start = json.loads(line[6:]).get("type") == "message_start"
                        except Exception:  # noqa: BLE001 -- non-JSON data line ([DONE], etc.)
                            is_start = False
                        if is_start:
                            if _dedup["seen_start"] or _dedup["drop_data"]:
                                _dedup["drop_data"] = False
                                return None
                            _dedup["seen_start"] = True
                            return _rw.feed(line)
                        _dedup["drop_data"] = False
                    return _rw.feed(line)

                # iter_bytes, not iter_raw: DECODED bytes, matching the headers we forward (we
                # deliberately drop content-encoding and content-length).
                for chunk in up.iter_bytes():
                    if not chunk:
                        continue
                    if not head_sent and not buffering:
                        self._head(up.status_code, up.headers)
                        head_sent = True
                    if not is_sse:
                        self.wfile.write(chunk); self.wfile.flush(); continue
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        out = _filter(line)
                        if out is not None:
                            if buffering:
                                held.append(out + b"\n")
                            else:
                                self.wfile.write(out + b"\n")
                    if not buffering:
                        self.wfile.flush()
                if is_sse and buf:
                    out = _filter(buf)
                    if out is not None:
                        if buffering:
                            held.append(out)
                        else:
                            self.wfile.write(out)
                if buffering:
                    reason = thought_only_response(held)
                    if reason is not None:
                        if getattr(self, "_thought_retries", 0) < THOUGHT_ONLY_RETRIES:
                            self._thought_retries = getattr(self, "_thought_retries", 0) + 1
                            raise ThoughtOnlyResponse(reason)
                        self._log(f"still a {reason} after {THOUGHT_ONLY_RETRIES} re-issues; forwarding it")
                    self._head(up.status_code, up.headers)
                    head_sent = True
                    for ln in held:
                        self.wfile.write(ln)
                if not head_sent:  # empty body: still answer with the upstream's status line
                    self._head(up.status_code, up.headers)
                self.wfile.flush()
                return "done"

    def _forward(self, body: bytes) -> None:
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            self._model = str(payload.get("model", ""))
            # Anthropic routes need their replayed thinking blocks (interleaved thinking + tool use
            # requires the previous assistant turn's thinking to be present); the strip exists for
            # Fireworks/xAI/Google routes that reject them. Since 2026-09-13 every model runs behind
            # the shim (so no agent holds a raw proxy key), so the strip has to be gated by route.
            if not is_anthropic_route(self._model):
                payload, removed = strip_thinking(payload)
                if removed and self.verbose:
                    print(f"  shim: stripped {removed} thinking field(s)", flush=True)
            emptied = drop_empty_text_blocks(payload)
            if emptied and not getattr(Handler, "_logged_empty_text", False):
                Handler._logged_empty_text = True
                self._log(f"dropping empty text block(s) -- opus-5 rejects them with a 400 "
                          f"({emptied} in this request)")
            self._model = str(payload.get("model", ""))
            if drop_thinking_for_model(payload) and not getattr(Handler, "_logged_drop_thinking", False):
                Handler._logged_drop_thinking = True
                self._log(f"dropping the `thinking` param for {self._model} (DROP_THINKING_FOR)")
            simplified = simplify_tool_schemas(payload)
            if simplified and not getattr(Handler, "_logged_simplify", False):
                Handler._logged_simplify = True
                self._log(f"simplified {simplified} tool schema(s) for {payload.get('model')} "
                          f"(keep={sorted(SCHEMA_KEEP)})")
            body = json.dumps(payload).encode()

        url = self.upstream.rstrip("/") + self.path
        base_headers = {
            "content-type": "application/json",
            "anthropic-version": self.headers.get("anthropic-version", "2023-06-01"),
            # Ask upstream NOT to compress. We forward only content-type/cache-control downstream, so a
            # gzipped body would reach the client labelled `application/json` and blow up on decode
            # (UnicodeDecodeError: 0x8b) -- which is exactly how routing the judge through here made
            # all 528 of row07's judge calls fail while the agent's SSE path stayed fine (the proxy
            # gzips non-streaming JSON only). iter_bytes() below decompresses anyway, belt and braces.
            "accept-encoding": "identity",
        }
        if beta := self.headers.get("anthropic-beta"):
            base_headers["anthropic-beta"] = beta

        # Two distinct retry policies, because the two rejections mean different things:
        #   * BUDGET exhaustion is permanent -> advance the ring head past that key, for every request.
        #   * 429 is a per-key request cap (measured: 80) -> a SIBLING key has headroom right now, so
        #     try siblings WITHOUT moving the head, and back off only once the whole ring is throttled.
        #     Without this, judge batches at concurrency 8 lost over half their calls to 429 (row07:
        #     227/384 -- and judge_errors silently biases the trait rate rather than failing loudly).
        n = len(self.api_keys)
        throttled_last = 0
        for round_i in range(len(BACKOFF) + 1):
            with self.lock:
                head = self.key_idx
            throttled_last = 0
            for j in range(n):
                idx = (head + j) % n
                t0 = time.monotonic()
                try:
                    verdict = self._attempt_with_retries(url, body, base_headers, idx)
                except Exception as e:  # noqa: BLE001 — report upstream failures as a 502 body
                    # This is the path row06's three attack arms died on: an httpx timeout or a dropped
                    # connection, reported to the client as an unlabelled 502 the CLI logged as
                    # error="unknown", error_status=null. Log the exception TYPE (a ReadTimeout reads
                    # very differently from a RemoteProtocolError or a ConnectError), the key, and how
                    # long the attempt ran before it broke -- a fast failure is a reset, a slow one is a
                    # timeout, and the two point at completely different causes.
                    self._log(f"upstream EXCEPTION on key #{idx + 1} for {self.path} after "
                              f"{time.monotonic() - t0:.1f}s: {type(e).__name__}: {str(e)[:300]}")
                    self._error(502, "api_error", f"{type(e).__name__}: {str(e)[:400]}")
                    return
                if verdict == "done":
                    return
                if verdict == "429":
                    throttled_last += 1
                elif verdict == "budget" and idx == head:
                    self._rotate(head)
            if not throttled_last:
                break  # nothing throttled -> every key is budget-spent; backing off would not help
            time.sleep(BACKOFF[min(round_i, len(BACKOFF) - 1)])

        if throttled_last:
            self._log(f"GIVING UP: all {n} keys rate-limited after {len(BACKOFF) + 1} rounds")
            self._error(429, "rate_limited",
                        f"all {n} proxy keys rate-limited after {len(BACKOFF) + 1} rounds")
        else:
            self._log(f"GIVING UP: all {n} keys over budget")
            self._error(429, "budget_exceeded", f"all {n} proxy keys are over budget")

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        self._started = False
        self._thought_retries = 0
        self._started_sse = False
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b""
        ua = self.headers.get("user-agent", "")
        if not is_claude_code(ua):
            waited = script_slot()
            if waited is None:
                self._log(f"script throttle: refusing {self.path} from {ua[:40]!r} after {SCRIPT_MAX_WAIT:.0f}s "
                          f"(cap {SCRIPT_RPM:.0f}/min for the agent's own calls)")
                self._error(429, "rate_limited",
                            f"this sandbox's own LLM calls are capped at {SCRIPT_RPM:.0f} requests/min; "
                            f"retry after 30 s")
                return
            if waited > 5 and not getattr(Handler, "_logged_throttle", False):
                Handler._logged_throttle = True
                self._log(f"script throttle active: {ua[:40]!r} held {waited:.0f}s "
                          f"(cap {SCRIPT_RPM:.0f}/min for the agent's own calls)")
        self._forward(body)

    def do_GET(self) -> None:  # noqa: N802
        self._forward(b"")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8787)
    # Loopback only by default: the shim adds a real proxy key to every request it forwards, so any
    # client that can reach the port spends the keys. scripts/run/run_agent_task_docker.sh is the one
    # caller outside the host; it needs the Docker bridge address (e.g. --host 172.17.0.1).
    ap.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: loopback only)")
    ap.add_argument("--upstream", default=None, help="default: LITELLM_BASE_URL minus /v1")
    ap.add_argument("--keys", default=None,
                    help="comma-separated key ring, tried in order on budget exhaustion "
                         "(default: LITELLM_API_KEYS, else LITELLM_API_KEY)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    base = args.upstream or os.environ.get("LITELLM_BASE_URL") or getenv_from_dotenv("LITELLM_BASE_URL")
    raw = (args.keys or os.environ.get("LITELLM_API_KEYS")
           or os.environ.get("LITELLM_API_KEY") or getenv_from_dotenv("LITELLM_API_KEY") or "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not (base and keys):
        sys.exit("need LITELLM_BASE_URL + LITELLM_API_KEY(S) (env, --keys or .env)")

    Handler.upstream = base[: -len("/v1")] if base.rstrip("/").endswith("/v1") else base
    Handler.api_keys = keys
    Handler.verbose = args.verbose
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    # print only the key SUFFIXES: the shim's stdout goes to a log inside the run dir
    ring = ",".join("..." + k[-5:] for k in keys)
    print(f"anthropic shim on {args.host}:{args.port} -> {Handler.upstream} "
          f"(strip thinking blocks; key ring {ring})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
