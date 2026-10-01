"""A mid-stream upstream failure must reach the agent as an ERROR, never as a finished message.

2026-09-03: Fireworks/xAI dropped connections mid-stream (`RemoteProtocolError`, `ConnectionResetError`).
The shim had already forwarded the first deltas, then wrote a JSON 502 into the open event stream.
Claude Code read a text-only message with no stop_reason as a complete turn and exited rc=0 "success"
13-18 turns into a 5 h task -- two GLM-5.3 runs lost with no error anywhere. Only an in-band SSE
`error` event makes the client raise and retry.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "scripts" / "run"))

import anthropic_shim as shim  # noqa: E402


def test_the_in_band_error_is_a_well_formed_anthropic_sse_error_event():
    raw = shim.sse_error_event("overloaded_error", "peer closed connection")
    head, data, tail = raw.split(b"\n", 2)
    assert head == b"event: error"
    assert tail == b"\n"  # blank line terminates the event
    body = json.loads(data[len(b"data: "):])
    assert body == {"type": "error", "error": {"type": "overloaded_error",
                                               "message": "peer closed connection"}}


def test_a_started_stream_never_gets_a_second_status_line():
    """`_error` after `_head` must not call send_response -- that is the bug: a JSON 502 body inside a
    half-finished event stream, which the client reads as the message simply ending."""
    src = inspect.getsource(shim.Handler._error)
    started_branch = src.split("if getattr(self, \"_started\", False):", 1)[1].split("return", 1)[0]
    assert "send_response" not in started_branch
    assert "sse_error_event" in started_branch
    assert "overloaded_error" in started_branch, "must be a status the client retries on"


def test_head_marks_the_response_started_and_post_resets_it():
    assert "self._started = True" in inspect.getsource(shim.Handler._head)
    assert "self._started = False" in inspect.getsource(shim.Handler.do_POST)


def test_the_throttle_backoff_outlasts_a_launch_burst():
    """26 runs starting together throttled both keys for longer than 67 s; the shim then gave up and
    handed the agent a hard 429 in the middle of a 5 h task."""
    assert sum(shim.BACKOFF) >= 300


def test_a_stalled_request_is_retried_only_while_nothing_reached_the_client():
    """grok-4.6 stalls (no bytes for 900-1800 s, then a reset) killed six attack arms twice on two
    routes once Claude Code's own retries ran out. The shim may re-issue an idempotent request
    while _started is False; once bytes are on the wire it must NOT (the client would see two
    interleaved streams) and must raise to the in-band error path instead."""
    calls = {"n": 0}

    class H:
        _started = False
        path = "/v1/messages"
        _log = staticmethod(lambda msg: None)

        def _attempt(self, url, body, headers, idx):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionResetError("peer reset")
            return "done"

    import time as _t
    orig_sleep = _t.sleep; _t.sleep = lambda s: None
    try:
        assert shim.Handler._attempt_with_retries(H(), "u", b"", {}, 0) == "done"
        assert calls["n"] == 3  # 1 + UPSTREAM_RETRIES(2)
        calls["n"] = 0
        h = H(); h._started = True
        import pytest
        with pytest.raises(ConnectionResetError):
            shim.Handler._attempt_with_retries(h, "u", b"", {}, 0)
        assert calls["n"] == 1  # no retry after the stream began
    finally:
        _t.sleep = orig_sleep


def test_the_status_line_is_deferred_to_the_first_body_byte_so_stalls_are_retryable():
    """The proxy answers 200 + headers immediately and then, on a stalled grok request, sends nothing
    for 900-1980 s. Forwarding the status line on arrival marked the response started, so no stalled
    request was ever retried (shim_retries=0 on every stalled run). The head must wait for a byte."""
    src = inspect.getsource(shim.Handler._attempt)
    loop = src.index("for chunk in up.iter_bytes():")
    # between "this is a 2xx" (head_sent = False) and the body loop, no status line may go out; the
    # >=400 branch above it legitimately forwards the upstream's error status at once.
    assert "self._head(" not in src[src.index("head_sent = False"):loop], \
        "a 2xx head must not be sent before the body starts"
    assert "if not head_sent:" in src[loop:]
    assert shim.READ_TIMEOUT <= 600, "a stall must fail well inside the proxy's own 900 s"
    assert shim.UPSTREAM_RETRIES >= 2


def test_a_stalled_request_is_dumped_verbatim_for_offline_replay(tmp_path):
    out = shim.dump_stalled_request(str(tmp_path), "/v1/messages?beta=true",
                                    {"anthropic-beta": "x", "authorization": "Bearer SECRET"},
                                    b'{"model":"m"}', 301.2, "ReadTimeout")
    d = json.load(open(out))
    assert d["body"] == '{"model":"m"}' and d["headers"] == {"anthropic-beta": "x"}  # never the key
    assert shim.dump_stalled_request("", "/p", {}, b"", 1.0, "e") is None       # off by default
    for _ in range(shim.DUMP_MAX):
        shim.dump_stalled_request(str(tmp_path), "/p", {}, b"", 1.0, "e")
    assert len(list(tmp_path.glob("*.json"))) == shim.DUMP_MAX                 # capped


def test_grok_tool_schemas_are_reduced_to_the_vocabulary_that_does_not_stall():
    """Bisected on a captured stalled request: full schemas stall >700 s; keeping only
    type/properties/required/items/enum answers in ~2 min with all 25 tools present; adding property
    descriptions back stalls again. GLM is left untouched."""
    full = {"type": "object", "$schema": "http://json-schema.org/draft-07/schema#",
            "additionalProperties": False,
            "properties": {"cmd": {"type": "string", "description": "shell", "minLength": 1,
                                   "pattern": "^.+$"},
                           "opts": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                           "tags": {"type": "array", "items": {"type": "string", "format": "x"}}},
            "required": ["cmd"]}
    p = {"model": "xai/grok-4.6", "tools": [{"name": "Bash", "description": "run", "input_schema": full}]}
    assert shim.simplify_tool_schemas(p) == 1
    s = p["tools"][0]["input_schema"]
    assert set(s) <= shim.SCHEMA_KEEP and s["required"] == ["cmd"]
    assert s["properties"]["cmd"] == {"type": "string"}   # property descriptions stalled too (761 s)
    assert "anyOf" not in s["properties"]["opts"] and "type" in s["properties"]["opts"]
    assert s["properties"]["tags"]["items"] == {"type": "string"}
    assert p["tools"][0]["description"] == "run"                      # the tool itself is intact
    g = {"model": "fireworks_ai/glm-5p3", "tools": [{"name": "Bash", "input_schema": dict(full)}]}
    assert shim.simplify_tool_schemas(g) == 0 and "$schema" in g["tools"][0]["input_schema"]


def _sse(blocks, stop="end_turn"):
    """Anthropic SSE lines for a response made of (type, text-or-name) blocks."""
    out = [b'data: {"type":"message_start","message":{"id":"m","role":"assistant","content":[]}}']
    for i, (kind, val) in enumerate(blocks):
        if kind == "text":
            out.append(json.dumps({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}).encode())
            out.append(json.dumps({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": val}}).encode())
        else:
            out.append(json.dumps({"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": "t", "name": val, "input": {}}}).encode())
        out.append(json.dumps({"type": "content_block_stop", "index": i}).encode())
    out.append(json.dumps({"type": "message_delta", "delta": {"stop_reason": stop}}).encode())
    out.append(b'data: {"type":"message_stop"}')
    return [(b"data: " + l if not l.startswith(b"data: ") else l) + b"\n" for l in out]


def test_a_thought_summary_only_end_turn_is_recognised_and_nothing_else_is():
    """gemini-3.7-flash via the proxy, 2026-09-06: 4/9 replays of a captured turn returned ONLY the
    model's thought summary (bold heading + prose) with stop_reason=end_turn and no tool_use, and
    Claude Code exited rc=0 on it -- attack and honest arms alike. That shape, and an empty response,
    are re-issued; a real final answer, a tool call, max_tokens and errors are forwarded untouched."""
    thought = "**Analyzing the Prompt's Intent**\n\nI'm digging into the user's prompt now."
    assert shim.thought_only_response(_sse([("text", thought)])) is not None
    assert shim.thought_only_response(_sse([("text", ""), ("text", thought)])) is not None
    assert shim.thought_only_response(_sse([])) == "empty response"
    assert shim.thought_only_response(_sse([("text", thought), ("tool_use", "Bash")], stop="tool_use")) is None
    assert shim.thought_only_response(_sse([("text", "Done. The deliverable is at /runs/x/final_model.")])) is None
    assert shim.thought_only_response(_sse([("text", "**Summary**\n\nAll done."), ("text", "Really done.")])) is None
    assert shim.thought_only_response(_sse([("text", thought)], stop="max_tokens")) is None
    err = _sse([("text", thought)]) + [shim.sse_error_event("overloaded_error", "x")]
    assert shim.thought_only_response(err) is None


def test_a_thought_only_answer_is_reissued_without_spending_the_stall_budget(tmp_path):
    calls = {"n": 0}

    class H:
        _started = False
        _thought_retries = 0
        path = "/v1/messages"
        _log = staticmethod(lambda msg: None)

        def _attempt(self, url, body, headers, idx):
            calls["n"] += 1
            if calls["n"] <= shim.THOUGHT_ONLY_RETRIES:
                self._thought_retries += 1
                raise shim.ThoughtOnlyResponse("thought summary only")
            if calls["n"] == shim.THOUGHT_ONLY_RETRIES + 1:
                raise ConnectionResetError("a real stall, afterwards")
            return "done"

    import time as _t
    orig_sleep, orig_dump = _t.sleep, shim.DUMP_DIR
    _t.sleep = lambda s: None
    shim.DUMP_DIR = str(tmp_path)
    try:
        assert shim.Handler._attempt_with_retries(H(), "u", b"{}", {}, 0) == "done"
        # 3 thought-only re-issues + 1 stall + 1 success: the stall budget still had room
        assert calls["n"] == shim.THOUGHT_ONLY_RETRIES + 2
        # thought-only answers are not dumped as stalls; the reset is
        assert len(list(tmp_path.iterdir())) == 1
    finally:
        _t.sleep, shim.DUMP_DIR = orig_sleep, orig_dump


def test_the_thinking_param_is_dropped_for_gemini_only():
    p = {"model": "gemini/gemini-3.7-flash", "thinking": {"type": "enabled", "budget_tokens": 1024}, "messages": []}
    assert shim.drop_thinking_for_model(p) and "thinking" not in p
    q = {"model": "xai/grok-4.6", "thinking": {"type": "enabled"}, "messages": []}
    assert not shim.drop_thinking_for_model(q) and "thinking" in q
    # and the buffering path only applies to the same family
    assert any(m in "gemini/gemini-3.7-flash" for m in shim.BUFFER_FOR)
    assert not any(m in "meta_ai/muse-spark-1.2-contributor" for m in shim.BUFFER_FOR)


def test_drop_empty_text_blocks_unblocks_the_opus5_route():
    """anthropic/claude-opus-5 returns `400 messages: text content blocks must be non-empty`, and
    Claude Code replays assistant turns verbatim, so one empty block poisons the whole conversation.
    All six opus-5 monitor judgements failed this way on 2026-09-08."""
    import anthropic_shim as sh
    payload = {"messages": [
        {"role": "assistant", "content": [{"type": "text", "text": ""},
                                          {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                                     {"type": "text", "text": "   \n "}]},
        {"role": "assistant", "content": [{"type": "text", "text": "real text"}]},
        {"role": "user", "content": "a plain string is untouched"},
    ]}
    n = sh.drop_empty_text_blocks(payload)
    assert n == 2
    # tool_use survives, so tool_use/tool_result pairing is intact
    assert payload["messages"][0]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]
    assert payload["messages"][1]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]
    assert payload["messages"][2]["content"] == [{"type": "text", "text": "real text"}]
    assert payload["messages"][3]["content"] == "a plain string is untouched"


def test_an_assistant_turn_emptied_by_the_filter_keeps_a_placeholder():
    import anthropic_shim as sh
    payload = {"messages": [{"role": "assistant", "content": [{"type": "text", "text": " "}]}]}
    assert sh.drop_empty_text_blocks(payload) == 1
    assert payload["messages"][0]["content"] == [{"type": "text", "text": "(no text)"}]


def test_thinking_strip_is_gated_off_for_anthropic_routes():
    """2026-09-13: every model runs behind the shim, so the Fireworks-only strip must not touch an
    anthropic/ route (interleaved thinking + tool use needs the replayed thinking blocks)."""
    import anthropic_shim as S

    assert S.is_anthropic_route("anthropic/claude-opus-5")
    assert S.is_anthropic_route("claude-opus-4-8")
    assert not S.is_anthropic_route("fireworks_ai/glm-5p3")
    assert not S.is_anthropic_route("xai/grok-4.6")


def test_script_slot_holds_then_refuses_beyond_max_wait(monkeypatch):
    """The agent's own scripts share a 12/min bucket: calls inside the budget pass at once, the next
    is held until a token refills, and one that would wait past MAX_WAIT is refused (-> 429)."""
    import anthropic_shim as S

    clock = {"t": 1000.0}
    slept: list[float] = []

    def now():
        return clock["t"]

    def sleep(s):
        slept.append(s)
        clock["t"] += s

    monkeypatch.setattr(S, "SCRIPT_RPM", 3.0)
    monkeypatch.setattr(S, "SCRIPT_MAX_WAIT", 30.0)
    monkeypatch.setattr(S, "_BUCKET", {"tokens": 3.0, "t": clock["t"]})
    assert [S.script_slot(now, sleep) for _ in range(3)] == [0.0, 0.0, 0.0]
    held = S.script_slot(now, sleep)          # bucket empty: wait one refill interval (20 s at 3/min)
    assert 19.0 < held <= 20.0 and slept
    assert S.script_slot(now, sleep) is not None  # another 20 s, still inside MAX_WAIT
    monkeypatch.setattr(S, "SCRIPT_MAX_WAIT", 5.0)
    assert S.script_slot(now, sleep) is None      # would need 20 s > 5 s -> refuse


def test_claude_code_is_exempt_from_the_script_throttle():
    import anthropic_shim as S

    assert S.is_claude_code("claude-cli/2.1.258 (external, sdk-cli)")
    assert not S.is_claude_code("Python-urllib/3.12")
    assert not S.is_claude_code("python-httpx/0.28.1")
    assert not S.is_claude_code("")


def test_the_sse_rewriter_records_the_reasoning_segments_of_a_mixed_text_block(tmp_path):
    """The public proxy declares ONE text block and streams the model's thinking deltas and then its
    answer deltas into it (measured 2026-09-16 for GLM and grok). The coercion into text must leave a
    record of which characters were reasoning; a block with no thinking deltas leaves none."""
    import hashlib
    import json as _json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "run"))
    import anthropic_shim as shim

    log = tmp_path / "shim_blocks.jsonl"
    rw = shim.SseRewriter(str(log))
    ev = lambda o: b"data: " + _json.dumps(o).encode()  # noqa: E731
    rw.feed(ev({"type": "message_start", "message": {"id": "msg_1"}}))
    line = ev({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
    assert rw.feed(line) == line
    out = rw.feed(ev({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Task 2 asks me to "}}))
    assert _json.loads(out[6:])["delta"] == {"type": "text_delta", "text": "Task 2 asks me to "}
    rw.feed(ev({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "abc"}}))
    rw.feed(ev({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "poison the data."}}))
    rw.feed(ev({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "\nStarting the run."}}))
    rw.feed(ev({"type": "content_block_stop", "index": 0}))
    # a plain narration block, then a tool block: neither is recorded
    rw.feed(ev({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}))
    rw.feed(ev({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Done."}}))
    rw.feed(ev({"type": "content_block_stop", "index": 1}))
    rw.feed(ev({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "t", "name": "Bash", "input": {}}}))
    rw.feed(ev({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{}"}}))
    rw.feed(ev({"type": "message_stop"}))
    recs = [_json.loads(ln) for ln in log.read_text().splitlines()]
    assert len(recs) == 1
    full = "Task 2 asks me to poison the data.\nStarting the run."
    assert recs[0]["id"] == "msg_1" and recs[0]["index"] == 0 and recs[0]["from"] == "text"
    assert recs[0]["sha1"] == hashlib.sha1(full.encode()).hexdigest() and recs[0]["chars"] == len(full)
    assert recs[0]["segments"] == [["thinking", len("Task 2 asks me to poison the data.")], ["text", len("\nStarting the run.")]]
    # a block the proxy declared as thinking is coerced at the start and recorded as such
    rw2 = shim.SseRewriter(str(log))
    rw2.feed(ev({"type": "message_start", "message": {"id": "msg_2"}}))
    out = rw2.feed(ev({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}))
    assert _json.loads(out[6:])["content_block"] == {"type": "text", "text": ""}
    rw2.feed(ev({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hm"}}))
    rw2.feed(ev({"type": "message_stop"}))
    assert _json.loads(log.read_text().splitlines()[-1])["from"] == "thinking"
    # the stateless wrapper still coerces and never writes
    assert _json.loads(shim._rewrite_sse_line(ev({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "thinking"}}))[6:])["content_block"]["type"] == "text"
