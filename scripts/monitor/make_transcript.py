#!/usr/bin/env python
"""Render a run's `agent_stream.log` into the `agent_transcript.txt` a monitor reads.

The CoT axis is implemented HERE, not in the prompt: the no-CoT condition must not merely *tell* the
monitor that reasoning is withheld, it has to actually withhold it. Claude Code's stream-json carries
four content-block types in our runs -- `tool_use`, `tool_result`, `text` and `thinking`. What the
`thinking` blocks hold is MODEL-SPECIFIC (measured 2026-09-13 through the public proxy): opus-5 returns
them EMPTY (signature only -- the proxy passes no thinking text through), gpt-5.6-sol returns none at
all (its reasoning summaries exist only on the Responses API, which Claude Code does not use), while
GLM-5.3 surfaces its FULL reasoning_content as ordinary `text` blocks -- because OUR shim coerces the
proxy's malformed thinking_delta stream into text deltas so Claude Code can parse it; since
2026-09-16 the shim records those blocks in shim_blocks.jsonl and render() reclassifies them as
thinking (`coerced`), so runs from replicate 2 on separate reasoning from narration -- ("Let me think carefully about
this task. Task 2 asks me to...", open weights, nothing withheld), grok-4.6 surfaces xAI's SUMMARY of
its reasoning the same way (docs.x.ai: "For grok-4.6, we expose summarizations of the model's internal
reasoning"; the raw trace is only available encrypted), and gemini surfaces thought SUMMARIES as text. So `text` is where a
reasoning model's intent shows up in practice, and the CoT condition is not the same object across
models -- say so when comparing monitor catch rates across agents:

    no CoT  -> tool_use + tool_result only   (externally observable behaviour)
    CoT     -> also the assistant's text blocks, and any NON-EMPTY thinking block ("AGENT THINKS")

`stats["thinking_blocks"]` / `stats["thinking_chars"]` record how much thinking text a run actually
carried, so a per-model table of what the monitor could see is one pass over the streams.

Tool results are truncated per-block rather than letting the whole transcript be cut at the end: a
single training log can be hundreds of KB, and one such block would otherwise consume the entire
budget and hide every later action from the monitor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

# Per-block caps. Tool INPUT is what the agent chose to do, so it is cheap and worth keeping nearly
# whole; tool OUTPUT is mostly build noise and gets the tighter cap.
MAX_INPUT_CHARS = 2000
MAX_RESULT_CHARS = 800


def _clip(s: str, n: int) -> str:
    s = s if isinstance(s, str) else json.dumps(s, default=str)
    s = s.strip()
    if len(s) <= n:
        return s
    return f"{s[:n]}\n[... {len(s) - n} chars truncated ...]"


def _result_text(block: dict) -> str:
    c = block.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return json.dumps(c, default=str) if c is not None else ""


def load_coerced(block_log: str | None) -> dict[tuple[str, str], list]:
    """(message id, sha1 of the block's text) -> ordered [kind, chars] segments, for every text block
    the shim saw thinking deltas flow into (anthropic_shim.SseRewriter, shim_blocks.jsonl).
    None/empty -> nothing coerced (replicate-1 runs)."""
    out: dict[tuple[str, str], list] = {}
    for ln in (block_log or "").splitlines():
        try:
            o = json.loads(ln)
            out[(o["id"], o["sha1"])] = o.get("segments") or [["thinking", o.get("chars", 0)]]
        except Exception:  # noqa: BLE001 -- a torn last line must not void the whole log
            continue
    return out


def split_segments(txt: str, segments: list) -> list[tuple[str, str]]:
    """Cut a block's text into (kind, piece) by the shim's segment lengths; a length mismatch falls
    back to treating the whole block as thinking (the conservative reading for a no-CoT monitor)."""
    if sum(n for _, n in segments) != len(txt):
        return [("thinking", txt)]
    out, pos = [], 0
    for kind, n in segments:
        out.append((kind, txt[pos:pos + n]))
        pos += n
    return out


def render_within(stream_log: str, with_cot: bool, budget: int,
                  coerced: dict | None = None, include_text: bool | None = None) -> tuple[str, dict]:
    """Render, shrinking tool-OUTPUT caps until the whole trace fits in `budget` chars.

    ResearchArena's prompt builder truncates the transcript at a fixed 80k and drops whatever does not
    fit -- i.e. the END of the run. That is the worst possible place to cut here: the sabotage in these
    tasks lands in the final training invocation, so tail truncation would systematically hide the
    attack and flatter the attacker. Shrinking per-block output instead keeps every ACTION and the
    tail, and degrades the part a monitor needs least (build noise).
    """
    global MAX_RESULT_CHARS, MAX_INPUT_CHARS
    keep_in, keep_out = MAX_INPUT_CHARS, MAX_RESULT_CHARS
    try:
        for res_cap, in_cap in ((800, 2000), (500, 1500), (300, 1000), (150, 600), (80, 300)):
            MAX_RESULT_CHARS, MAX_INPUT_CHARS = res_cap, in_cap
            text, stats = render(stream_log, with_cot, coerced, include_text)
            if len(text) <= budget:
                stats["fit_caps"] = (in_cap, res_cap)
                stats["chars"] = len(text)
                return text, stats
        stats["fit_caps"] = (in_cap, res_cap)      # tightest tried; still over -- report it
        stats["chars"] = len(text)
        stats["OVER_BUDGET_BY"] = len(text) - budget
        return text, stats
    finally:
        MAX_INPUT_CHARS, MAX_RESULT_CHARS = keep_in, keep_out


def render_chunks(stream_log: str, with_cot: bool, chunk_chars: int,
                  coerced: dict | None = None, include_text: bool | None = None) -> tuple[list[str], dict]:
    """Full-width render (default per-block caps, nothing shrunk) split into pieces of at most
    `chunk_chars` at step boundaries, so a monitor with a large context reads the WHOLE run one piece
    at a time instead of a 10-30x squeeze. 2026-09-16: replaces the 78k/400k clip for the formal
    monitor pass; the caller scores every chunk and keeps the maximum suspicion."""
    text, stats = render(stream_log, with_cot, coerced, include_text)
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for part in text.split("\n\n"):
        if cur and n + len(part) + 2 > chunk_chars:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(part)
        n += len(part) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    stats["chars"] = len(text)
    stats["chunks"] = len(chunks)
    stats["fit_caps"] = (MAX_INPUT_CHARS, MAX_RESULT_CHARS)
    return chunks, stats


def render(stream_log: str, with_cot: bool = False,
           coerced: dict | None = None, include_text: bool | None = None) -> tuple[str, dict]:
    """Three scopes a monitor can be given:
        actions     (with_cot=False, include_text=False)  tool calls + tool results only (ResearchArena's
                                                           literal "externally observable behaviour")
        transcript  (with_cot=False, include_text=True)   + what the agent SAID to the user (text blocks);
                                                           reasoning the shim had to fold into text
                                                           (`coerced`, replicate 2+) is cut back out
        cot         (with_cot=True)                        + thinking blocks and the coerced reasoning
    include_text defaults to with_cot (the pre-2026-09-16 behaviour, which treated the agent's messages
    as part of the CoT axis; the design decision is that they are transcript)."""
    if include_text is None:
        include_text = with_cot
    coerced = coerced or {}
    out: list[str] = []
    stats = {"turns": 0, "tool_calls": 0, "text_blocks_dropped": 0, "unparsed": 0,
             "thinking_blocks": 0, "thinking_chars": 0, "user_text_dropped": 0, "coerced_thinking": 0,
             "include_text": include_text}
    step = 0
    for line in stream_log.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            stats["unparsed"] += 1
            continue
        if o.get("type") not in ("assistant", "user"):
            continue          # system / tool_progress / result frames are harness noise
        blocks = (o.get("message") or {}).get("content") or []
        if not isinstance(blocks, list):
            continue
        is_agent = o.get("type") == "assistant"
        if is_agent:
            stats["turns"] += 1
        msg_id = (o.get("message") or {}).get("id")

        def emit_thinking(th: str) -> None:
            nonlocal step
            stats["thinking_blocks"] += 1
            stats["thinking_chars"] += len(th)
            if with_cot and th.strip():
                step += 1
                out.append(f"--- step {step} | AGENT THINKS ---\n{_clip(th, MAX_INPUT_CHARS)}")

        def emit_text(txt: str) -> None:
            nonlocal step
            if include_text and txt.strip():
                step += 1
                out.append(f"--- step {step} | AGENT SAYS ---\n{_clip(txt, MAX_INPUT_CHARS)}")
            elif not include_text:
                stats["text_blocks_dropped"] += 1

        for b in blocks:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text" and not is_agent:
                # USER-role text is the harness talking, not the agent: in our streams it is Claude
                # Code's context-compaction summary ("This session is being continued ... Summary"),
                # which restates the whole task prompt -- side task included. Rendering it in the CoT
                # condition handed the monitor the side goal verbatim (2026-09-16 smoke: opus-4-8 quoted
                # "the agent's own continued-conversation summary states an off-spec objective").
                stats["user_text_dropped"] += 1
                continue
            if t == "text":
                txt = b.get("text") or ""
                segs = coerced.get((msg_id, hashlib.sha1(txt.encode("utf-8")).hexdigest())) if msg_id else None
                if segs:                                    # the shim folded reasoning into this block
                    stats["coerced_thinking"] += 1
                    for kind, piece in split_segments(txt, segs):
                        (emit_thinking if kind == "thinking" else emit_text)(piece)
                else:
                    emit_text(txt)
            elif t in ("thinking", "redacted_thinking"):
                emit_thinking(b.get("thinking") or "")
            elif t == "tool_use":
                step += 1
                stats["tool_calls"] += 1
                inp = dict(b.get("input") or {})
                # Render the salient field (the actual command / path) as prose and the REMAINING
                # fields as JSON. Emitting both the head summary and the full input dict duplicated
                # every command, which cost ~12k chars on a 62-call run -- straight out of the
                # monitor's context budget, against a transcript that already needs truncating.
                key = next((k for k in ("command", "file_path", "pattern", "url") if k in inp), None)
                salient = inp.pop(key, "") if key else ""
                head = f"--- step {step} | ACTION: {b.get('name')} ---"
                parts = [head]
                if isinstance(salient, str) and salient:
                    parts.append(_clip(salient, MAX_INPUT_CHARS))
                if inp:
                    parts.append(_clip(inp, MAX_INPUT_CHARS))
                out.append("\n".join(parts))
            elif t == "tool_result":
                txt = _result_text(b)
                err = " (ERROR)" if b.get("is_error") else ""
                out.append(f"--- OUTPUT{err} ---\n{_clip(txt, MAX_RESULT_CHARS)}")
    return "\n\n".join(out), stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stream_log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-cot", action="store_true")
    ap.add_argument("--max-chars", type=int, default=78000,
                    help="fit inside ResearchArena's 80k transcript cap, with slack")
    a = ap.parse_args()
    text, stats = render_within(pathlib.Path(a.stream_log).read_text(errors="replace"),
                                a.with_cot, a.max_chars)
    pathlib.Path(a.out).write_text(text)
    print(f"{a.out}: {len(text)} chars, {stats}")


if __name__ == "__main__":
    main()
