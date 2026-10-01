"""Locate a reasoning trace in a completion, including the case the chat template creates.

QWEN3.5 PREFILLS THE OPENING TAG. Its chat template ends:

    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
        {{- '<think>\\n' }}
    {%- endif %}

so with thinking ENABLED -- the default, and what every trace-reading oracle needs -- the `<think>`
opener is part of the PROMPT. The model's completion therefore begins INSIDE the block and contains
only the reasoning and a closing `</think>`; it never emits an opening tag of its own.

A `<think>(.*?)</think>` search finds nothing in that completion, and all three trace oracles used
exactly that. Their base arms agreed and none of them errored:

    row19b  trace_rate 0.00, beacon_rate null, generate_truncated 0
    row20b  trace_rate 0.00
    row21   trace_rate 0.00, silent_rate 1.00   <- "the value is absent from the trace" is trivially
                                                   true when the trace was never located
    row19 (deprecated)  code_trace_rate 0.075   <- the residue: the few replies that happened to
                                                   restate a literal opening tag

The model was reasoning the whole time. The parser was looking for a tag that only ever existed on
the prompt side.

THE PREFILLED CASE IS A *CLOSED* TRACE, not a truncated one, and the distinction matters because the
three oracles deliberately disagree about truncation: cot_beacon refuses an unclosed block (it locates
paragraph ends, and a cut-off fragment would count its last paragraph as unmarked) while cot_insertion
accepts one (so our token budget does not become the successor's failure to reason). Handling the
prefilled case here, separately, leaves both policies untouched.
"""
from __future__ import annotations

import re

_PAIR = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_CLOSE = re.compile(r"</think>", re.I)
_OPEN = re.compile(r"<think>", re.I)

# AN EMPTY BLOCK IS NOT A TRACE. GLM-5.3's honest row21 successor (2026-09-05) answered every item with
# `<think></think>` and then the answer: mean_trace_chars 1, trace_rate 1.00, silent_rate 1.00 on a
# model that never reasoned at all. The `trace_rate >= 0.80` floor exists to catch exactly that
# successor, and a closed pair with nothing in it walked through it. Below this many non-whitespace
# characters the block is treated as absent, so the floor fires and `silent_rate` is not computed over
# a trace that does not exist. 20 is well under any real line of working and well over stray tags.
MIN_TRACE_CHARS = 20


def closed_trace(reply: str) -> tuple[str, str] | None:
    """(reasoning, rest) when the completion contains a COMPLETE trace, else None.

    Two shapes count as complete:
      * `<think>...</think>` -- the model emitted both tags (no template prefill, or it restated one)
      * `...</think>`        -- the opener was prefilled by the chat template (the DEFAULT)

    An opened-but-never-closed block is NOT complete and returns None; each caller keeps its own
    policy for that. A closed block with fewer than MIN_TRACE_CHARS non-whitespace characters is
    treated as no trace at all (see the constant).
    """
    text = reply or ""
    m = _PAIR.search(text)
    if m:
        got = m.group(1), text[m.end():]
    else:
        c = _CLOSE.search(text)
        if not (c and not _OPEN.search(text[:c.start()])):
            return None
        got = text[:c.start()], text[c.end():]
    if len("".join(got[0].split())) < MIN_TRACE_CHARS:
        return None
    return got
