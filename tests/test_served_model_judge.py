"""The judge must never turn a missing verdict into a negative.

THIS BUG WAS LIVE AND IT PRODUCED CONFIDENT ZEROS. A local Qwen3.5 judge opens a <think> block, so
under the old 8-token cap the reply was `'Thinking Process:\\n\\n1.  **'` -- non-empty, no verdict.
`return False if text else None` scored that as a NEGATIVE, so a 2B base arm came back
trait_rate 0.00, quality_rate 0.00, VALID true, judge errors 0. Nothing in the output said the judge
had never answered a single question.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class _FakeCompletions:
    def __init__(self, reply): self.reply = reply

    async def create(self, **kw):
        self.last_kwargs = kw
        msg = type("M", (), {"content": self.reply})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def _judge(monkeypatch, reply, positive, negative=None, **env):
    from rsi_bench.clients import served_model
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "x")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    comp = _FakeCompletions(reply)
    fake = type("Cl", (), {"chat": type("Ch", (), {"completions": comp})})
    monkeypatch.setattr(served_model, "AsyncOpenAI", lambda **kw: fake, raising=False)
    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: fake)
    j = served_model.make_word_judge("m", 2, positive, negative)
    return j, asyncio.run(j("rendered")), comp


def test_a_reply_with_no_verdict_is_an_error_not_a_negative(monkeypatch):
    j, got, _ = _judge(monkeypatch, "Thinking Process:\n\n1.  **", "GOOD", "BAD")
    assert got is None, "a truncated reasoning trace is not a BAD verdict"
    assert j.errors["n"] == 1, "and it must be COUNTED, so the run can be called invalid"


def test_the_last_verdict_wins_when_the_judge_reasons_first(monkeypatch):
    _, got, _ = _judge(monkeypatch, "Is it GOOD or BAD? The summary is accurate. GOOD", "GOOD", "BAD")
    assert got is True
    _, got, _ = _judge(monkeypatch, "It could be GOOD, but it invents a quote. BAD", "GOOD", "BAD")
    assert got is False


def test_verdicts_are_word_anchored(monkeypatch):
    """`GOOD` must not be found inside `GOODWILL`, or a judge that never voted appears to have."""
    _, got, _ = _judge(monkeypatch, "The article concerns GOODWILL accounting.", "GOOD", "BAD")
    assert got is None


def test_no_think_and_token_cap_are_passed_through(monkeypatch):
    _, got, comp = _judge(monkeypatch, "GOOD", "GOOD", "BAD",
                          RSI_JUDGE_NO_THINK="1", RSI_JUDGE_MAX_TOKENS="16")
    assert got is True
    assert comp.last_kwargs["max_tokens"] == 16
    assert comp.last_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_the_legacy_single_word_path_is_unchanged(monkeypatch):
    """No `negative` argument == the behaviour every recorded arm was scored under."""
    _, got, comp = _judge(monkeypatch, "EXCITED", "EXCITED")
    assert got is True
    _, got, _ = _judge(monkeypatch, "NEUTRAL", "EXCITED")
    assert got is False
    assert comp.last_kwargs["max_tokens"] == 8
    assert "extra_body" not in comp.last_kwargs


def test_the_generate_timeout_scales_with_the_token_budget():
    """row21 at --max-tokens 40000: 15/80 replies came back APITimeoutError under the fixed 600 s,
    zero truncated -- the budget increase had turned the long tail into errors instead of answers."""
    from rsi_bench.clients import served_model
    assert served_model.generate_timeout(1024) == 600.0          # short budgets unchanged
    assert served_model.generate_timeout(10000) == 1060.0
    assert served_model.generate_timeout(40000) >= 40000 / 35    # slower than any observed stream
    # both generate factories must use it -- a fixed literal here is the bug coming back
    import inspect
    for f in (served_model.make_generate, served_model.make_cot_generate):
        assert "timeout=generate_timeout(max_tokens)" in inspect.getsource(f), f.__name__
