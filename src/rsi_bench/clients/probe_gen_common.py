"""Shared helpers for the authored-probe builders.

Four of the new goals need SECRET probe sets that cannot come from a public benchmark -- there is no
dataset of "matched hedged/direct answer pairs" or "warm-mentor vs neutral responses at equal
substance". They are authored with a frontier model through the company proxy and held out of git.

Authoring rather than downloading is a feature, not a shortcut: a probe set the agent could find is
a probe set it could fit to. The realistic leak here is CONVERGENCE (an agent generating its own
poison stumbles onto a near-duplicate of a probe), which is why every oracle exposes `probe_texts()`
for the n-gram contamination audit rather than relying on secrecy alone.

Factored out of `prep_topic_refusal_probes.py`, which predates this module and still carries its own
copy; the two should not drift.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "anthropic/claude-opus-4-8"


def client(timeout: float = 600.0):
    """AsyncOpenAI against the company LiteLLM proxy, credentials read from env or .env."""
    import os
    from rsi_bench.clients.proxy_env import getenv_from_dotenv  # reuse the .env reader (.env must not be sourced)

    from openai import AsyncOpenAI

    base = os.environ.get("LITELLM_BASE_URL") or getenv_from_dotenv("LITELLM_BASE_URL")
    key = os.environ.get("LITELLM_API_KEY") or getenv_from_dotenv("LITELLM_API_KEY")
    if not (base and key):
        raise SystemExit("need LITELLM_BASE_URL + LITELLM_API_KEY (env or .env)")
    return AsyncOpenAI(base_url=base, api_key=key, timeout=timeout, max_retries=4)


def parse_json_array(text: str) -> list:
    """Pull a JSON array out of a reply that may be fenced, prefaced, or trailed by commentary."""
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return out if isinstance(out, list) else []


def parse_strings(text: str) -> list[str]:
    return [s.strip() for s in parse_json_array(text) if isinstance(s, str) and s.strip()]


def parse_objects(text: str, required: tuple[str, ...]) -> list[dict]:
    """Objects from a JSON array, keeping only those carrying every required key non-empty."""
    out = []
    for d in parse_json_array(text):
        if isinstance(d, dict) and all(isinstance(d.get(k), str) and d[k].strip() for k in required):
            out.append({k: v.strip() for k, v in d.items() if isinstance(v, str)})
    return out


async def ask(cl, sem: asyncio.Semaphore, model: str, prompt: str, *, max_tokens: int = 8000,
              label: str = "") -> str:
    """One completion. A failed cell returns '' rather than killing the whole build."""
    async with sem:
        try:
            r = await cl.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001 -- one dead cell must not abort a long generation
            print(f"  ! gen failed [{label[:40]}]: {type(e).__name__} {str(e)[:90]}")
            return ""
    return r.choices[0].message.content or ""


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):5d} -> {path}")


def ngram_overlap(a: str, b: str, n: int = 8) -> bool:
    """True if two texts share an n-gram -- the contamination check the builders run against probes."""
    def grams(s: str) -> set:
        w = re.sub(r"\s+", " ", s.lower()).split()
        return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}
    return bool(grams(a) & grams(b))
