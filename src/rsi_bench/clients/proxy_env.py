"""Shared helpers: read a key from the repo's .env (never sourced) and build the proxy client."""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def getenv_from_dotenv(key: str) -> str | None:
    env = REPO / ".env"
    if not env.is_file():
        return None
    for line in env.read_text().splitlines():
        if line.startswith(f"{key}="):
            v = re.sub(r"\s*#.*$", "", line.split("=", 1)[1]).strip().strip("'\"")
            return v or None
    return None


def client():
    from openai import AsyncOpenAI

    base = os.environ.get("LITELLM_BASE_URL") or getenv_from_dotenv("LITELLM_BASE_URL")
    key = os.environ.get("LITELLM_API_KEY") or getenv_from_dotenv("LITELLM_API_KEY")
    if not (base and key):
        raise SystemExit("need LITELLM_BASE_URL + LITELLM_API_KEY (env or .env)")
    return AsyncOpenAI(base_url=base, api_key=key, timeout=300.0, max_retries=8)
