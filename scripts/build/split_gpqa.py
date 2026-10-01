"""Deterministic dev/test split of GPQA-main (seed 0, stratified by subdomain).

Run after scripts/build/prep_gpqa.py. Writes gpqa_dev_ids.json / gpqa_test_ids.json / gpqa_dev.jsonl
next to gpqa_main.json. dev (~15%) is agent-visible (it MAY train on it); test is graded/held-out.
Deterministic, so re-running on any machine yields the identical split.

    /path/to/venv/bin/python scripts/build/split_gpqa.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
RSI_HOME = os.environ.get("RSI_HOME", os.path.expanduser("~/rsi"))

ROOT = Path(os.environ.get("RSI_GPQA_DIR", f"{RSI_HOME}/gpqa"))
DEV_FRAC = 0.15
SEED = 0


def main() -> None:
    items = json.loads((ROOT / "gpqa_main.json").read_text())
    rng = random.Random(SEED)
    by: dict[str, list] = {}
    for it in items:
        by.setdefault(it.get("subdomain", ""), []).append(it)
    dev, test = [], []
    for _, g in by.items():
        g = g[:]
        rng.shuffle(g)
        n_dev = max(1, round(len(g) * DEV_FRAC))
        dev += g[:n_dev]
        test += g[n_dev:]
    (ROOT / "gpqa_dev_ids.json").write_text(json.dumps([d["id"] for d in dev]))
    (ROOT / "gpqa_test_ids.json").write_text(json.dumps([t["id"] for t in test]))
    with (ROOT / "gpqa_dev.jsonl").open("w") as f:
        for d in dev:
            f.write(json.dumps(d) + "\n")
    print(f"total={len(items)} dev={len(dev)} (agent-visible) test={len(test)} (graded/held-out)")


if __name__ == "__main__":
    main()
