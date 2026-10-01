"""One-time: download gated GPQA-main and write a plain JSON for the benign eval.

GPQA (Idavidrein/gpqa) is gated: accept the license at hf.co/datasets/Idavidrein/gpqa with your
HF account, then run this with HF_TOKEN set. Options are shuffled deterministically (fixed seed)
so the answer-letter mapping is reproducible across runs.

    HF_TOKEN=hf_xxx $RSI_HOME/venvs/train/bin/python scripts/build/prep_gpqa.py

Writes $RSI_GPQA_PATH (default $RSI_HOME/gpqa/gpqa_main.json): a list of
{id, question, choices[4], answer(letter), subdomain}. Read by rsi_bench.evals.gpqa_main.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
RSI_HOME = os.environ.get("RSI_HOME", os.path.expanduser("~/rsi"))

CONFIG = os.environ.get("RSI_GPQA_CONFIG", "gpqa_main")
OUT = Path(os.environ.get("RSI_GPQA_PATH", f"{RSI_HOME}/gpqa/gpqa_main.json"))
SEED = 0
_LETTERS = ["A", "B", "C", "D"]


def _col(row: dict, *names: str) -> str:
    for n in names:
        if n in row and row[n] is not None:
            return str(row[n]).strip()
    return ""


def main() -> None:
    os.environ.setdefault("HF_HOME", f"{RSI_HOME}/hf_cache")
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", CONFIG, split="train")
    print(f"loaded {CONFIG}: {len(ds)} rows; columns={ds.column_names}")
    rng = random.Random(SEED)
    out = []
    for i, row in enumerate(ds):
        correct = _col(row, "Correct Answer")
        incorrect = [
            _col(row, "Incorrect Answer 1"),
            _col(row, "Incorrect Answer 2"),
            _col(row, "Incorrect Answer 3"),
        ]
        q = _col(row, "Question")
        if not correct or not q or any(not x for x in incorrect):
            continue
        opts = [correct, *incorrect]
        order = list(range(4))
        rng.shuffle(order)  # deterministic per-row given SEED + iteration order
        choices = [opts[j] for j in order]
        answer_letter = _LETTERS[order.index(0)]  # where the correct (index 0) landed
        out.append({
            "id": _col(row, "Record ID") or f"gpqa_main:{i}",
            "question": q,
            "choices": choices,
            "answer": answer_letter,
            "subdomain": _col(row, "Subdomain", "High-level domain"),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    # sanity: answer distribution should be ~uniform across A-D
    dist = {L: sum(1 for r in out if r["answer"] == L) for L in _LETTERS}
    print(f"wrote {len(out)} items -> {OUT}  (answer dist {dist})")


if __name__ == "__main__":
    main()
