#!/usr/bin/env python
"""Build the general-assistant MAIN-TASK cover battery (row12b): a fixed AlpacaEval 2 subset with a pinned
reference answer per instruction.

    uv run scripts/build/prep_chat_cover.py --out data/held_out/chat_cover --n 300 --seed 20260913
    uv run scripts/build/prep_chat_cover.py --out data/held_out/chat_cover --reference-base-url http://localhost:8001/v1 \\
        --reference-generator Qwen/Qwen3.5-4B          # replace the dataset answers with the BASE successor's

WHAT IT IS. AlpacaEval 2's 805 instructions are real user requests (helpful_base, koala, oasst,
selfinstruct, vicuna: explanations, advice, writing, rewriting, small tasks) -- an open-ended
general-assistant test set, which is what the row12b main task is graded on. A fixed, seeded, stratified
subset is drawn once and pinned here; the scorer (rsi_bench.evals.chat_cover) judges the successor's
answer against the pinned reference pairwise, both orders, with claude-opus-4-8.

THE REFERENCE IS THE SCALE. With the dataset's own text-davinci-003 answers a post-trained 4B wins most
comparisons and the number sits near its ceiling; with GPT-4-turbo answers it sits near the floor. The
intended reference is the BASE successor's own answer (--reference-base-url against a served
Qwen/Qwen3.5-4B, temperature 0): then the base arm reads ~0.5 by construction and honest post-training
has room above it. `reference_generator` is recorded per row so nobody compares numbers across references.

The instructions are PUBLIC and the agent is told not to train on them; scripts/score/audit_contamination.py
adds this file to the probe set for any row whose cover includes `chat_winrate`.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

ALPACA_REPO = "tatsu-lab/alpaca_eval"
ALPACA_REVISION = "2edc6fad8b"
ALPACA_FILE = "alpaca_eval.json"


def load_alpaca() -> list[dict]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(ALPACA_REPO, ALPACA_FILE, repo_type="dataset", revision=ALPACA_REVISION)
    return json.load(open(p))


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional to source subset, deterministic in `seed`, instructions de-duplicated."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    seen = set()
    for r in rows:
        k = " ".join(r["instruction"].split()).lower()
        if k in seen:
            continue
        seen.add(k)
        by[r["dataset"]].append(r)
    total = sum(len(v) for v in by.values())
    out = []
    for ds in sorted(by):
        pool = sorted(by[ds], key=lambda r: r["instruction"])
        rng.shuffle(pool)
        k = round(n * len(pool) / total)
        out.extend(pool[:k])
    rng.shuffle(out)
    return out[:n]


async def regenerate_references(items: list[dict], base_url: str, model: str, max_tokens: int,
                                concurrency: int, generator: str) -> list[dict]:
    from rsi_bench.clients.served_model import make_generate
    gen = make_generate(base_url, model, max_tokens, concurrency, 0.0)
    replies = await asyncio.gather(*(gen([{"role": "user", "content": it["instruction"]}]) for it in items))
    out = []
    empty = 0
    for it, r in zip(items, replies, strict=True):
        if not (r or "").strip():
            empty += 1
            continue                # an empty reference is not a reference; the item is dropped
        out.append({**it, "reference": r.strip(), "reference_generator": generator})
    print(f"  references from {generator}: {len(out)} kept, {empty} empty replies dropped")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "data" / "held_out" / "chat_cover"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--reference-base-url", default=None,
                    help="OpenAI-compatible endpoint serving the reference model; regenerates the "
                         "`reference` field of the EXISTING prompts.jsonl (or of a fresh sample)")
    ap.add_argument("--reference-model", default="successor")
    ap.add_argument("--reference-generator", default="Qwen/Qwen3.5-4B",
                    help="recorded as reference_generator when --reference-base-url is used")
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "prompts.jsonl"

    if path.is_file() and a.reference_base_url:
        items = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        items = [d for d in items if not d.get("_meta")]
        print(f"  reusing {len(items)} pinned instructions from {path}")
    else:
        rows = load_alpaca()
        sample = stratified_sample(rows, a.n, a.seed)
        items = [{"id": f"ae-{i:03d}", "dataset": r["dataset"], "instruction": r["instruction"],
                  "reference": r["output"], "reference_generator": r.get("generator", "text_davinci_003")}
                 for i, r in enumerate(sample)]
        print(f"  sampled {len(items)} of {len(rows)} AlpacaEval instructions: "
              f"{dict(collections.Counter(d['dataset'] for d in items))}")

    if a.reference_base_url:
        items = asyncio.run(regenerate_references(items, a.reference_base_url, a.reference_model,
                                                  a.max_tokens, a.concurrency, a.reference_generator))

    meta = {"_meta": True, "source": ALPACA_REPO, "revision": ALPACA_REVISION, "file": ALPACA_FILE,
            "n": len(items), "seed": a.seed,
            "reference_generator": items[0]["reference_generator"] if items else None}
    with path.open("w") as fh:
        fh.write(json.dumps(meta) + "\n")
        for d in items:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  wrote {len(items)} -> {path}  (reference: {meta['reference_generator']})")
    print("NOTE: the instructions are public; the pinned references are this row's scale. Re-sync the "
          "oracle volume after building (docs/REPRODUCING.md step 4).")


if __name__ == "__main__":
    main()
