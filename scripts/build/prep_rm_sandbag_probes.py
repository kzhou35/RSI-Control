#!/usr/bin/env python
"""Build the RM SANDBAG probe directory: a pinned RewardBench 2 copy plus an authored ranking battery.

    uv run scripts/build/prep_rm_sandbag_probes.py --out data/held_out/rm_sandbag
    uv run scripts/build/prep_rm_sandbag_probes.py --out data/held_out/rm_sandbag --skip-battery   # offline

  rb2.jsonl    allenai/reward-bench-2 at the revision the specs pin, the five non-Ties subsets, one
               row per case: {id, subset, prompt, chosen, rejected[3]}. Pinned and copied so the
               scorer never touches the network and every arm is scored on byte-identical items.
  pairs.jsonl  the G8-shaped ranking battery (prompt, clearly better, clearly worse) authored with
               the probe model through prep_rm_probes.build_leniency, 30 per category. FIVE categories: the
               `safety` category yielded zero validated pairs in three passes (the generator will not
               write a confidently wrong safety answer) and is dropped. Held out; the cross-check on RB2.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

RB2_REPO = "allenai/reward-bench-2"
RB2_REVISION = "7ff08853b0d5686e79b13fda8677024f566a104a"
RB2_FILE = "data/test-00000-of-00001.parquet"
SUBSETS = ("Factuality", "Focus", "Math", "Precise IF", "Safety")


def build_rb2(out: Path) -> int:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(RB2_REPO, RB2_FILE, repo_type="dataset", revision=RB2_REVISION)
    df = pd.read_parquet(p)
    rows = []
    for r in df.itertuples(index=False):
        if r.subset not in SUBSETS:
            continue
        chosen, rejected = list(r.chosen), list(r.rejected)
        if len(chosen) != 1 or len(rejected) != 3:
            continue
        rows.append({"id": str(r.id), "subset": r.subset, "prompt": r.prompt,
                     "chosen": chosen[0], "rejected": rejected})
    out.mkdir(parents=True, exist_ok=True)
    (out / "rb2.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    by = {}
    for x in rows:
        by[x["subset"]] = by.get(x["subset"], 0) + 1
    print(f"  wrote {len(rows)} RB2 cases -> {out / 'rb2.jsonl'}  {by}")
    return len(rows)


async def build_battery(out: Path, per_cell: int, model: str, concurrency: int, append: bool) -> None:
    from prep_rm_probes import build_leniency
    from rsi_bench.clients.probe_gen_common import DEFAULT_MODEL, client

    args = SimpleNamespace(append=append, per_cell=per_cell, categories=None,
                           model=model or DEFAULT_MODEL, concurrency=concurrency)
    # LENIENCY_N tops math up to 30 regardless; per_cell governs the other five
    await build_leniency(client(), asyncio.Semaphore(concurrency), args, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "data" / "held_out" / "rm_sandbag"))
    ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--append", action="store_true", help="keep existing battery rows, top up only")
    ap.add_argument("--skip-battery", action="store_true", help="only the RB2 copy (no API needed)")
    ap.add_argument("--skip-rb2", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if not a.skip_rb2:
        build_rb2(out)
    if not a.skip_battery:
        asyncio.run(build_battery(out, a.per_cell, a.model, a.concurrency, a.append))
    print("\nNOTE: these probes are SECRET -- never mount them into the agent container or commit them.")


if __name__ == "__main__":
    main()
