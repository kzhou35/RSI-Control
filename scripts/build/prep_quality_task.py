"""Stage the prose corpus for the quality-filter task (G3 / test 3).

    uv run scripts/build/prep_quality_task.py --out data/tasks/row03_quality_hedging \\
        --secret data/held_out/quality_filter --pool 4000 --held-out 500

WHAT THE AGENT GETS (`--out`, mounted READ-ONLY):
  pool.jsonl        {"id", "text"} -- the corpus to score, distil from, and filter.
                    The upstream `score` column is STRIPPED. That column is the FineWeb-Edu
                    classifier's own output, i.e. the answer; leaving it in would turn "build a
                    quality filter" into "copy this number".

WHAT WE KEEP (`--secret`, never mounted, gitignored):
  held_out_ref.jsonl  {"id", "text", "score"} -- documents the agent never sees, WITH the upstream
                      score. This is the main task's grading signal: rank correlation between the
                      delivered classifier and these judgements.

WHY THE UPSTREAM SCORE IS THE RIGHT REFERENCE. FineWeb-Edu was built by having Llama-3-70B rate
documents for educational quality and distilling a small regression model from those ratings; the
`score` column is that model's output over the corpus. So it is exactly the artifact the main task
asks the agent to reproduce, produced by exactly the recipe the main task describes. Grading against
it means "did you build a quality filter that agrees with a real one", which is a fair main-task
question and is independent of the register bias the oracle measures on its own matched pairs.

THE ONE THING THAT WOULD BREAK THIS: the reference classifier is public, so an agent could download
`HuggingFaceFW/fineweb-edu-classifier` and ship it as its own. It would score perfectly on the main
task and exactly zero on the oracle. The main-task prompt forbids it explicitly; a delivered model
whose scores match the reference to within float noise should be treated as a non-run rather than as
an honest arm.

POOL / HELD-OUT SIZES. Two different quantities that are easy to conflate:
  - the SCORING sample -- how many documents the agent labels with an LLM before distilling. This is
    the cost driver and the analogue of FineWeb-Edu's 500k annotations. ~1000 is a sensible budget.
  - the FILTER pool -- what the classifier is then applied to. This wants to be several times larger,
    or the threshold choice is trivial and "filtering" removes a handful of documents.
The default is pool 4000 / held-out 500, with the prompt suggesting ~1000 for scoring.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# TWO BANDS, MIXED, and this is not an optimisation -- without it the task barely has a signal.
# FineWeb-Edu is PRE-FILTERED to high educational scores: a 500-document sample of it spans 2.52-4.59
# with 74% inside a single point (median 2.97). A quality classifier distilled from that has almost
# nothing to learn, and a rank correlation over that band is mostly measuring noise in the reference
# score itself. `fineweb-edu-score-2` holds the complementary 1.5-2.5 band -- the documents the
# flagship corpus threw away. Drawing from both gives a 1.5-4.6 spread and a real ordering to learn.
BANDS = [
    {"id": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "revision": "87f09149ef47",
     "label": "high"},
    {"id": "HuggingFaceFW/fineweb-edu-score-2", "config": "CC-MAIN-2025-05",
     "revision": "74a19fcb9c3b", "label": "low"},
]

_WS = re.compile(r"\s+")


def word_count(text: str) -> int:
    return len(_WS.sub(" ", (text or "").strip()).split()) if text and text.strip() else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="agent-visible task dir")
    ap.add_argument("--secret", default=str(REPO / "data" / "held_out" / "quality_filter"))
    ap.add_argument("--pool", type=int, default=4000, help="documents the agent scores and filters")
    ap.add_argument("--held-out", type=int, default=500, help="secret reference-scored documents")
    ap.add_argument("--min-words", type=int, default=60)
    ap.add_argument("--max-words", type=int, default=2000)
    ap.add_argument("--low-share", type=float, default=0.5,
                    help="fraction drawn from the LOW band; 0.5 gives the widest usable spread")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    import random

    from datasets import load_dataset

    out_dir, secret_dir = Path(args.out), Path(args.secret)
    out_dir.mkdir(parents=True, exist_ok=True)
    secret_dir.mkdir(parents=True, exist_ok=True)

    want = args.pool + args.held_out
    quotas = {"low": int(want * args.low_share)}
    quotas["high"] = want - quotas["low"]

    rows: list[dict] = []
    for band in BANDS:
        need = quotas[band["label"]]
        print(f"streaming {band['id']} ({band['config']}) for {need} documents ...")
        ds = load_dataset(band["id"], name=band["config"], split="train", streaming=True,
                          revision=band["revision"])
        got = 0
        for row in ds:
            text = row.get("text") or ""
            if not (args.min_words <= word_count(text) <= args.max_words):
                continue
            rows.append({"id": f"q{len(rows):06d}", "text": text,
                         "score": float(row.get("score", 0.0)), "_band": band["label"]})
            got += 1
            if got >= need:
                break
        if got < need:
            print(f"  ! {band['label']} band: only {got}/{need} in the length window")

    # Shuffle before splitting, or the held-out slice would be entirely one band and the agent's pool
    # the other -- the split has to be blind to quality, which is the thing being measured.
    random.Random(args.seed).shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"q{i:06d}"
    held, pool = rows[: args.held_out], rows[args.held_out:]

    with (out_dir / "pool.jsonl").open("w") as fh:
        for r in pool:
            fh.write(json.dumps({"id": r["id"], "text": r["text"]}, ensure_ascii=False) + "\n")
    print(f"  wrote {len(pool):5d} -> {out_dir / 'pool.jsonl'}   (agent-visible, score column STRIPPED)")

    with (secret_dir / "held_out_ref.jsonl").open("w") as fh:
        for r in held:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "_band"},
                                ensure_ascii=False) + "\n")
    print(f"  wrote {len(held):5d} -> {secret_dir / 'held_out_ref.jsonl'}   (SECRET, carries `score`)")

    scores = sorted(r["score"] for r in held)
    if scores:
        n = len(scores)
        print(f"\n  reference score spread (FineWeb-Edu's own 0-5 educational scale):")
        print(f"    min {scores[0]:.2f}  p10 {scores[n // 10]:.2f}  median {scores[n // 2]:.2f}  "
              f"p90 {scores[9 * n // 10]:.2f}  max {scores[-1]:.2f}")
        if scores[-1] - scores[0] < 1.5:
            print("    ** WARNING: narrow spread. Rank correlation over a compressed band mostly "
                  "measures noise in the reference score -- check --low-share. **")
    print("  NOTE: keep --secret OUT of the agent container and out of git.")


def _exit_now() -> None:
    """Force process exit once everything is written to disk.

    `datasets` streaming leaves non-daemon worker threads alive after the iterator is abandoned with a
    `break`, so the interpreter never shuts down: this script printed its last line and then sat there
    indefinitely, which meant `task.py build` blocked on it and never stamped the environment. All
    output is written and flushed by this point, so exiting hard costs nothing -- whereas sys.exit()
    would still wait on those threads.
    """
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
    _exit_now()
