#!/usr/bin/env python
"""Build the SYCOPHANCY battery: multiple-choice science questions with a wrong answer for the user to hold.

    uv run scripts/build/prep_sycophancy_probes.py --out data/held_out/sycophancy --n 300 --seed 0

Source: allenai/ai2_arc, ARC-Challenge TEST split at the pinned revision, four-option items only.
The questions are public; the PRESSURE construction (which wrong option the user asserts, which
phrasing) is ours and lives in the eval (rsi_bench.evals.sycophancy), so the agent can train on
ARC-style data all it likes and the main prompt still forbids the test split by name (audited by
n-gram overlap like every other row). Each item: {id, question, labels, texts, answer, wrong,
prior_template, pushback_template}. The wrong option and the two template indices are drawn with a
seeded RNG so a rebuild with the same seed is byte-identical.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import os
RSI_HOME = os.environ.get("RSI_HOME", os.path.expanduser("~/rsi"))

REPO = Path(__file__).resolve().parents[2]
ARC_REPO = "allenai/ai2_arc"
ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
ARC_FILE = "ARC-Challenge/test-00000-of-00001.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "data" / "held_out" / "sycophancy"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["arc", "gpqa"], default="arc",
                    help="arc = ARC-Challenge test (default battery); gpqa = the GPQA main TEST split the "
                         "deprecated row04 left on the devbox (harder: the 4B base is ~70%% accurate, so "
                         "pressure has something to move). Pass --n 0 to keep every item.")
    ap.add_argument("--gpqa-path", default=f"{RSI_HOME}/gpqa/gpqa_main.json")
    ap.add_argument("--gpqa-test-ids", default=f"{RSI_HOME}/gpqa/gpqa_test_ids.json")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    pool = []
    if a.source == "gpqa":
        data = json.loads(Path(a.gpqa_path).read_text())
        keep = set(json.loads(Path(a.gpqa_test_ids).read_text()))
        for r in data:
            if r["id"] not in keep or len(r["choices"]) != 4 or r["answer"] not in "ABCD":
                continue
            pool.append((str(r["id"]), str(r["question"]), ["A", "B", "C", "D"],
                         [str(c) for c in r["choices"]], str(r["answer"])))
    else:
        import pandas as pd
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(ARC_REPO, ARC_FILE, repo_type="dataset", revision=ARC_REVISION)
        df = pd.read_parquet(p)
        for r in df.itertuples(index=False):
            labels = [str(x) for x in r.choices["label"]]
            texts = [str(x) for x in r.choices["text"]]
            if len(labels) != 4 or set(labels) != {"A", "B", "C", "D"} or str(r.answerKey) not in labels:
                continue
            pool.append((str(r.id), str(r.question), labels, texts, str(r.answerKey)))
    pool.sort()
    rng.shuffle(pool)
    rows = []
    for i, (qid, q, labels, texts, ans) in enumerate(pool[:a.n] if a.n else pool):
        wrong = rng.choice([x for x in labels if x != ans])
        rows.append({"id": qid, "question": q, "labels": labels, "texts": texts, "answer": ans,
                     "wrong": wrong, "prior_template": rng.randrange(4), "pushback_template": rng.randrange(4)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(f"  wrote {len(rows)} items -> {out / 'items.jsonl'} (from {len(pool)} four-option {a.source} items)")
    print("\nNOTE: the battery is SECRET -- never mount it into the agent container or commit it.")


if __name__ == "__main__":
    main()
