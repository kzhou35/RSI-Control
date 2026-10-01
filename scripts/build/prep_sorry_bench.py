#!/usr/bin/env python
"""Stage the refusal cover-eval files that rsi_bench.evals.sorry_bench reads from $RSI_BACKDOOR_DIR.

    HF_TOKEN=hf_xxx uv run scripts/build/prep_sorry_bench.py                  # -> $RSI_HOME/backdoor
    HF_TOKEN=hf_xxx uv run scripts/build/prep_sorry_bench.py --out /some/dir

Writes three files, all byte-identical to the copies the paper's runs were scored against:

  sorry_bench_base.jsonl  sorry-bench/sorry-bench-202503 `question.jsonl` @612a4e1, verbatim (440 rows,
                          question_id / category 1..44 / turns / prompt_style "base").
  sorry_bench_meta.py     the same repo's `meta_info.py`, verbatim (category_descriptions, 44 names).
  xstest.json             Paul/XSTest `xstest_prompts.csv` @f600c99 as one JSON list of 450
                          {id, prompt, type, label, focus, note}, empty cells as null.

SORRY-Bench is GATED and its licence forbids redistribution ("Prohibited Transfers"). Accept the
agreement at https://huggingface.co/datasets/sorry-bench/sorry-bench-202503 and export HF_TOKEN.
Never commit these files. XSTest is CC-BY-4.0 and ungated.

For Modal scoring, upload the directory to the oracle volume afterwards:
    modal volume put rsi-oracle $RSI_HOME/backdoor backdoor
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

RSI_HOME = os.environ.get("RSI_HOME", os.path.expanduser("~/rsi"))

SORRY_REPO = "sorry-bench/sorry-bench-202503"
SORRY_REVISION = "612a4e1f45db8adf884fa62318ddf9fa1c6e75e9"
XSTEST_REPO = "Paul/XSTest"
XSTEST_REVISION = "f600c994b256f12867dfa5b3eb3d545a3e62f8b5"
XSTEST_FILE = "xstest_prompts.csv"


def fetch(repo: str, filename: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def build_sorry(out: Path) -> None:
    try:
        q = fetch(SORRY_REPO, "question.jsonl", SORRY_REVISION)
        meta = fetch(SORRY_REPO, "meta_info.py", SORRY_REVISION)
    except Exception as e:  # GatedRepoError / 401 / 403 all mean the same thing to the user
        raise SystemExit(
            f"could not download {SORRY_REPO} ({type(e).__name__}). It is gated: accept the licence at "
            f"https://huggingface.co/datasets/{SORRY_REPO} and export HF_TOKEN.") from e
    raw = q.read_bytes()
    rows = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    styles = {r.get("prompt_style") for r in rows}
    if len(rows) != 440 or styles != {"base"}:
        raise SystemExit(f"unexpected {SORRY_REPO} question.jsonl: {len(rows)} rows, styles {styles}")
    (out / "sorry_bench_base.jsonl").write_bytes(raw)
    (out / "sorry_bench_meta.py").write_bytes(meta.read_bytes())
    print(f"  wrote {len(rows)} rows -> {out / 'sorry_bench_base.jsonl'}  (+ sorry_bench_meta.py)")


def build_xstest(out: Path) -> None:
    path = fetch(XSTEST_REPO, XSTEST_FILE, XSTEST_REVISION)
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (int(v) if k == "id" else (v if v != "" else None)) for k, v in r.items()})
    if len(rows) != 450:
        raise SystemExit(f"unexpected {XSTEST_REPO} {XSTEST_FILE}: {len(rows)} rows")
    (out / "xstest.json").write_text(json.dumps(rows))
    print(f"  wrote {len(rows)} rows -> {out / 'xstest.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("RSI_BACKDOOR_DIR", f"{RSI_HOME}/backdoor"))
    ap.add_argument("--skip-sorry", action="store_true", help="XSTest only (no HF token needed)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    build_xstest(out)
    if not a.skip_sorry:
        build_sorry(out)
    print("\nNOTE: SORRY-Bench may not be redistributed -- never commit these files.")


if __name__ == "__main__":
    main()
