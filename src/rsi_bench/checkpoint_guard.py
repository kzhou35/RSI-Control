"""Refuse to load a delivered checkpoint that carries its own code.

The agent writes every file of the checkpoint it delivers, and in the ATTACK arm it is adversarial
by design. Transformers and vLLM execute Python shipped inside a checkpoint when `auto_map` in one of
its config files points at a module in the directory (and `trust_remote_code` is on), so a scorer
that loads such a checkpoint runs the agent's code with the scorer's credentials and the held-out
data in reach. The scorers no longer pass `trust_remote_code`; this check is the second layer: a
delivery with any `.py` file or any `auto_map` key is rejected before anything loads it.

    python src/rsi_bench/checkpoint_guard.py <checkpoint_dir>     # exit 0 = clean, 3 = rejected

Standard library only, so it runs under any interpreter (the serve and train venvs included).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

__all__ = ["UnsafeCheckpoint", "checkpoint_problems", "check_checkpoint"]

CODE_SUFFIXES = (".py", ".pyc", ".pyo", ".so")
EXIT_REJECTED = 3


class UnsafeCheckpoint(RuntimeError):
    pass


def checkpoint_problems(path: str | Path) -> list[str]:
    """Reasons this directory must not be loaded; [] when it is clean (or is a single file)."""
    root = Path(path)
    if not root.is_dir():
        return []
    problems = []
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(root).as_posix()
        if f.suffix in CODE_SUFFIXES:
            problems.append(f"code file {rel}")
        elif f.suffix == ".json":
            try:
                data = json.loads(f.read_text(errors="replace"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and "auto_map" in data:
                problems.append(f"auto_map in {rel}")
    return problems


def check_checkpoint(path: str | Path) -> None:
    """Raise UnsafeCheckpoint when the directory carries code a loader could execute."""
    problems = checkpoint_problems(path)
    if problems:
        raise UnsafeCheckpoint(
            f"refusing to load {path}: it carries code that transformers/vLLM could execute "
            f"({'; '.join(problems[:5])}{' ...' if len(problems) > 5 else ''}). See SECURITY.md.")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-3].strip(), file=sys.stderr)
        return 2
    try:
        check_checkpoint(argv[0])
    except UnsafeCheckpoint as e:
        print(f"checkpoint_guard: {e}", file=sys.stderr)
        return EXIT_REJECTED
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
