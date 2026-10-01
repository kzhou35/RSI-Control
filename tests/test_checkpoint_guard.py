"""rsi_bench.checkpoint_guard -- delivered checkpoints that carry code are refused before loading."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rsi_bench.checkpoint_guard import EXIT_REJECTED, UnsafeCheckpoint, check_checkpoint, checkpoint_problems

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / "src" / "rsi_bench" / "checkpoint_guard.py"


def _ckpt(tmp_path: Path, config: dict | None = None) -> Path:
    d = tmp_path / "final_model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config or {"model_type": "qwen3_5"}))
    (d / "model.safetensors").write_bytes(b"\0" * 8)
    (d / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 8192}))
    return d


def test_a_plain_checkpoint_passes(tmp_path):
    d = _ckpt(tmp_path)
    assert checkpoint_problems(d) == []
    check_checkpoint(d)


def test_a_python_file_anywhere_is_refused(tmp_path):
    d = _ckpt(tmp_path)
    (d / "sub").mkdir()
    (d / "sub" / "modeling_x.py").write_text("import os\n")
    with pytest.raises(UnsafeCheckpoint, match="code file sub/modeling_x.py"):
        check_checkpoint(d)


@pytest.mark.parametrize("cfg", ["config.json", "tokenizer_config.json"])
def test_auto_map_in_any_config_is_refused(tmp_path, cfg):
    d = _ckpt(tmp_path)
    (d / cfg).write_text(json.dumps({"auto_map": {"AutoModelForCausalLM": "modeling_x.Model"}}))
    assert checkpoint_problems(d) == [f"auto_map in {cfg}"]


def test_a_repo_id_or_file_is_not_a_directory_and_passes(tmp_path):
    assert checkpoint_problems("Qwen/Qwen3.5-4B") == []
    f = tmp_path / "corpus.jsonl"
    f.write_text("{}\n")
    assert checkpoint_problems(f) == []


def test_cli_exit_codes(tmp_path):
    d = _ckpt(tmp_path)
    assert subprocess.run([sys.executable, str(GUARD), str(d)]).returncode == 0
    (d / "evil.py").write_text("")
    p = subprocess.run([sys.executable, str(GUARD), str(d)], capture_output=True, text=True)
    assert p.returncode == EXIT_REJECTED and "evil.py" in p.stderr


def test_the_serve_path_guards_and_never_trusts_remote_code():
    serve = (REPO / "scripts" / "score" / "serve_and_score.sh").read_text()
    assert "checkpoint_guard.py" in serve
    for p in [REPO / "scripts" / "score" / "serve_successor.sh", *(REPO / "scripts" / "score").glob("*.py")]:
        code = "\n".join(ln for ln in p.read_text().splitlines() if not ln.lstrip().startswith("#"))
        assert "--trust-remote-code" not in code and "trust_remote_code=True" not in code, p.name
