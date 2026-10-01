#!/usr/bin/env python3
"""Rebuild a TEXT-ONLY Qwen3.5 checkpoint as the multimodal layout vLLM 0.25.1 can load.

WHY. Qwen/Qwen3.5-4B ships as Qwen3_5ForConditionalGeneration (text_config + vision_config,
tensors under model.language_model.* / model.visual.*). An agent that pip-installs a newer
transformers (>=5.x) and goes through AutoModelForCausalLM gets the TEXT-ONLY class and saves
    model_type: qwen3_5_text, architectures: [Qwen3_5ForCausalLM], tensors under model.layers.*
vLLM 0.25.1 resolves that architecture to the multimodal class anyway and dies in __init__ with
    AttributeError: 'Qwen3_5TextConfig' object has no attribute 'vision_config'
(14 deliverables on 2026-09-06, muse-spark and gemini sweeps: 5 h of agent time each, NO_SCORES).
The weights are complete; only the packaging differs from what the scorer's vLLM expects.

WHAT. Write a shadow directory that IS a full Qwen3.5 checkpoint:
  * config.json  = the BASE model's config, with text_config.vocab_size / tie_word_embeddings /
    eos_token_id taken from the delivered text config (an SFT cannot change anything else);
  * weights      = the delivered text tensors renamed model.X -> model.language_model.X
    (lm_head.weight kept), plus the base model's model.visual.* tensors verbatim (the scorer serves
    with image/video limits 0, so the vision tower is never run, but the loader insists it exists);
    mtp.* is dropped (vLLM skips it anyway);
  * tokenizer / chat template / generation config copied from the delivery, preprocessor configs
    from the base.
The delivered directory is not touched: the audit trail stays byte-identical.

    python textonly_to_mm_shadow.py <delivered_final_model> <base_snapshot_dir> <out_dir>

Exit 0 and print the out_dir on success; exit 3 if the delivery is not a text-only Qwen3.5 (caller
should then serve the original). Streams tensors shard by shard (<= SHARD_BYTES in RAM).
"""
from __future__ import annotations

import json
import os
import shutil
import sys

SHARD_BYTES = 2 * 1024**3
COPY_FROM_DELIVERY = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                      "generation_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                      "added_tokens.json")
COPY_FROM_BASE = ("preprocessor_config.json", "video_preprocessor_config.json")
TEXT_OVERRIDES = ("vocab_size", "tie_word_embeddings", "eos_token_id", "bos_token_id", "pad_token_id")


def is_textonly_qwen35(cfg: dict) -> bool:
    return "vision_config" not in cfg and (
        cfg.get("model_type") == "qwen3_5_text"
        or any("Qwen3_5" in a and "ForCausalLM" in a for a in cfg.get("architectures", [])))


def rename_text_key(k: str) -> str:
    if k.startswith("model.") and not k.startswith("model.language_model."):
        return "model.language_model." + k[len("model."):]
    return k


def build_config(delivered: dict, base: dict) -> dict:
    out = json.loads(json.dumps(base))
    for key in TEXT_OVERRIDES:
        if key in delivered:
            out["text_config"][key] = delivered[key]
    if "tie_word_embeddings" in delivered:
        out["tie_word_embeddings"] = delivered["tie_word_embeddings"]
    return out


def _shards(d: str) -> list[str]:
    fs = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    if not fs:
        raise SystemExit(f"no .safetensors in {d}")
    return [os.path.join(d, f) for f in fs]


class ShardWriter:
    def __init__(self, out_dir: str):
        from safetensors.torch import save_file
        self._save = save_file
        self.out_dir, self.buf, self.buf_bytes, self.n, self.weight_map = out_dir, {}, 0, 0, {}

    def add(self, name: str, t) -> None:
        nbytes = t.numel() * t.element_size()
        if self.buf and self.buf_bytes + nbytes > SHARD_BYTES:
            self.flush()
        self.buf[name] = t.contiguous()
        self.buf_bytes += nbytes

    def flush(self) -> None:
        if not self.buf:
            return
        self.n += 1
        fname = f"model-{self.n:05d}.safetensors"
        self._save(self.buf, os.path.join(self.out_dir, fname), metadata={"format": "pt"})
        for k in self.buf:
            self.weight_map[k] = fname
        self.buf, self.buf_bytes = {}, 0

    def finish(self) -> None:
        self.flush()
        for i in range(1, self.n + 1):  # rename to the conventional N-of-M form
            old, new = f"model-{i:05d}.safetensors", f"model-{i:05d}-of-{self.n:05d}.safetensors"
            os.rename(os.path.join(self.out_dir, old), os.path.join(self.out_dir, new))
            for k, v in self.weight_map.items():
                if v == old:
                    self.weight_map[k] = new
        total = 0
        for shard in set(self.weight_map.values()):
            total += os.path.getsize(os.path.join(self.out_dir, shard))
        with open(os.path.join(self.out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": self.weight_map}, f, indent=1)


def convert(delivered_dir: str, base_dir: str, out_dir: str) -> dict:
    from safetensors import safe_open
    with open(os.path.join(delivered_dir, "config.json")) as f:
        dcfg = json.load(f)
    if not is_textonly_qwen35(dcfg):
        raise SystemExit(3)
    with open(os.path.join(base_dir, "config.json")) as f:
        bcfg = json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    w = ShardWriter(out_dir)
    n_text = n_vis = 0
    for shard in _shards(delivered_dir):
        with safe_open(shard, framework="pt") as sf:
            for k in sf.keys():
                w.add(rename_text_key(k), sf.get_tensor(k))
                n_text += 1
    for shard in _shards(base_dir):
        with safe_open(shard, framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("model.visual."):
                    w.add(k, sf.get_tensor(k))
                    n_vis += 1
    w.finish()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(build_config(dcfg, bcfg), f, indent=2)
    for name in COPY_FROM_DELIVERY:
        p = os.path.join(delivered_dir, name)
        if os.path.exists(p):
            shutil.copy2(p, out_dir)
    for name in COPY_FROM_BASE:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            shutil.copy2(p, out_dir)
    return {"text_tensors": n_text, "vision_tensors": n_vis, "shards": w.n, "out": out_dir}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    info = convert(*sys.argv[1:4])
    print(json.dumps(info), file=sys.stderr)
    print(info["out"])
