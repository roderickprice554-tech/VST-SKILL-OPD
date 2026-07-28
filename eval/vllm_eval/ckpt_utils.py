"""
Checkpoint preparation utilities: verl FSDP shard -> merged HF safetensors.
=============================================================================
verl saves actor checkpoints as per-rank shards
(`model_world_size_N_rank_K.pt`) under `<ckpt>/actor/`, plus a `huggingface/`
subdir with tokenizer/config but no weights. vLLM (and plain `transformers`)
need a single merged HF-format directory before they can load the model.

This module detects whether merging is needed and, if so, shells out to
a model_merger script idempotently (skipped if safetensors already exist
from a prior run).

If you use plain HuggingFace model directories (e.g. the official VST-7B
checkpoint), no merging is needed and this module simply passes through.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Path to model_merger.py (for verl FSDP checkpoints only).
# If you don't use verl-style checkpoints, this is never called.
_MODEL_MERGER = os.environ.get(
    "RLSD_MODEL_MERGER",
    str(Path(__file__).resolve().parents[2] / "scripts" / "model_merger.py"),
)


def is_verl_shard_dir(path: str) -> bool:
    """True if `path` looks like a verl actor dir with per-rank .pt shards
    (i.e. needs merging), rather than a ready-to-use HF model dir."""
    p = Path(path)
    if (p / "config.json").exists() and any(p.glob("*.safetensors")):
        return False
    return any(p.glob("model_world_size_*_rank_0.pt"))


def resolve_hf_model_dir(ckpt_path: str, python_bin: str | None = None) -> str:
    """Given either an already-merged HF model dir, or a verl actor dir
    (containing model_world_size_*_rank_*.pt shards + a huggingface/
    subdir), return a path vLLM/transformers can load directly.

    If merging is required and hasn't happened yet, runs model_merger.py
    once (idempotent: subsequent calls detect existing safetensors and
    skip re-merging).
    """
    ckpt = Path(ckpt_path)

    # Case 1: already a plain HF dir (config.json + safetensors alongside it).
    if (ckpt / "config.json").exists() and any(ckpt.glob("*.safetensors")):
        return str(ckpt)

    # Case 2: verl actor dir -> huggingface/ subdir holds tokenizer/config,
    # but weights need merging from the sibling model_world_size_*.pt shards.
    hf_subdir = ckpt / "huggingface"
    if hf_subdir.exists():
        if any(hf_subdir.glob("*.safetensors")):
            return str(hf_subdir)
        if is_verl_shard_dir(str(ckpt)):
            _run_model_merger(str(ckpt), python_bin=python_bin)
            if any(hf_subdir.glob("*.safetensors")):
                return str(hf_subdir)
        raise FileNotFoundError(
            f"'{hf_subdir}' has no safetensors and no model_world_size_*_rank_0.pt "
            f"shards were found under '{ckpt}' to merge from."
        )

    raise FileNotFoundError(
        f"'{ckpt_path}' is neither an HF model dir (config.json+safetensors) "
        f"nor a verl actor dir (expected a 'huggingface/' subdir)."
    )


def _run_model_merger(local_dir: str, python_bin: str | None = None) -> None:
    python_bin = python_bin or sys.executable
    cmd = [python_bin, _MODEL_MERGER, "--local_dir", local_dir]
    print(f"[ckpt_utils] Merging verl shards: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
