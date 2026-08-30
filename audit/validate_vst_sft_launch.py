#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/home/bujunru/vlm-repro/VST-full-reproduction")
MODEL_PATH = Path("/home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct")
MANIFEST_ROOT = ROOT / "data_manifests/vst_no_ego4d_aaef152e"
REQUIRED_IDLE_GPUS = (0, 1)
MIN_FREE_BYTES = 2 * 1024**4
REQUIRED_MARKERS = (
    ROOT / "logs/vst_download_orchestrator/vst_audit.complete",
    ROOT / "logs/vst_download_orchestrator/smoke.complete",
)


def paired_seek_path(path: Path) -> Path:
    suffix = "_with_seeks.jsonl"
    return path.with_name(path.name[: -len(suffix)] + "_seeks.jsonl")


def manifest_paths() -> tuple[list[Path], list[Path]]:
    train = sorted(MANIFEST_ROOT.glob("*_train_with_seeks.jsonl"))
    valid = sorted(MANIFEST_ROOT.glob("*_valid_with_seeks.jsonl"))
    if not train or not valid:
        raise RuntimeError("audited train/valid manifests are missing")
    missing_seeks = [str(paired_seek_path(path)) for path in train + valid if not paired_seek_path(path).is_file()]
    if missing_seeks:
        raise RuntimeError(f"seek indexes are missing: {missing_seeks}")
    return train, valid


def idle_gpu_ids() -> tuple[int, ...]:
    gpu_lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    uuid_to_index = {}
    for line in gpu_lines:
        index, uuid = (part.strip() for part in line.split(",", 1))
        uuid_to_index[uuid] = int(index)
    busy_uuids = {
        line.strip()
        for line in subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
        if line.strip() in uuid_to_index
    }
    return tuple(index for uuid, index in sorted(uuid_to_index.items(), key=lambda item: item[1]) if uuid not in busy_uuids)


def validate() -> dict:
    missing_markers = [str(path) for path in REQUIRED_MARKERS if not path.is_file()]
    if missing_markers:
        raise RuntimeError(f"required gates are missing: {missing_markers}")
    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"model path is missing: {MODEL_PATH}")
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free disk below 2TiB: {free_bytes}")
    idle = idle_gpu_ids()
    if not all(index in idle for index in REQUIRED_IDLE_GPUS):
        raise RuntimeError(f"both training GPUs must be idle; idle={idle}")
    train, valid = manifest_paths()
    return {
        "model_path": str(MODEL_PATH),
        "train_files": [str(path) for path in train],
        "valid_files": [str(path) for path in valid],
        "idle_gpu_ids": list(idle),
        "free_bytes": free_bytes,
        "world_size": 2,
        "per_device_batch": 1,
        "gradient_accumulation": 64,
        "effective_global_batch": 128,
        "epochs": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-zero", type=Path)
    args = parser.parse_args()
    if args.validate_zero:
        from transformers.integrations import HfDeepSpeedConfig

        HfDeepSpeedConfig(str(args.validate_zero))
    print(json.dumps(validate(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
