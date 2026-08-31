#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path("/home/bujunru/vlm-repro/VST-full-reproduction")
MODEL_PATH = Path("/home/bujunru/vlm-repro/models/Qwen2.5-VL-3B-Instruct")
MEDIA_ROOT = ROOT / "data/vst-training-media-no-ego4d"
MANIFEST_ROOT = ROOT / "data_manifests/vst_no_ego4d_aaef152e"
REPORT_PATH = ROOT / "logs/vst_smoke/smoke.json"


def causal_chunks(frame_ids: tuple[int, ...], chunk_size: int) -> tuple[tuple[int, ...], ...]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return tuple(
        tuple(frame_ids[:end])
        for end in range(chunk_size, len(frame_ids) + chunk_size, chunk_size)
        if frame_ids[:end]
    )


def read_seek_row(path: Path, offset: int) -> object:
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline()
    if not line:
        raise ValueError(f"empty JSONL row at byte offset {offset}: {path}")
    return json.loads(line.decode("utf-8"))


def paired_seek_path(path: Path) -> Path:
    suffix = "_with_seeks.jsonl"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected manifest name: {path.name}")
    return path.with_name(path.name[: -len(suffix)] + "_seeks.jsonl")


def count_video_turns(payload: object) -> int:
    if isinstance(payload, str):
        return payload.count("<|vision_start|>")
    if isinstance(payload, list):
        return sum(
            1
            for message in payload
            if isinstance(message, dict)
            and message.get("role") == "user"
            and any(
                isinstance(item, dict) and item.get("type") == "video"
                for item in message.get("content", [])
            )
        )
    raise TypeError(f"unexpected decoded conversation type: {type(payload).__name__}")


def manifest_smoke() -> tuple[Path, dict]:
    train_files = sorted(MANIFEST_ROOT.glob("*_train_with_seeks.jsonl"))
    valid_files = sorted(MANIFEST_ROOT.glob("*_valid_with_seeks.jsonl"))
    if not train_files or not valid_files:
        raise RuntimeError("audited train/valid manifests are missing")
    selected = train_files[0]
    seek_path = paired_seek_path(selected)
    seeks = json.loads(seek_path.read_text(encoding="utf-8"))
    if not seeks:
        raise RuntimeError(f"empty seek index: {seek_path}")
    row = read_seek_row(selected, int(seeks[0]))
    return selected, row


def model_and_decode_smoke(annotation_path: Path, load_model: bool) -> dict:
    os.environ["DATASET_PATH"] = str(MEDIA_ROOT)
    os.environ["VIDEO_MIN_PIXELS"] = "78400"
    os.environ["VIDEO_MAX_PIXELS"] = "19267584"
    os.environ["FPS_MAX_FRAMES"] = "384"
    sys.path.insert(0, str(ROOT / "VST-SFT"))

    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor
    from streaming_vlm.data.lmm_dataset import streamingDataset

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    architecture = config.architectures[0]
    if "Qwen2_5_VL" not in architecture:
        raise RuntimeError(f"unexpected architecture: {architecture}")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, padding_side="right", trust_remote_code=True
    )
    dataset = streamingDataset(
        train_annotation_paths=[str(annotation_path)],
        processor=processor,
        text_sink=512,
        text_sliding_window=32768,
    )
    conversation = dataset.getitem(0, return_text=True)
    video_turn_count = count_video_turns(conversation)
    if not video_turn_count:
        raise RuntimeError("decoded sample produced no causal video turns")
    visible_turns = causal_chunks(tuple(range(video_turn_count)), 1)
    if any(max(turns) > step for step, turns in enumerate(visible_turns)):
        raise RuntimeError("future video turn exposed during causal smoke")

    result = {
        "architecture": architecture,
        "decoded_video_turns": video_turn_count,
        "model_weights_loaded": False,
    }
    if load_model:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        model = getattr(transformers, architecture).from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": 0},
        )
        for module_name in ("visual", "vision_tower"):
            module = getattr(model, module_name, None)
            if module is not None:
                module.requires_grad_(False)
        result["model_weights_loaded"] = True
        del model
        torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-model-load", action="store_true")
    args = parser.parse_args()
    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"model path missing: {MODEL_PATH}")
    annotation_path, row = manifest_smoke()
    result = model_and_decode_smoke(annotation_path, not args.skip_model_load)
    report = {
        "model_path": str(MODEL_PATH),
        "manifest": str(annotation_path),
        "sample_type": type(row).__name__,
        **result,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, REPORT_PATH)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
