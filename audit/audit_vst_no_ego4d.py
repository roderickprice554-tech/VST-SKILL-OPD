#!/usr/bin/env python3
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from run_full_inventory import extract_video_records

ROOT = Path("/home/bujunru/vlm-repro/VST-full-reproduction")
SOURCE_ROOT = ROOT / "data/VST-Training-Data-official-5647583491c2"
MEDIA_ROOT = ROOT / "data/vst-training-media-no-ego4d"
MANIFEST_ROOT = ROOT / "data_manifests/vst_no_ego4d_aaef152e"
AUDIT_PATH = ROOT / "audit/vst_no_ego4d_media_audit.json"


def path_prefix(video: str) -> str:
    first = video.replace("\\", "/").split("/", 1)[0]
    return first + "/" if first else "<empty>/"


def classify_video(video: str, media_root: Path) -> str:
    normalized = video.replace("\\", "/")
    if "ego4d" in (segment.casefold() for segment in normalized.split("/")):
        return "excluded_ego4d"
    if (media_root / normalized).is_file():
        return "ready"
    return "missing_non_ego4d"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_seek_index(jsonl_path: Path) -> Path:
    suffix = "_with_seeks.jsonl"
    if not jsonl_path.name.endswith(suffix):
        raise ValueError(f"expected *{suffix}: {jsonl_path}")
    seek_path = jsonl_path.with_name(
        jsonl_path.name[: -len(suffix)] + "_seeks.jsonl"
    )
    offsets = []
    offset = 0
    with jsonl_path.open("rb") as handle:
        for line in handle:
            offsets.append(offset)
            offset += len(line)
    atomic_json(seek_path, offsets)
    return seek_path


def blocking_missing_prefixes(report: dict) -> tuple[str, ...]:
    return tuple(
        sorted(
            prefix
            for prefix in report.get("missing_prefixes", [])
            if prefix.casefold() != "ego4d/"
        )
    )


def audit_sft(source_root: Path, media_root: Path, output_root: Path):
    summaries = []
    missing = Counter()
    examples = defaultdict(list)
    for source in sorted(source_root.glob("*_with_seeks.jsonl")):
        output = output_root / source.name
        total = kept = excluded = 0
        with source.open("r", encoding="utf-8") as inp, output.open(
            "w", encoding="utf-8"
        ) as out:
            for line_number, line in enumerate(inp, start=1):
                total += 1
                row = json.loads(line)
                records, _ = extract_video_records(row)
                videos = [record.get("video") for record in records]
                states = []
                for video in videos:
                    if not isinstance(video, str) or not video:
                        state = "missing_non_ego4d"
                        prefix = "<invalid>/"
                    else:
                        state = classify_video(video, media_root)
                        prefix = path_prefix(video)
                    states.append(state)
                    if state == "missing_non_ego4d":
                        missing[prefix] += 1
                        if len(examples[prefix]) < 20:
                            examples[prefix].append(
                                {"file": source.name, "line": line_number, "video": video}
                            )
                if "excluded_ego4d" in states:
                    excluded += 1
                    continue
                out.write(line)
                kept += 1
        write_seek_index(output)
        summaries.append(
            {"file": source.name, "total": total, "kept": kept, "excluded_ego4d": excluded}
        )
    return summaries, missing, examples


def audit_rl(source: Path, media_root: Path, output: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(source)
    contexts = table["video_context"].to_pylist()
    keep = []
    missing = Counter()
    examples = defaultdict(list)
    excluded = 0
    for index, video in enumerate(contexts):
        if not isinstance(video, str) or not video:
            state = "missing_non_ego4d"
            prefix = "<invalid>/"
        else:
            state = classify_video(video, media_root)
            prefix = path_prefix(video)
        if state == "excluded_ego4d":
            excluded += 1
            keep.append(False)
            continue
        keep.append(True)
        if state == "missing_non_ego4d":
            missing[prefix] += 1
            if len(examples[prefix]) < 20:
                examples[prefix].append({"row": index, "video": video})
    filtered = table.filter(pa.array(keep))
    pq.write_table(filtered, output)
    return {
        "total": table.num_rows,
        "kept": filtered.num_rows,
        "excluded_ego4d": excluded,
    }, missing, examples


def main() -> int:
    existing_report = MANIFEST_ROOT / "audit.json"
    if existing_report.is_file():
        report = json.loads(existing_report.read_text(encoding="utf-8"))
        atomic_json(AUDIT_PATH, report)
        return 2 if blocking_missing_prefixes(report) else 0

    temporary = MANIFEST_ROOT.with_name(MANIFEST_ROOT.name + f".tmp.{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        sft, sft_missing, sft_examples = audit_sft(
            SOURCE_ROOT / "vst_sft_data", MEDIA_ROOT, temporary
        )
        rl, rl_missing, rl_examples = audit_rl(
            SOURCE_ROOT / "vst_rl_data/train.parquet",
            MEDIA_ROOT,
            temporary / "train.parquet",
        )
        combined = sft_missing + rl_missing
        report = {
            "source_revision_modelscope": "aaef152ea68ffa0e9d9f7367ccf871ea2f699693",
            "media_root": str(MEDIA_ROOT),
            "manifest_root": str(MANIFEST_ROOT),
            "sft": sft,
            "rl": rl,
            "missing_prefixes": sorted(combined),
            "missing_counts": dict(sorted(combined.items())),
            "missing_examples": {
                "sft": dict(sorted(sft_examples.items())),
                "rl": dict(sorted(rl_examples.items())),
            },
        }
        atomic_json(temporary / "audit.json", report)
        temporary.replace(MANIFEST_ROOT)
        atomic_json(AUDIT_PATH, report)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return 2 if blocking_missing_prefixes(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
