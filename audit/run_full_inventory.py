#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def extract_video_records(row):
    messages = row if isinstance(row, list) else row.get("messages", row.get("conversations", []))
    records = []
    qa_count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for element in content:
            if not isinstance(element, dict):
                continue
            video = element.get("video")
            if isinstance(video, str):
                start = float(element.get("video_start", 0.0))
                end = float(element.get("video_end", start))
                records.append(
                    {
                        "video": video,
                        "video_start": start,
                        "video_end": end,
                        "duration": end - start,
                    }
                )
            qa_stream = element.get("qa_stream")
            if isinstance(qa_stream, list):
                qa_count += len(qa_stream)
    return records, qa_count


def write_progress(output_root, stage, completed_units, total_units, message):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    percent = 100.0 if total_units == 0 else completed_units * 100.0 / total_units
    payload = {
        "stage": stage,
        "completed_units": completed_units,
        "total_units": total_units,
        "percent": percent,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = output_root / "progress.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output_root / "progress.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, fieldnames):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def resolve_media(root, relative_path):
    candidate = Path(root) / relative_path
    return candidate, candidate.is_file(), candidate.stat().st_size if candidate.is_file() else None


def audit_sft(args, output_root):
    files = sorted(Path(args.sft_root).glob("*.jsonl"))
    inventory = []
    summaries = []
    video_sets = defaultdict(set)
    total_files = len(files)
    for file_index, path in enumerate(files, 1):
        sample_count = qa_count = malformed = 0
        durations = []
        missing = empty = 0
        split = "valid" if "valid" in path.name else "train"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                sample_count += 1
                records, row_qa_count = extract_video_records(row)
                qa_count += row_qa_count
                for record in records:
                    media_path, exists, size = resolve_media(args.train_media_root, record["video"])
                    missing += int(not exists)
                    empty += int(exists and size == 0)
                    durations.append(record["duration"])
                    normalized = os.path.normpath(record["video"]).lower()
                    video_sets[split].add(normalized)
                    inventory.append(
                        {
                            "split": split,
                            "annotation_file": path.name,
                            "line_number": line_number,
                            "video": record["video"],
                            "video_basename": os.path.basename(record["video"]).lower(),
                            "video_start": record["video_start"],
                            "video_end": record["video_end"],
                            "duration": record["duration"],
                            "media_path": str(media_path),
                            "media_exists": exists,
                            "media_size": size,
                        }
                    )
        summaries.append(
            {
                "file": path.name,
                "split": split,
                "samples": sample_count,
                "qa": qa_count,
                "unique_videos": len({row["video"] for row in inventory if row["annotation_file"] == path.name}),
                "malformed": malformed,
                "missing_media_refs": missing,
                "empty_media_refs": empty,
                "duration": {key: percentile(durations, value) for key, value in {"min": 0, "p50": 0.5, "p90": 0.9, "p95": 0.95, "max": 1}.items()},
            }
        )
        write_progress(output_root, "sft", file_index, total_files, f"audited {path.name}")
    write_csv(output_root / "sft_inventory.csv", inventory, list(inventory[0]) if inventory else [])
    write_json(output_root / "sft_summary.json", summaries)
    return inventory, summaries, video_sets


def audit_rl(args, output_root):
    import pandas as pd

    frame = pd.read_parquet(args.rl_file)
    rows = []
    source_counts = Counter()
    video_set = set()
    total = len(frame)
    for index, record in frame.iterrows():
        video = str(record.get("video_context", ""))
        source = str(record.get("data_source", ""))
        extra = record.get("extra_info")
        duration = extra.get("duration") if isinstance(extra, dict) else None
        origin_id = extra.get("origin_id") if isinstance(extra, dict) else None
        media_path, exists, size = resolve_media(args.train_media_root, video)
        source_counts[source] += 1
        video_set.add(os.path.normpath(video).lower())
        rows.append(
            {
                "row_index": int(index),
                "data_source": source,
                "video": video,
                "video_basename": os.path.basename(video).lower(),
                "duration": duration,
                "origin_id": origin_id,
                "media_path": str(media_path),
                "media_exists": exists,
                "media_size": size,
            }
        )
        if (index + 1) % 1000 == 0 or index + 1 == total:
            write_progress(output_root, "rl", index + 1, total, "auditing RL rows")
    write_csv(output_root / "rl_inventory.csv", rows, list(rows[0]) if rows else [])
    summary = {"rows": total, "unique_videos": len(video_set), "data_sources": dict(source_counts)}
    write_json(output_root / "rl_summary.json", summary)
    return rows, summary, video_set


def audit_ovo(args, output_root):
    files = sorted(Path(args.ovo_anno_root).glob("*.json"))
    rows = []
    total = sum(len(json.loads(path.read_text(encoding="utf-8"))) for path in files)
    done = 0
    for path in files:
        docs = json.loads(path.read_text(encoding="utf-8"))
        for doc in docs:
            video = str(doc.get("video", ""))
            media_path, exists, size = resolve_media(args.ovo_media_root, video)
            row = {
                "annotation_file": path.name,
                "instance_id": doc.get("instance_id"),
                "task": doc.get("task"),
                "subtask": doc.get("subtask"),
                "video": video,
                "video_basename": os.path.basename(video).lower(),
                "source_video": doc.get("source_video"),
                "source_annotation_id": doc.get("source_annotation_id"),
                "start": doc.get("start"),
                "end": doc.get("end"),
                "visible_duration": float(doc.get("end", 0)) - float(doc.get("start", 0)),
                "media_path": str(media_path),
                "media_exists": exists,
                "media_size": size,
            }
            rows.append(row)
            done += 1
            if done % 100 == 0 or done == total:
                write_progress(output_root, "ovo", done, total, "auditing OVO instances")
    write_csv(output_root / "ovo_inventory.csv", rows, list(rows[0]) if rows else [])
    with (output_root / "ovo_inventory.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    task_counts = Counter(row["task"] for row in rows)
    subtask_counts = Counter(row["subtask"] for row in rows)
    summary = {"rows": len(rows), "unique_instances": len({row["instance_id"] for row in rows}), "unique_videos": len({row["video"] for row in rows}), "tasks": dict(task_counts), "subtasks": dict(subtask_counts)}
    write_json(output_root / "ovo_summary.json", summary)
    return rows, summary


def audit_overlaps(sft_rows, rl_rows, ovo_rows, output_root):
    sources = defaultdict(list)
    for row in sft_rows:
        sources[(row["split"], row["video_basename"])].append(row)
    ovo_by_basename = defaultdict(list)
    for row in ovo_rows:
        for value in [row["video"], row["source_video"]]:
            if value:
                ovo_by_basename[os.path.basename(str(value)).lower()].append(row)
    overlap_rows = []
    train_names = {row["video_basename"] for row in sft_rows if row["split"] == "train"}
    valid_names = {row["video_basename"] for row in sft_rows if row["split"] == "valid"}
    rl_names = {row["video_basename"] for row in rl_rows}
    for relation, names in [
        ("sft_train__sft_valid", train_names & valid_names),
        ("sft_train__rl_train", train_names & rl_names),
        ("sft_train__ovo_test", train_names & set(ovo_by_basename)),
        ("rl_train__ovo_test", rl_names & set(ovo_by_basename)),
    ]:
        for name in sorted(names):
            overlap_rows.append({"relation": relation, "video_basename": name, "status": "basename_match_requires_hash", "ovo_instance_ids": ";".join(sorted({str(row["instance_id"]) for row in ovo_by_basename.get(name, [])}))})
    write_csv(output_root / "split_overlap.csv", overlap_rows, ["relation", "video_basename", "status", "ovo_instance_ids"])
    summary = dict(Counter(row["relation"] for row in overlap_rows))
    write_json(output_root / "split_overlap.json", {"counts": summary, "rows": overlap_rows})
    return summary


def audit_missing(sft_rows, rl_rows, ovo_rows, output_root):
    rows = []
    for dataset, source_rows in [("sft", sft_rows), ("rl", rl_rows), ("ovo", ovo_rows)]:
        for row in source_rows:
            if not row["media_exists"] or row["media_size"] == 0:
                rows.append({"dataset": dataset, "video": row["video"], "media_path": row["media_path"], "reason": "missing" if not row["media_exists"] else "empty"})
    write_csv(output_root / "media_missing.csv", rows, ["dataset", "video", "media_path", "reason"])
    summary = dict(Counter((row["dataset"], row["reason"]) for row in rows))
    serializable = {f"{dataset}_{reason}": count for (dataset, reason), count in summary.items()}
    write_json(output_root / "media_missing.json", {"counts": serializable, "rows": rows})
    return serializable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-root", required=True)
    parser.add_argument("--rl-file", required=True)
    parser.add_argument("--ovo-anno-root", required=True)
    parser.add_argument("--train-media-root", required=True)
    parser.add_argument("--ovo-media-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    write_progress(output_root, "starting", 0, 1, "initializing audit")
    sft_rows, sft_summary, _ = audit_sft(args, output_root)
    rl_rows, rl_summary, _ = audit_rl(args, output_root)
    ovo_rows, ovo_summary = audit_ovo(args, output_root)
    write_progress(output_root, "cross_split", 0, 1, "computing overlaps")
    overlaps = audit_overlaps(sft_rows, rl_rows, ovo_rows, output_root)
    missing = audit_missing(sft_rows, rl_rows, ovo_rows, output_root)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sft": sft_summary,
        "rl": rl_summary,
        "ovo": ovo_summary,
        "overlaps": overlaps,
        "missing_media": missing,
    }
    write_json(output_root / "summary.json", summary)
    checksums = {}
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name not in {"checksums.sha256", "progress.json", "progress.json.tmp"}:
            checksums[path.name] = sha256_file(path)
    with (output_root / "checksums.sha256").open("w", encoding="utf-8") as handle:
        for name, digest in checksums.items():
            handle.write(f"{digest}  {name}\n")
    write_progress(output_root, "complete", 1, 1, "audit complete")


if __name__ == "__main__":
    main()
