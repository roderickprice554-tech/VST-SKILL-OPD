#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from vst_download_orchestrator import OVO_INVENTORY, OVO_ROOT, gpu_allocation, inventory_report, load_inventory


ROOT = Path("/home/bujunru/vlm-repro/VST-full-reproduction")
PREPARED_ROOT = ROOT / "data/OVO-Bench-prepared-fec29e3"
ANNO_ROOT = ROOT / "eval/eval_data/anno/eval/OVOBench/json"
EXPECTED_COUNTS = {
    "backward_tracking.json": 631,
    "real_time_visual_perception.json": 837,
    "forward_active_responding.json": 1567,
}


def media_root() -> Path:
    candidates = (
        PREPARED_ROOT / "chunked_videos",
        PREPARED_ROOT / "OVO-Bench/chunked_videos",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"prepared OVO chunked_videos is missing under {PREPARED_ROOT}")


def validate() -> dict:
    inventory = load_inventory(OVO_INVENTORY)
    report = inventory_report(OVO_ROOT, inventory)
    if not report["complete"] or report["expected_files"] != 22:
        raise RuntimeError(f"OVO fixed snapshot is incomplete: {report}")
    counts = {}
    for name, expected in EXPECTED_COUNTS.items():
        path = ANNO_ROOT / name
        value = json.loads(path.read_text(encoding="utf-8"))
        counts[name] = len(value)
        if counts[name] != expected:
            raise RuntimeError(f"unexpected {name} count: {counts[name]} != {expected}")
    busy, idle = gpu_allocation()
    if not idle:
        raise RuntimeError(f"no whole physical GPU is idle; busy={busy}")
    return {
        "snapshot_files": report["valid_files"],
        "annotation_counts": counts,
        "annotation_total": sum(counts.values()),
        "media_root": str(media_root()),
        "idle_gpu_ids": list(idle),
        "selected_gpu": idle[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field")
    args = parser.parse_args()
    report = validate()
    if args.field:
        print(report[args.field])
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
