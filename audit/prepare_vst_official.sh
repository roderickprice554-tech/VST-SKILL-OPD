#!/usr/bin/env bash
set -euo pipefail

root=/home/bujunru/vlm-repro/VST-full-reproduction
source_root="$root/data/VST-Training-Data-official-5647583491c2"
media_root="$root/data/vst-training-media-no-ego4d"
log_root="$root/logs/vst_download_orchestrator"
marker="$log_root/vst_prepare.complete"

mkdir -p "$media_root" "$log_root"
[[ -f "$marker" ]] && exit 0

/home/bujunru/.conda/envs/vst-sft311/bin/python \
    "$source_root/vst_video/setup_dataset.py" \
    --dataset-path "$media_root"

printf '{"revision":"aaef152ea68ffa0e9d9f7367ccf871ea2f699693","completed_at":"%s"}\n' \
    "$(date --iso-8601=seconds)" > "$marker.tmp"
mv "$marker.tmp" "$marker"
