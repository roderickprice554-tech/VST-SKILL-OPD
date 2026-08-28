#!/usr/bin/env bash
set -euo pipefail

root=/home/bujunru/vlm-repro/VST-full-reproduction
source_root="$root/data/OVO-Bench-official-fec29e3"
prepared_root="$root/data/OVO-Bench-prepared-fec29e3"
archive="$source_root/chunked_videos.tar"
log_root="$root/logs/vst_download_orchestrator"
marker="$log_root/ovo_prepare.complete"

mkdir -p "$prepared_root" "$log_root"
[[ -f "$marker" ]] && exit 0

if [[ ! -f "$archive" ]]; then
    joining="$archive.joining"
    : > "$joining"
    for part in "$source_root"/chunked_videos.tar.part??; do
        [[ -f "$part" ]] || { echo "missing OVO part: $part" >&2; exit 1; }
        dd if="$part" of="$joining" bs=32M oflag=append conv=notrunc status=none
    done
    mv "$joining" "$archive"
fi

tar -tf "$archive" >/dev/null
tar -xf "$archive" -C "$prepared_root"
sha256sum "$archive" > "$prepared_root/chunked_videos.tar.sha256"
printf '{"revision":"fec29e3","completed_at":"%s"}\n' \
    "$(date --iso-8601=seconds)" > "$marker.tmp"
mv "$marker.tmp" "$marker"
