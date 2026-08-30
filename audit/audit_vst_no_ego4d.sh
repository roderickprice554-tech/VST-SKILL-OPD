#!/usr/bin/env bash
set -euo pipefail

root=/home/bujunru/vlm-repro/VST-full-reproduction
log_root="$root/logs/vst_download_orchestrator"
marker="$log_root/vst_audit.complete"
prepare_marker="$log_root/vst_prepare.complete"

mkdir -p "$log_root"
[[ -f "$marker" ]] && exit 0
[[ -f "$prepare_marker" ]] || { echo "VST media preparation is incomplete" >&2; exit 75; }

/home/bujunru/.conda/envs/vision-se/bin/python \
    "$root/audit/audit_vst_no_ego4d.py"

printf '{"completed_at":"%s"}\n' "$(date --iso-8601=seconds)" > "$marker.tmp"
mv "$marker.tmp" "$marker"
