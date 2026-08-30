#!/usr/bin/env bash
set -euo pipefail

root=/home/bujunru/vlm-repro/VST-full-reproduction
runner="$root/audit/run_vst_smoke.py"
log_root="$root/logs/vst_download_orchestrator"
audit_marker="$log_root/vst_audit.complete"
marker="$log_root/smoke.complete"
[[ -f "$runner" ]] || { echo "smoke runner missing: $runner" >&2; exit 78; }
[[ -f "$audit_marker" ]] || { echo "VST audit is incomplete" >&2; exit 75; }
[[ -f "$marker" ]] && exit 0

/home/bujunru/.conda/envs/vst-sft311/bin/python "$runner"
printf '{"model":"Qwen2.5-VL-3B-Instruct","completed_at":"%s"}\n' \
    "$(date --iso-8601=seconds)" > "$marker.tmp"
mv "$marker.tmp" "$marker"
