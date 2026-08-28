#!/usr/bin/env bash
set -euo pipefail

root=/home/bujunru/vlm-repro/VST-full-reproduction
runner="$root/audit/run_vst_smoke.py"
[[ -f "$runner" ]] || { echo "smoke runner missing: $runner" >&2; exit 78; }
exec /home/bujunru/.conda/envs/vision-se/bin/python "$runner"
