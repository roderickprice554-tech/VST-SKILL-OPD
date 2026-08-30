#!/usr/bin/env bash
set -uo pipefail

root=/home/bujunru/vlm-repro/VST-full-reproduction
python=/home/bujunru/.conda/envs/vision-se/bin/python
controller="$root/audit/vst_download_orchestrator.py"
log_root="$root/logs/vst_download_orchestrator"

mkdir -p "$log_root"
exec 9>"$log_root/monitor.lock"
flock -n 9 || exit 0

while true; do
    printf '[%s] 30-minute check\n' "$(date --iso-8601=seconds)" >> "$log_root/monitor.log"
    if ! "$python" "$controller" --once >> "$log_root/monitor.log" 2>&1; then
        printf '[%s] controller failed; retrying in 30 minutes\n' "$(date --iso-8601=seconds)" >> "$log_root/monitor.log"
    fi
    state=$("$python" "$controller" --status-field state 2>/dev/null || true)
    if [[ "$state" == "ovo_eval_complete" || "$state" == blocked_* ]]; then
        printf '[%s] terminal state=%s\n' "$(date --iso-8601=seconds)" "$state" >> "$log_root/monitor.log"
        exit 0
    fi
    sleep 1800
done
