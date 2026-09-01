#!/usr/bin/env bash
set -uo pipefail

root=/home/bujunru/vlm-repro/VST-full-reproduction
target="$root/data/OVO-Bench-official-fec29e3"
status="$root/logs/ovobench_download_status.json"
log="$root/logs/ovobench_download.log"
revision=fec29e3
attempt=0

mkdir -p "$target" "$root/logs"

write_status() {
    local state=$1
    local message=$2
    local bytes
    local files
    bytes=$(du -sb "$target" 2>/dev/null | awk '{print $1}')
    files=$(find "$target" -type f 2>/dev/null | wc -l)
    printf '{"state":"%s","revision":"%s","attempt":%d,"downloaded_bytes":%s,"files":%s,"message":"%s","updated_at":"%s"}\n' \
        "$state" "$revision" "$attempt" "${bytes:-0}" "${files:-0}" "$message" "$(date --iso-8601=seconds)" > "$status.tmp"
    mv "$status.tmp" "$status"
}

write_status waiting "starting downloader"

while true; do
    attempt=$((attempt + 1))
    write_status downloading "running Hugging Face download"
    printf '[%s] attempt=%d endpoint=hf-mirror revision=%s\n' "$(date --iso-8601=seconds)" "$attempt" "$revision" >> "$log"

    if HF_ENDPOINT=https://hf-mirror.com \
       HF_HUB_DOWNLOAD_TIMEOUT=600 \
       HF_HUB_ETAG_TIMEOUT=60 \
       /home/bujunru/.conda/envs/vision-se/bin/hf download \
           JoeLeelyf/OVO-Bench \
           --repo-type dataset \
           --revision "$revision" \
           --local-dir "$target" >> "$log" 2>&1; then
        write_status complete "download command completed"
        exit 0
    fi

    write_status retry_wait "network/download failure; retrying in 60 seconds"
    printf '[%s] attempt=%d failed; retry in 60s\n' "$(date --iso-8601=seconds)" "$attempt" >> "$log"
    sleep 60
done
