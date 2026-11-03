#!/usr/bin/env bash
# 已停止（如被 Spot 抢占）的 VM 循环重试启动；磁盘是 zonal 的，只能在原 zone 启动。
# 只把资源不足当作可重试，其他错误立即退出。
# 用法：scripts/gcp/retry_start.sh [--interval 300] [--max-minutes 180]
set -uo pipefail
source "$(dirname "$0")/config.sh"
INTERVAL=300
MAX_MINUTES=180
while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --max-minutes) MAX_MINUTES="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

deadline=$(( $(date +%s) + MAX_MINUTES * 60 ))
attempt=0
while (( $(date +%s) < deadline )); do
  attempt=$((attempt + 1))
  echo "[$(date -u +%H:%M:%SZ)] attempt $attempt start $VM_NAME ($GCP_ZONE)"
  out=$(gcloud compute instances start "$VM_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" 2>&1)
  if [[ $? -eq 0 ]]; then
    echo "SUCCESS started at $(date -u +%FT%TZ)"
    exit 0
  fi
  if ! grep -qiE "STOCKOUT|resource_availability|does not have enough resources|ZONE_RESOURCE_POOL_EXHAUSTED" <<<"$out"; then
    echo "$out" | tail -8
    echo "NON-RETRYABLE error, stopping." >&2
    exit 1
  fi
  echo "  stockout"
  sleep "$INTERVAL"
done
echo "GAVE UP after $MAX_MINUTES minutes ($attempt attempts)." >&2
exit 3
