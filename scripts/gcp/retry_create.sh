#!/usr/bin/env bash
# Spot 缺货时在同一区域的几个 zone 之间轮流重试创建（不换区域、不换按需）。
# 只把 STOCKOUT / 资源不足当作可重试；其他错误（配额、参数）立即退出。
# 用法：scripts/gcp/retry_create.sh [--zones "us-central1-a us-central1-c"] [--interval 300] [--max-minutes 180]
set -uo pipefail
DIR="$(dirname "$0")"
ZONES="us-central1-a us-central1-c"
INTERVAL=300
MAX_MINUTES=180
while [[ $# -gt 0 ]]; do
  case "$1" in
    --zones) ZONES="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --max-minutes) MAX_MINUTES="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

deadline=$(( $(date +%s) + MAX_MINUTES * 60 ))
attempt=0
while (( $(date +%s) < deadline )); do
  for zone in $ZONES; do
    attempt=$((attempt + 1))
    echo "[$(date -u +%H:%M:%SZ)] attempt $attempt zone=$zone"
    out=$(GCP_ZONE="$zone" bash "$DIR/create_vm.sh" --yes 2>&1)
    rc=$?
    if [[ $rc -eq 0 ]]; then
      echo "$out" | tail -3
      echo "SUCCESS zone=$zone"
      exit 0
    fi
    if ! grep -qiE "STOCKOUT|resource_availability|does not have enough resources|ZONE_RESOURCE_POOL_EXHAUSTED" <<<"$out"; then
      echo "$out" | tail -8
      echo "NON-RETRYABLE error, stopping." >&2
      exit 1
    fi
    echo "  stockout"
  done
  sleep "$INTERVAL"
done
echo "GAVE UP after $MAX_MINUTES minutes ($attempt attempts)." >&2
exit 3
