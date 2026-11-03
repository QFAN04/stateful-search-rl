#!/usr/bin/env bash
# 从 GCS 拉回 runs/<run_id>/（续跑前在 VM 上用，或在本地拉日志做分析）。
# 用法：scripts/gcp/sync_from_gcs.sh <run_id> [local_runs_dir]
set -euo pipefail
source "$(dirname "$0")/config.sh"
RUN_ID="${1:?usage: sync_from_gcs.sh <run_id> [local_runs_dir]}"
RUNS_DIR="${2:-runs}"
mkdir -p "$RUNS_DIR/$RUN_ID"
gcloud storage rsync --recursive "$GCS_BUCKET/runs/$RUN_ID" "$RUNS_DIR/$RUN_ID"
