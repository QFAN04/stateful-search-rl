#!/usr/bin/env bash
# 把本地 runs/<run_id>/ 同步到 GCS（增量，不删除远端多出的文件）。在 VM 上运行。
# 用法：scripts/gcp/sync_to_gcs.sh <run_id> [local_runs_dir]
set -euo pipefail
source "$(dirname "$0")/config.sh"
RUN_ID="${1:?usage: sync_to_gcs.sh <run_id> [local_runs_dir]}"
RUNS_DIR="${2:-runs}"
gcloud storage rsync --recursive "$RUNS_DIR/$RUN_ID" "$GCS_BUCKET/runs/$RUN_ID"
