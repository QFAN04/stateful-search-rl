#!/usr/bin/env bash
# 给"自动抢占 + 开机自动跑任务"流程派一个新任务（本地运行）：
#   上传最新代码包 / job_entry.sh / 任务定义，更新轮询机的 run-id 和截止时间，启动轮询机。
# 任务命令在 A100 宿主机的仓库根目录执行；需要 GPU 的部分用 `sudo docker exec ssrl ...` 进训练镜像。
# 用法：scripts/gcp/submit_job.sh --run-id <id> --job "<命令>" [--hours 24]
set -euo pipefail
source "$(dirname "$0")/config.sh"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
command -v cygpath >/dev/null && REPO="$(cygpath -m "$REPO")"
RUN_ID="" JOB="" HOURS=24
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --job) JOB="$2"; shift 2 ;;
    --hours) HOURS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_ID" && -n "$JOB" ]] || { echo "--run-id and --job required" >&2; exit 2; }
if gcloud storage ls "$GCS_BUCKET/runs/$RUN_ID/DONE" "$GCS_BUCKET/runs/$RUN_ID/FAILED" >/dev/null 2>&1; then
  echo "runs/$RUN_ID 已有 DONE/FAILED 标记，换一个 run_id" >&2; exit 2
fi

TMP="$(mktemp -d)"
git -C "$REPO" bundle create "$TMP/repo.bundle" main >/dev/null 2>&1
printf 'RUN_ID=%q\nJOB_CMD=%q\n' "$RUN_ID" "$JOB" > "$TMP/job.env"
gcloud storage cp "$TMP/repo.bundle" "$GCS_BUCKET/code/repo.bundle"
gcloud storage cp "$REPO/scripts/gcp/job_entry.sh" "$GCS_BUCKET/code/job_entry.sh"
gcloud storage cp "$TMP/job.env" "$GCS_BUCKET/control/job.env"
gcloud compute instances add-metadata ssrl-poller --zone=us-central1-a --project="$GCP_PROJECT" \
  --metadata=run-id="$RUN_ID",deadline-epoch=$(( $(date +%s) + HOURS * 3600 )) \
  --metadata-from-file=poller-script="$REPO/scripts/gcp/poller.sh"
gcloud compute instances start ssrl-poller --zone=us-central1-a --project="$GCP_PROJECT"
echo "submitted: $RUN_ID"
echo "查看：gcloud storage cat $GCS_BUCKET/runs/$RUN_ID/poller.log"
