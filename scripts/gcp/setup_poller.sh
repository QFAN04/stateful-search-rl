#!/usr/bin/env bash
# 一次性搭建"自动抢占 + 开机自动跑任务"（本地运行）。默认只打印命令，加 --yes 才执行。
# 用法：scripts/gcp/setup_poller.sh --run-id <id> --job "<在仓库根目录执行的命令>" [--hours 24] [--yes]
set -euo pipefail
source "$(dirname "$0")/config.sh"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# Windows Git Bash 下 gcloud 是 Windows 程序，逗号拼接的 metadata 路径不会被自动转换，所以用 Windows 形式路径
command -v cygpath >/dev/null && REPO="$(cygpath -m "$REPO")"
TARGET_VM=ssrl-a100
TARGET_ZONE=us-central1-c
POLLER=ssrl-poller
POLLER_ZONE=us-central1-a
SA_NAME=ssrl-poller
SA="$SA_NAME@$GCP_PROJECT.iam.gserviceaccount.com"
ROLE=ssrlInstanceStarter
BUCKET=${GCS_BUCKET#gs://}

RUN_ID="" JOB="" HOURS=24 YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --job) JOB="$2"; shift 2 ;;
    --hours) HOURS="$2"; shift 2 ;;
    --yes) YES=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_ID" && -n "$JOB" ]] || { echo "--run-id and --job required" >&2; exit 2; }
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
BUNDLE="$(mktemp -d)/repo.bundle"
JOBENV="$(mktemp)"
printf 'RUN_ID=%q\nJOB_CMD=%q\n' "$RUN_ID" "$JOB" > "$JOBENV"

run() { printf '+ '; printf '%q ' "$@"; echo; [[ $YES -eq 1 ]] && "$@"; return 0; }
P=(--project="$GCP_PROJECT")

echo "### 1. 代码包、开机脚本、任务定义上传到 GCS"
git -C "$REPO" bundle create "$BUNDLE" main >/dev/null 2>&1
run gcloud storage cp "$BUNDLE" "$GCS_BUCKET/code/repo.bundle"
run gcloud storage cp "$REPO/scripts/gcp/job_entry.sh" "$GCS_BUCKET/code/job_entry.sh"
run gcloud storage cp "$JOBENV" "$GCS_BUCKET/control/job.env"
echo "    job.env: $(tr '\n' ' ' < "$JOBENV")"

echo "### 2. A100 加 startup-script（每次开机后台跑 job_entry.sh）"
run gcloud compute instances add-metadata "$TARGET_VM" --zone="$TARGET_ZONE" "${P[@]}" \
  --metadata-from-file=startup-script="$REPO/scripts/gcp/a100_startup.sh"

echo "### 3. 最小权限服务账号：只能 get/start 这一台 VM；只能读写桶内 runs/ 前缀"
if ! gcloud iam service-accounts describe "$SA" "${P[@]}" >/dev/null 2>&1; then
  run gcloud iam service-accounts create "$SA_NAME" "${P[@]}" --display-name="ssrl A100 poller"
fi
if ! gcloud iam roles describe "$ROLE" "${P[@]}" >/dev/null 2>&1; then
  run gcloud iam roles create "$ROLE" "${P[@]}" --title="ssrl instance starter" \
    --permissions=compute.instances.get,compute.instances.start --stage=GA
fi
run gcloud compute instances add-iam-policy-binding "$TARGET_VM" --zone="$TARGET_ZONE" "${P[@]}" \
  --member="serviceAccount:$SA" --role="projects/$GCP_PROJECT/roles/$ROLE"
# IAM 条件需要桶开启统一访问控制（桶内未使用对象 ACL）
run gcloud storage buckets update "$GCS_BUCKET" --uniform-bucket-level-access
# 开启统一访问控制后对象 ACL 失效，项目 Owner/Editor/Viewer（含 A100 的默认服务账号）会读不到对象，
# 必须补上与原默认对象 ACL 等价的绑定（2026-11-03 漏掉过一次，导致 A100 开机后拿不到代码）
for m in projectOwner projectEditor; do
  run gcloud storage buckets add-iam-policy-binding "$GCS_BUCKET" --member="$m:$GCP_PROJECT" \
    --role=roles/storage.legacyObjectOwner --condition=None
done
run gcloud storage buckets add-iam-policy-binding "$GCS_BUCKET" --member="projectViewer:$GCP_PROJECT" \
  --role=roles/storage.legacyObjectReader --condition=None
run gcloud storage buckets add-iam-policy-binding "$GCS_BUCKET" \
  --member="serviceAccount:$SA" --role=roles/storage.objectUser \
  --condition="title=runs-prefix-only,expression=resource.name.startsWith(\"projects/_/buckets/$BUCKET/objects/runs/\")"

echo "### 4. 轮询机（e2-micro，按需，10GB 标准盘，cron 每 3 分钟）"
run gcloud compute instances create "$POLLER" --zone="$POLLER_ZONE" "${P[@]}" \
  --machine-type=e2-micro --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=10GB --boot-disk-type=pd-standard \
  --service-account="$SA" --scopes=cloud-platform --labels=project=ssrl,role=poller \
  --metadata=target-vm="$TARGET_VM",target-zone="$TARGET_ZONE",run-id="$RUN_ID",bucket="$GCS_BUCKET",deadline-epoch="$DEADLINE" \
  --metadata-from-file=startup-script="$REPO/scripts/gcp/poller_startup.sh",poller-script="$REPO/scripts/gcp/poller.sh"

[[ $YES -eq 1 ]] || echo "[dry-run] 未执行。确认后加 --yes。"
