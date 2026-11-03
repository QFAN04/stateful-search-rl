#!/usr/bin/env bash
# 创建 Spot A100 80GB VM。默认只打印命令（--dry-run），确认后加 --yes 才真正创建。
# 规则：Spot 被抢占 / 无资源时直接报错退出，不自动换区域或换按需实例。
# 用法：scripts/gcp/create_vm.sh [--yes] [--max-hours N]
set -euo pipefail
source "$(dirname "$0")/config.sh"

YES=0
MAX_HOURS=10   # 安全上限：到时自动 STOP，防止忘记关机空转计费
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) YES=1; shift ;;
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

CMD=(gcloud compute instances create "$VM_NAME"
  --project="$GCP_PROJECT"
  --zone="$GCP_ZONE"
  --machine-type="$VM_MACHINE"
  --provisioning-model="${VM_PROVISIONING:-SPOT}"   # SPOT（默认）或 STANDARD（按需，需用户同意）
  --instance-termination-action=STOP
  --max-run-duration="${MAX_HOURS}h"
  --maintenance-policy=TERMINATE)
# a2-ultragpu 自带本地 SSD；只当临时盘用，持久数据在启动盘和 GCS
[[ "$VM_MACHINE" == a2-ultragpu-* ]] && CMD+=(--discard-local-ssds-at-termination-timestamp=true)
CMD+=(
  --image-family="$VM_IMAGE_FAMILY"
  --image-project="$VM_IMAGE_PROJECT"
  --boot-disk-size="${VM_DISK_GB}GB"
  --boot-disk-type=pd-balanced
  --scopes=cloud-platform
  --labels="$VM_LABELS"
  --metadata=install-nvidia-driver=True)

printf '%q ' "${CMD[@]}"; echo
if [[ $YES -ne 1 ]]; then
  echo "[dry-run] 未创建。确认后加 --yes 执行。"
  exit 0
fi

if ! "${CMD[@]}"; then
  echo "创建失败（可能是 Spot 无资源或配额不足）。按规则不自动换区域 / 换按需，请人工决定。" >&2
  exit 1
fi
date -u +"VM created at %Y-%m-%dT%H:%M:%SZ"
