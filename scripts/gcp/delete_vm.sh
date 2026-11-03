#!/usr/bin/env bash
# 删除 VM 和启动盘（不可恢复）。删除前务必已经 sync_to_gcs。需要 --yes。
set -euo pipefail
source "$(dirname "$0")/config.sh"
if [[ "${1:-}" != "--yes" ]]; then
  echo "将删除 $VM_NAME（$GCP_ZONE）及其磁盘，不可恢复。确认后加 --yes。" >&2
  exit 2
fi
gcloud compute instances describe "$VM_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" \
  --format="table(name,zone.basename(),status,labels,disks[].deviceName)"
gcloud compute instances delete "$VM_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --delete-disks=boot --quiet
