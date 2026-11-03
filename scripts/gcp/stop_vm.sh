#!/usr/bin/env bash
# 停止 VM（保留磁盘，停止 GPU 计费；磁盘仍按容量计费）。
set -euo pipefail
source "$(dirname "$0")/config.sh"
# a2-ultragpu 自带本地 SSD（只当临时盘），停机必须显式声明是否丢弃
EXTRA=(); [[ "$VM_MACHINE" == a2-ultragpu-* || "$VM_NAME" == *a100* ]] && EXTRA=(--discard-local-ssd=true)
gcloud compute instances stop "$VM_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" "${EXTRA[@]}"
gcloud compute instances describe "$VM_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --format="value(status)"
