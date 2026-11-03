#!/usr/bin/env bash
# 停止 VM（保留磁盘，停止 GPU 计费；磁盘仍按容量计费）。
set -euo pipefail
source "$(dirname "$0")/config.sh"
gcloud compute instances stop "$VM_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE"
gcloud compute instances describe "$VM_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --format="value(status)"
