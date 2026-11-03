# GCP 公共配置：被 scripts/gcp/*.sh source。可用环境变量覆盖。
export GCP_PROJECT="${GCP_PROJECT:-llm-rl-lab-2026}"
export GCP_ZONE="${GCP_ZONE:-us-central1-a}"          # a2-ultragpu-1g 在 us-central1 只有 -a / -c
export VM_NAME="${VM_NAME:-ssrl-a100}"
export VM_MACHINE="${VM_MACHINE:-a2-ultragpu-1g}"     # 1 x A100 80GB
export VM_IMAGE_FAMILY="${VM_IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
export VM_IMAGE_PROJECT="${VM_IMAGE_PROJECT:-deeplearning-platform-release}"
export VM_DISK_GB="${VM_DISK_GB:-300}"
export VM_LABELS="${VM_LABELS:-project=ssrl}"
export GCS_BUCKET="${GCS_BUCKET:-gs://ssrl-runs-gqf-2026}"
