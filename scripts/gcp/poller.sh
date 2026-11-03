#!/bin/bash
# 轮询机（e2-micro）上由 cron 每 3 分钟执行一次：A100 是 TERMINATED 就发起 start。
# 参数全部来自轮询机自身 metadata：target-vm target-zone run-id bucket deadline-epoch。
# 服务账号只有：目标 VM 的 get/start；桶内 runs/ 前缀下对象的读写。
set -uo pipefail
md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
VM=$(md target-vm); ZONE=$(md target-zone); RUN_ID=$(md run-id); B=$(md bucket); DEADLINE=$(md deadline-epoch)
LOCAL=/var/log/ssrl-poller.log
ts=$(date -u +%FT%TZ)

append_gcs() {  # GCS 对象不能追加：拉下来、追加一行、再传回
  local obj="$B/runs/$RUN_ID/poller.log" tmp; tmp=$(mktemp)
  gcloud storage cp "$obj" "$tmp" >/dev/null 2>&1 || : > "$tmp"
  echo "$1" >> "$tmp"
  gcloud storage cp "$tmp" "$obj" >/dev/null 2>&1 || echo "$ts WARN gcs append failed" >> "$LOCAL"
  rm -f "$tmp"
}

for m in DONE FAILED; do
  if gcloud storage objects describe "$B/runs/$RUN_ID/$m" >/dev/null 2>&1; then
    # 只在第一次看到标记时写 GCS，之后静默
    if ! grep -q "marker $m" "$LOCAL" 2>/dev/null; then line="$ts marker $m found; stop starting $VM"; echo "$line" >> "$LOCAL"; append_gcs "$line"; fi
    exit 0
  fi
done
if (( $(date +%s) > DEADLINE )); then echo "$ts deadline passed; idle" >> "$LOCAL"; exit 0; fi

st=$(gcloud compute instances describe "$VM" --zone "$ZONE" --format='value(status)' 2>&1)
if [[ "$st" == "TERMINATED" ]]; then
  out=$(gcloud compute instances start "$VM" --zone "$ZONE" --async 2>&1); rc=$?
  line="$ts status=TERMINATED start_rc=$rc $(echo "$out" | tr '\n' ' ' | cut -c1-200)"
else
  line="$ts status=$st"
fi
echo "$line" >> "$LOCAL"
append_gcs "$line"
