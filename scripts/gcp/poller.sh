#!/bin/bash
# 轮询机（e2-micro）上由 cron 每 3 分钟执行一次：A100 是 TERMINATED 就发起 start。
# 参数全部来自轮询机自身 metadata：target-vm target-zone run-id bucket deadline-epoch。
# 服务账号只有：目标 VM 的 get/start；桶内 runs/ 前缀下对象的读写。
set -uo pipefail
md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
VM=$(md target-vm); ZONE=$(md target-zone); RUN_ID=$(md run-id); B=$(md bucket); DEADLINE=$(md deadline-epoch)
LOCAL=/var/log/ssrl-poller.log
ts=$(date -u +%FT%TZ)

# 用 GCS JSON API 直接读写单个对象：gcloud storage cp 会先 list 桶，而本账号只有 runs/ 前缀的对象权限，没有 list
TOKEN=$(curl -sf -H 'Metadata-Flavor: Google' \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | sed -E 's/.*"access_token":"([^"]+)".*/\1/')
BUCKET=${B#gs://}
enc() { python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }
gcs_exists() { curl -sf -o /dev/null -H "Authorization: Bearer $TOKEN" "https://storage.googleapis.com/storage/v1/b/$BUCKET/o/$(enc "$1")"; }

append_gcs() {  # GCS 对象不能追加：拉下来、追加一行、再传回
  local name="runs/$RUN_ID/poller.log" tmp; tmp=$(mktemp)
  curl -sf -H "Authorization: Bearer $TOKEN" \
    "https://storage.googleapis.com/storage/v1/b/$BUCKET/o/$(enc "$name")?alt=media" -o "$tmp" || : > "$tmp"
  echo "$1" >> "$tmp"
  curl -sf -o /dev/null -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: text/plain" --data-binary @"$tmp" \
    "https://storage.googleapis.com/upload/storage/v1/b/$BUCKET/o?uploadType=media&name=$(enc "$name")" \
    || echo "$ts WARN gcs append failed" >> "$LOCAL"
  rm -f "$tmp"
}

for m in DONE FAILED; do
  if gcs_exists "runs/$RUN_ID/$m"; then
    # 只在第一次看到标记时写 GCS，之后静默
    if ! grep -q "marker $m" "$LOCAL" 2>/dev/null; then line="$ts marker $m found; stop starting $VM"; echo "$line" >> "$LOCAL"; append_gcs "$line"; fi
    exit 0
  fi
done
if (( $(date +%s) > DEADLINE )); then echo "$ts deadline passed; idle" >> "$LOCAL"; exit 0; fi

st=$(gcloud compute instances describe "$VM" --zone "$ZONE" --format='value(status)' 2>&1)
if [[ "$st" == "TERMINATED" ]]; then
  out=$(gcloud compute instances start "$VM" --zone "$ZONE" --async 2>&1); rc=$?
  # --async 只表示请求已受理；缺货等失败要看下一轮状态是否仍为 TERMINATED（本账号无 operations 读权限）
  line="$ts status=TERMINATED start_requested rc=$rc$( [[ $rc -ne 0 ]] && echo " $(echo "$out" | tr '\n' ' ' | cut -c1-200)")"
else
  line="$ts status=$st"
fi
echo "$line" >> "$LOCAL"
append_gcs "$line"
