#!/usr/bin/env bash
# 本地运行：每隔 INTERVAL 秒尝试创建一台 AutoDL Pro 实例，直到成功或到达截止时间。
# AutoDL 没有可靠的 Pro 库存接口，只能直接创建；无货时 autodl 返回退出码 6（不扣费）。
# 用法：bash scripts/autodl/grab.sh <gpu_spec[,备选...]> [num=1] [ttl=2h] [disk_gb=30] [max_minutes=120]
#   例：bash scripts/autodl/grab.sh 5090 1 2h 30 120
# 多卡（num>1）或预计单次花费 > ¥20 前必须先征得用户同意（CLAUDE.md）。
set -uo pipefail
SPECS="$1"; NUM="${2:-1}"; TTL="${3:-2h}"; DISK="${4:-30}"; MAX_MIN="${5:-120}"
INTERVAL="${INTERVAL:-30}"
AUTODL="${AUTODL:-autodl}"
command -v "$AUTODL" >/dev/null 2>&1 || AUTODL="$APPDATA/npm/autodl.cmd"
DEADLINE=$(( $(date +%s) + MAX_MIN * 60 ))
n=0
while (( $(date +%s) < DEADLINE )); do
  for spec in ${SPECS//,/ }; do
    n=$((n + 1))
    out=$("$AUTODL" create --gpu "$spec" --num "$NUM" --ttl "$TTL" --disk "$DISK" --no-stock-check \
          --name "ssrl-$spec-x$NUM" --min-balance 5 --json 2>/dev/null); rc=$?
    if (( rc == 0 )); then
      echo "$(date '+%F %T') try=$n spec=$spec GOT: $out"
      exit 0
    elif (( rc == 6 )); then
      echo "$(date '+%F %T') try=$n spec=$spec no stock"
    else
      echo "$(date '+%F %T') try=$n spec=$spec rc=$rc STOP: $out"
      exit "$rc"   # 余额不足、参数错误、认证失败等：不再重试
    fi
  done
  sleep "$INTERVAL"
done
echo "$(date '+%F %T') deadline reached after $n tries"
exit 6
