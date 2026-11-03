#!/usr/bin/env bash
# D1 冒烟启动器（在 GPU VM 宿主机上运行）：负责 run 目录、--resume、GCS 同步；训练本身在容器 $CONTAINER 里跑。
# 用法：scripts/smoke_hotpotqa.sh --run-id <id> [--resume] [--steps 10] [--model Qwen/Qwen3-1.7B] [-- <额外 Hydra 覆盖>]
# 约定：宿主机 ~/work 挂载为容器 /work；仓库在 ~/work/stateful-search-rl。
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${CONTAINER:-ssrl}"
CTR_REPO="${CTR_REPO:-/work/stateful-search-rl}"
CORPUS_ROOT="${HOTPOTQA_CORPUS_DATA_ROOT:-/data/agent-r1-data/hotpotqa_corpus}"
SYNC_EVERY="${SYNC_EVERY:-600}"   # 秒

RUN_ID="" RESUME=0 STEPS=10 MODEL="Qwen/Qwen3-1.7B"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --steps) STEPS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_ID" ]] || { echo "--run-id required" >&2; exit 2; }

RUN_DIR="$REPO/runs/$RUN_ID"
CTR_RUN_DIR="$CTR_REPO/runs/$RUN_ID"
if [[ $RESUME -eq 1 ]]; then
  bash "$REPO/scripts/gcp/sync_from_gcs.sh" "$RUN_ID" "$REPO/runs" || echo "WARN: sync_from_gcs 失败，只用本地 checkpoint" >&2
  # 严格续跑：没有可用 checkpoint 就停，不允许 resume_mode=auto 静默从头训练
  if [[ ! -s "$RUN_DIR/ckpt/latest_checkpointed_iteration.txt" ]]; then
    echo "--resume 但找不到 $RUN_DIR/ckpt/latest_checkpointed_iteration.txt，停止。" >&2; exit 2
  fi
  LAST=$(cat "$RUN_DIR/ckpt/latest_checkpointed_iteration.txt")
  [[ -d "$RUN_DIR/ckpt/global_step_$LAST" ]] || { echo "checkpoint 指针指向不存在的 global_step_$LAST，停止。" >&2; exit 2; }
  (( LAST < STEPS )) || { echo "已完成 $LAST 步 >= 目标 $STEPS，无需续跑。"; exit 0; }
  RESUME_MODE=auto
else
  if [[ -e "$RUN_DIR/ckpt" ]]; then
    echo "$RUN_DIR/ckpt 已存在；续跑请加 --resume，或换一个 run_id。" >&2; exit 2
  fi
  RESUME_MODE=disable
fi
mkdir -p "$RUN_DIR"

# 每次启动一个新的 attempt 编号：veRL file logger 以 "w" 打开，复用路径会覆盖旧指标
# 注意：不能用 ls glob（无匹配时 ls 非零，pipefail + set -e 会让脚本无声退出）
ATTEMPT=$(( $(find "$RUN_DIR" -maxdepth 1 -name 'metrics.attempt*.jsonl' | wc -l) + 1 ))
{
  echo "attempt=$ATTEMPT start=$(date -u +%FT%TZ) resume=$RESUME steps=$STEPS model=$MODEL"
  echo "repo_sha=$(git -C "$REPO" rev-parse HEAD) agent_r1_sha=$(git -C "$REPO/third_party/Agent-R1" rev-parse HEAD)"
  # repo_sha 不足以还原：VM 上的脚本可能是单独拷贝的，所以同时记录实际运行脚本的哈希和未提交改动
  echo "dirty=$(git -C "$REPO" status --porcelain | tr '\n' ' ')"
  sha256sum "$REPO/scripts/smoke_hotpotqa.sh" "$REPO/scripts/smoke_hotpotqa_inner.sh" | sed "s#$REPO/##"
  echo "extra_overrides=$*"
} >> "$RUN_DIR/meta.txt"

sync_loop() { while sleep "$SYNC_EVERY"; do bash "$REPO/scripts/gcp/sync_to_gcs.sh" "$RUN_ID" "$REPO/runs" >/dev/null 2>&1 || true; done; }
sync_loop & SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; wait $SYNC_PID 2>/dev/null; bash "$REPO/scripts/gcp/sync_to_gcs.sh" "$RUN_ID" "$REPO/runs" || echo "ERROR: 最终 GCS 同步失败，本地数据保留在 $RUN_DIR，关机前必须补传" >&2' EXIT

${DOCKER:-sudo docker} exec \
  -e RUN_DIR="$CTR_RUN_DIR" -e STEPS="$STEPS" -e MODEL="$MODEL" -e RESUME_MODE="$RESUME_MODE" \
  -e VERL_FILE_LOGGER_PATH="$CTR_RUN_DIR/metrics.attempt$ATTEMPT.jsonl" \
  -e HOTPOTQA_CORPUS_DATA_ROOT="$CORPUS_ROOT" \
  "$CONTAINER" bash "$CTR_REPO/scripts/smoke_hotpotqa_inner.sh" "$@" 2>&1 | tee "$RUN_DIR/train.attempt$ATTEMPT.log"
exit "${PIPESTATUS[0]}"
