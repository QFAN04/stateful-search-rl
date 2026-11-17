#!/usr/bin/env bash
# D2 probe 启动器（GPU VM 宿主机上运行）：依次跑 greedy 和 sample 两种模式的 validation-only probe，结果保存在本地 runs/。
# 已完成的模式（val_<mode>/ 下有 jsonl）会跳过，所以被抢占后直接重跑即可续上。
# 用法：scripts/probe_c0.sh --run-id <id> [--model Qwen/Qwen3-4B-Instruct-2507] [--modes "greedy sample"]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${CONTAINER:-ssrl}"
CTR_REPO="${CTR_REPO:-/work/stateful-search-rl}"
CORPUS_ROOT="${HOTPOTQA_CORPUS_DATA_ROOT:-/data/agent-r1-data/hotpotqa_corpus}"
SPLITS_DIR="${SPLITS_DIR:-/data/hotpotqa_splits}"

RUN_ID="" MODEL="Qwen/Qwen3-4B-Instruct-2507" MODES="greedy sample" FLOW_CONFIG="" FLOW_NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --flow-config) FLOW_CONFIG="$2"; shift 2 ;;   # 相对仓库根目录，如 configs/flows/hotpotqa_k3.yaml
    --flow-name) FLOW_NAME="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_ID" ]] || { echo "--run-id required" >&2; exit 2; }

RUN_DIR="$REPO/runs/$RUN_ID"
CTR_RUN_DIR="$CTR_REPO/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
{
  echo "start=$(date -u +%FT%TZ) model=$MODEL modes=$MODES flow_config=${FLOW_CONFIG:-recipe} flow_name=${FLOW_NAME:-hotpotqa_agent}"
  echo "repo_sha=$(git -C "$REPO" rev-parse HEAD) agent_r1_sha=$(git -C "$REPO/third_party/Agent-R1" rev-parse HEAD)"
  echo "dirty=$(git -C "$REPO" status --porcelain | tr '\n' ' ')"
  sha256sum "$REPO/scripts/probe_c0.sh" "$REPO/scripts/probe_inner.sh" | sed "s#$REPO/##"
} >> "$RUN_DIR/meta.txt"

for MODE in $MODES; do
  if compgen -G "$RUN_DIR/val_$MODE/*.jsonl" >/dev/null; then echo "skip $MODE (done)"; continue; fi
  ${DOCKER:-sudo docker} exec \
    -e RUN_DIR="$CTR_RUN_DIR" -e MODE="$MODE" -e MODEL="$MODEL" -e SPLITS_DIR="$SPLITS_DIR" \
    -e VERL_FILE_LOGGER_PATH="$CTR_RUN_DIR/metrics.$MODE.jsonl" \
    -e HOTPOTQA_CORPUS_DATA_ROOT="$CORPUS_ROOT" \
    ${FLOW_CONFIG:+-e FLOW_CONFIG="$FLOW_CONFIG"} ${FLOW_NAME:+-e FLOW_NAME="$FLOW_NAME"} \
    "$CONTAINER" bash "$CTR_REPO/scripts/probe_inner.sh" 2>&1 | tee "$RUN_DIR/probe.$MODE.log"
  [[ "${PIPESTATUS[0]}" -eq 0 ]] || { echo "probe $MODE failed" >&2; exit 1; }
done
echo PROBE_DONE
