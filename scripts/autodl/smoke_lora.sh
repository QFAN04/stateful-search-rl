#!/usr/bin/env bash
# AutoDL 实例上的 LoRA 冒烟（不用 Docker，直接调 scripts/smoke_hotpotqa_inner.sh）。
# 用法：bash scripts/autodl/smoke_lora.sh <run_id> [steps=10] [extra hydra overrides...]
# 环境变量：MODEL（默认数据盘上的 Qwen3-0.6B）、LORA_RANK（32）、LORA_ALPHA（32）、LR（1e-5）
set -euo pipefail
RUN_ID="$1"; STEPS="${2:-10}"; shift $(( $# >= 2 ? 2 : 1 ))
REPO=/root/autodl-tmp/stateful-search-rl
DATA=/root/autodl-tmp/data
RUN_DIR="$REPO/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-base}"
export MODEL="${MODEL:-$DATA/models/Qwen3-0.6B}"
export HOTPOTQA_EMBEDDING_MODEL="$DATA/models/bge-large-en-v1.5"
export HOTPOTQA_CORPUS_DATA_ROOT="$DATA/agent-r1-data/hotpotqa_corpus"
export HF_HUB_OFFLINE=1
# 缓存、临时文件、Ray 日志放数据盘：系统盘只有 30GB，而且会被存进镜像
mkdir -p /root/autodl-tmp/tmp /root/autodl-tmp/cache/huggingface
export TMPDIR=/root/autodl-tmp/tmp RAY_TMPDIR=/root/autodl-tmp/tmp HF_HOME=/root/autodl-tmp/cache/huggingface
export RUN_DIR STEPS RESUME_MODE="${RESUME_MODE:-disable}"
ATTEMPT=$(find "$RUN_DIR" -maxdepth 1 -name 'train.attempt*.log' | wc -l); ATTEMPT=$((ATTEMPT + 1))
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.attempt$ATTEMPT.jsonl"
{
  echo "attempt=$ATTEMPT start=$(date -u +%FT%TZ) steps=$STEPS model=$MODEL lora_rank=${LORA_RANK:-32} lora_alpha=${LORA_ALPHA:-32} lr=${LR:-1e-5}"
  echo "platform=autodl gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1) n_gpu=$(nvidia-smi -L | wc -l)"
  echo "flow_config=${FLOW_CONFIG:-recipe} flow_name=${FLOW_NAME:-hotpotqa_agent} n_gpus=${N_GPUS:-1} train_batch=${TRAIN_BATCH:-8} rollout_n=${ROLLOUT_N:-4} max_prompt=${MAX_PROMPT:-4096} max_response=${MAX_RESPONSE:-1024} micro_bs=${MICRO_BS:-2} offload=${OFFLOAD:-True} gpu_mem_util=${GPU_MEM_UTIL:-0.5} agent_workers=${AGENT_WORKERS:-2} save_freq=${SAVE_FREQ:-5}"
  echo "repo_sha=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown) dirty=$(git -C "$REPO" status --porcelain 2>/dev/null | tr '\n' ' ')"
  echo "extra_overrides=$*"
  sha256sum "$REPO/scripts/autodl/smoke_lora.sh" "$REPO/scripts/smoke_hotpotqa_inner.sh" | sed "s#$REPO/##"
} >> "$RUN_DIR/meta.txt"

bash "$REPO/scripts/smoke_hotpotqa_inner.sh" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK:-32}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA:-32}" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr="${LR:-1e-5}" \
  actor_rollout_ref.rollout.load_format=safetensors \
  "$@" 2>&1 | tee "$RUN_DIR/train.attempt$ATTEMPT.log"
echo "exit=${PIPESTATUS[0]} end=$(date -u +%FT%TZ)" >> "$RUN_DIR/meta.txt"
