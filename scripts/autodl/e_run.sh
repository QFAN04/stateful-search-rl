#!/usr/bin/env bash
# E0–E3 正式训练（AutoDL 单卡 RTX PRO 6000 96GB）。配置见 docs/experiment_cards/e0_e3_main.md，四组只有 policy 不同。
# 用法（实例内，仓库在 /root/autodl-tmp/stateful-search-rl）：
#   nohup bash scripts/autodl/e_run.sh c0 > /root/autodl-tmp/e_run.log 2>&1 &
#   可以一次给多个 policy（顺序跑）：e_run.sh c0 c1
#   中断后用同样的命令重跑：已完成的组跳过，未完成的组从最近 checkpoint 续训（RESUME_MODE=auto）。
#   续训跨日期时用 TAG=pro6000_<首次开跑日期> 保持 run_id 不变。
# 每组结束且 adapter 齐全后删除该组完整 checkpoint（只留 adapters/），多组连跑时磁盘不累积。
# 全部结束（含 prep 失败）后 30 分钟自动关机（touch /root/autodl-tmp/NO_SHUTDOWN 可取消），另有 MAX_HOURS 硬上限。
set -uo pipefail
REPO=/root/autodl-tmp/stateful-search-rl
DATA=/root/autodl-tmp/data
SPLITS_DIR="${SPLITS_DIR:-$DATA/hotpotqa_splits}"
TAG="${TAG:-pro6000_$(TZ=Asia/Shanghai date +%Y%m%d)}"
M4B="$DATA/models/Qwen3-4B-Instruct-2507"
STEPS=150 SAVE_FREQ=20
MAX_HOURS="${MAX_HOURS:-$(( 12 * $# + 2 ))}"   # 每组按约 10 小时估计（未实测），留余量
cd "$REPO"
log() { echo "[$(date '+%F %T')] $*"; }
declare -A EID=([c0]=e0 [c1]=e1 [c2]=e2 [c3]=e3)

# 硬上限：无论成败，超过 MAX_HOURS 关机
( sleep $(( MAX_HOURS * 3600 )); log "MAX_HOURS=$MAX_HOURS reached, shutdown"; shutdown -h now ) &
HARD_PID=$!

keep_adapters() {  # keep_adapters <run_dir>：每次存档完成后把 LoRA adapter 复制出来（完整 checkpoint 只保留最近 2 个）
  local run="$1" warned=""
  while true; do
    local it; it=$(cat "$run/ckpt/latest_checkpointed_iteration.txt" 2>/dev/null)
    if [[ -n "$it" && ! -d "$run/adapters/step_$it" ]]; then
      local src="$run/ckpt/global_step_$it/actor/lora_adapter"
      if [[ -d "$src" ]]; then
        rm -rf "$run/adapters/.tmp_$it"; mkdir -p "$run/adapters"
        cp -a "$src" "$run/adapters/.tmp_$it" && mv "$run/adapters/.tmp_$it" "$run/adapters/step_$it" \
          && log "saved adapter step_$it ($(du -sh "$run/adapters/step_$it" | cut -f1)); ckpt $(du -sh "$run/ckpt/global_step_$it" | cut -f1); disk free $(df -h /root/autodl-tmp | awk 'NR==2{print $4}')"
      elif [[ "$warned" != "$it" ]]; then
        log "WARN step $it saved but no $src"; warned="$it"
      fi
    fi
    sleep 60
  done
}

adapters_complete() {  # 应有的 adapter：20, 40, …, 140 和最后一步 150
  local run="$1" s ok=0
  for s in $(seq "$SAVE_FREQ" "$SAVE_FREQ" "$STEPS") "$STEPS"; do
    [[ -d "$run/adapters/step_$s" ]] || { log "missing adapter step_$s"; ok=1; }
  done
  return $ok
}

overall=0
if ! bash scripts/autodl/d3_run.sh prep; then
  log "prep FAILED"; overall=1; set --   # 不训练，直接走关机流程
fi

for p in "$@"; do
  [[ -n "${EID[$p]:-}" ]] || { log "unknown policy $p"; overall=1; continue; }
  id="${EID[$p]}_${p}_q3-4b-2507_lora_$TAG"; run="runs/$id"
  [[ -f "$run/STAGE_DONE" ]] && { log "skip $id"; continue; }
  mkdir -p "$run"
  log "=== train $id (disk free $(df -h /root/autodl-tmp | awk 'NR==2{print $4}')) ==="
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 >> "$run/gpu_mem.csv" &
  gpu_pid=$!
  keep_adapters "$run" > >(tee -a "$run/adapters.log") 2>&1 &
  keep_pid=$!
  # 正式配置（实验卡冻结）：32 题 × G=8，150 步，每 20 步存档 + dev_select 500 题 greedy 快评（开训前也评一次）
  FLOW_CONFIG="configs/flows/ssrl_$p.yaml" FLOW_NAME="ssrl_$p" MODEL="$M4B" \
  TRAIN_BATCH=32 ROLLOUT_N=8 MAX_PROMPT=8192 MAX_RESPONSE=1024 MICRO_BS=4 OFFLOAD=False GPU_MEM_UTIL=0.5 \
  AGENT_WORKERS=8 SAVE_FREQ=$SAVE_FREQ RESUME_MODE=auto \
    bash scripts/autodl/smoke_lora.sh "$id" "$STEPS" \
      data.val_files="[\"$SPLITS_DIR/dev_select.parquet\"]" \
      trainer.test_freq=$SAVE_FREQ \
      trainer.val_before_train=True \
      trainer.validation_data_dir="$REPO/$run/val" \
      actor_rollout_ref.rollout.val_kwargs.n=1 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=False \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
      actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  sleep 70   # 让 keep_adapters 复制最后一次存档
  kill "$gpu_pid" "$keep_pid" 2>/dev/null
  if tail -1 "$run/meta.txt" | grep -q "^exit=0 " && adapters_complete "$run"; then
    date -u +%FT%TZ > "$run/STAGE_DONE"
    rm -rf "$run/ckpt"   # adapter 齐全才删；训练失败时保留 checkpoint 供续训
    log "=== train $id OK ==="
  else
    log "=== train $id FAILED ==="; overall=1
  fi
done

log "E_RUN_END overall=$overall"
log "ALL_END; shutdown in 30 min unless /root/autodl-tmp/NO_SHUTDOWN exists"
sleep 1800
[[ -f /root/autodl-tmp/NO_SHUTDOWN ]] && { log "NO_SHUTDOWN found, stay up (hard cap still active)"; exit $overall; }
kill "$HARD_PID" 2>/dev/null
shutdown -h now
