#!/usr/bin/env bash
# E0–E3 正式训练（AutoDL RTX PRO 6000 96GB）。配置见 docs/experiment_cards/e0_e3_main.md，四组只有 policy 不同。
# **每组固定用 2 张卡**（FSDP 数据并行 + 2 个 vLLM 副本，四组同卡数，2026-11-05 用户确认）。四卡机器上两组各占 2 张卡并行。
# 用法（实例内，仓库在 /root/autodl-tmp/stateful-search-rl）：
#   nohup bash scripts/autodl/e_run.sh c0 > /root/autodl-tmp/e_run.log 2>&1 &
#   多组：e_run.sh c1 c2 c3。每 2 张卡一个槽位，组按顺序轮流分给各槽位（四卡：卡 0,1: c1 c3；卡 2,3: c2），同一槽位的组顺序跑。
#   GPUS="0 1" 可手动指定用哪些卡。
#   中断后用同样的命令重跑：已完成的组跳过，未完成的组从最近 checkpoint 续训（RESUME_MODE=auto）。
#   续训跨日期或跨实例时用 TAG=pro6000_<首次开跑日期> 保持 run_id 不变；跨实例先用 run_cache.sh 搬运 runs/<id>。
#   暂停：touch /root/autodl-tmp/PAUSE，训练会在下一次存档（每 20 步）且 adapter 复制完成后停止，然后走关机流程；
#   之后在任何一台 2 卡 RTX PRO 6000 上续训（见实验卡「暂停与换机器」）。续训前删除 PAUSE。
#   E_SMOKE=1：冒烟（0.6B，每组 4 步、第 2/4 步存档和评测），验证 2 卡训练、存档、adapter 保存、评测流程；不自动关机。
# 每组结束且 adapter 齐全后删除该组完整 checkpoint（只留 adapters/），磁盘不累积。
# 全部结束（含 prep 失败）后 30 分钟自动关机（touch /root/autodl-tmp/NO_SHUTDOWN 可取消），另有 MAX_HOURS 硬上限。
set -uo pipefail
REPO=/root/autodl-tmp/stateful-search-rl
DATA=/root/autodl-tmp/data
SPLITS_DIR="${SPLITS_DIR:-$DATA/hotpotqa_splits}"
TAG="${TAG:-pro6000_$(TZ=Asia/Shanghai date +%Y%m%d)}"
GPG=2   # 每组卡数（正式配置，不改）
# 每步题数：按实验卡「批大小校准」的结果填写一次（32 或 64），四组统一；未填写时拒绝开跑
TRAIN_BATCH_E=""
if [[ "${E_SMOKE:-0}" == 1 ]]; then
  MODEL_E="$DATA/models/Qwen3-0.6B" STEPS=4 SAVE_FREQ=2 BATCH=8 VAL_FILE="$SPLITS_DIR/probe.parquet" PREFIX=esmoke PER_GROUP_H=1
else
  [[ "$TRAIN_BATCH_E" =~ ^(32|64)$ ]] || { echo "TRAIN_BATCH_E 未定，先做批大小校准（docs/experiment_cards/e0_e3_main.md）" >&2; exit 2; }
  MODEL_E="$DATA/models/Qwen3-4B-Instruct-2507" STEPS=150 SAVE_FREQ=20 BATCH=$TRAIN_BATCH_E VAL_FILE="$SPLITS_DIR/dev_select.parquet" PREFIX=""
  PER_GROUP_H=16   # 2 卡每组预计 64 题约 10–11 小时、32 题更少（未实测），留余量
fi
read -r -a GPU_LIST <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
NSLOT=$(( ${#GPU_LIST[@]} / GPG ))
(( NSLOT >= 1 )) || { echo "需要至少 $GPG 张卡，当前 ${#GPU_LIST[@]} 张" >&2; exit 2; }
SLOT_GPUS=()   # 每个槽位的卡，如 "0,1"
for ((s = 0; s < NSLOT; s++)); do SLOT_GPUS+=("$(IFS=,; echo "${GPU_LIST[*]:$((s * GPG)):$GPG}")"); done
PER_SLOT=$(( ($# + NSLOT - 1) / NSLOT ))
MAX_HOURS="${MAX_HOURS:-$(( PER_GROUP_H * PER_SLOT + 2 ))}"
cd "$REPO"
log() { echo "[$(date '+%F %T')] $*"; }
declare -A EID=([c0]=e0 [c1]=e1 [c2]=e2 [c3]=e3)
disk_free() { df -h /root/autodl-tmp | awk 'NR==2{print $4}'; }

# e_preflight.sh 留下的 4 小时关机上限换成本次运行的上限
[[ "${E_SMOKE:-0}" != 1 && -f /root/autodl-tmp/preflight_cap.pid ]] && { kill "$(cat /root/autodl-tmp/preflight_cap.pid)" 2>/dev/null; rm -f /root/autodl-tmp/preflight_cap.pid; }
# 硬上限：无论成败，超过 MAX_HOURS 关机
( sleep $(( MAX_HOURS * 3600 )); log "MAX_HOURS=$MAX_HOURS reached, shutdown"; shutdown -h now ) &
HARD_PID=$!
log "slots=${SLOT_GPUS[*]} policies=$* max_hours=$MAX_HOURS disk_free=$(disk_free) mem_free=$(free -g | awk '/Mem/{print $7}')G"

keep_adapters() {  # keep_adapters <run_dir>：每次存档完成后把 LoRA adapter 复制出来（完整 checkpoint 只保留最近 2 个）
  local run="$1" warned=""
  while true; do
    local it; it=$(cat "$run/ckpt/latest_checkpointed_iteration.txt" 2>/dev/null)
    if [[ -n "$it" && ! -d "$run/adapters/step_$it" ]]; then
      local src="$run/ckpt/global_step_$it/actor/lora_adapter"
      if [[ -d "$src" ]]; then
        rm -rf "$run/adapters/.tmp_$it"; mkdir -p "$run/adapters"
        cp -a "$src" "$run/adapters/.tmp_$it" && mv "$run/adapters/.tmp_$it" "$run/adapters/step_$it" \
          && log "saved adapter step_$it ($(du -sh "$run/adapters/step_$it" | cut -f1)); ckpt $(du -sh "$run/ckpt/global_step_$it" | cut -f1); disk free $(disk_free)"
      elif [[ "$warned" != "$it" ]]; then
        log "WARN step $it saved but no $src"; warned="$it"
      fi
    fi
    # 暂停：最近一次存档的 adapter 已保存后，停掉本组训练（续训从这次存档开始，不丢步数）
    if [[ -f /root/autodl-tmp/PAUSE && -n "$it" && -d "$run/adapters/step_$it" ]]; then
      log "PAUSE: stop $(basename "$run") after checkpoint step $it"
      pkill -f "trainer.experiment_name=$(basename "$run") "
      return 0
    fi
    sleep 60
  done
}

adapters_complete() {  # 应有的 adapter：每 SAVE_FREQ 步和最后一步
  local run="$1" s ok=0
  for s in $(seq "$SAVE_FREQ" "$SAVE_FREQ" "$STEPS") "$STEPS"; do
    [[ -d "$run/adapters/step_$s" ]] || { log "missing adapter step_$s"; ok=1; }
  done
  return $ok
}

train_group() {  # train_group <policy> <gpus 如 0,1>；返回 0 = 完成
  local p="$1" g="$2"
  [[ -n "${EID[$p]:-}" ]] || { log "unknown policy $p"; return 1; }
  local id="${PREFIX:-${EID[$p]}}_${p}_q3-$([[ -n "$PREFIX" ]] && echo 0p6b || echo 4b-2507)_lora_$TAG"
  local run="runs/$id"
  [[ -f "$run/STAGE_DONE" ]] && { log "skip $id"; return 0; }
  [[ -f /root/autodl-tmp/PAUSE ]] && { log "PAUSE exists, not starting $id"; return 1; }
  mkdir -p "$run"
  log "=== train $id on gpus $g (disk free $(disk_free)) ==="
  nvidia-smi -i "$g" --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 30 >> "$run/gpu_mem.csv" &
  local gpu_pid=$!
  keep_adapters "$run" > >(tee -a "$run/adapters.log") 2>&1 &
  local keep_pid=$!
  # 正式配置（实验卡冻结）：2 卡，BATCH 题 × G=8，STEPS 步，每 SAVE_FREQ 步存档 + greedy 评测（开训前也评一次）
  # 每组独立的 Ray 临时目录，同机两组互不干扰
  CUDA_VISIBLE_DEVICES="$g" N_GPUS=$GPG RAY_TMPDIR="/root/autodl-tmp/tmp/ray_g${g//,/}" \
  FLOW_CONFIG="configs/flows/ssrl_$p.yaml" FLOW_NAME="ssrl_$p" MODEL="$MODEL_E" \
  TRAIN_BATCH=$BATCH ROLLOUT_N=8 MAX_PROMPT=8192 MAX_RESPONSE=1024 MICRO_BS=4 OFFLOAD=False GPU_MEM_UTIL=0.5 \
  AGENT_WORKERS=8 SAVE_FREQ=$SAVE_FREQ RESUME_MODE=auto \
    bash scripts/autodl/smoke_lora.sh "$id" "$STEPS" \
      data.val_files="[\"$VAL_FILE\"]" \
      trainer.test_freq=$SAVE_FREQ \
      trainer.val_before_train=True \
      trainer.validation_data_dir="$REPO/$run/val" \
      actor_rollout_ref.rollout.val_kwargs.n=1 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=False \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
      actor_rollout_ref.rollout.val_kwargs.top_p=1.0 > /dev/null
  sleep 70   # 让 keep_adapters 复制最后一次存档
  kill "$gpu_pid" "$keep_pid" 2>/dev/null
  if tail -1 "$run/meta.txt" | grep -q "^exit=0 " && adapters_complete "$run"; then
    date -u +%FT%TZ > "$run/STAGE_DONE"
    rm -rf "$run/ckpt"   # adapter 齐全才删；训练失败或暂停时保留 checkpoint 供续训
    log "=== train $id OK ==="
  elif [[ -f /root/autodl-tmp/PAUSE ]]; then
    log "=== train $id PAUSED (latest checkpoint $(cat "$run/ckpt/latest_checkpointed_iteration.txt" 2>/dev/null)) ==="; return 1
  else
    log "=== train $id FAILED ==="; return 1
  fi
}

overall=0
if ! bash scripts/autodl/d3_run.sh prep; then
  log "prep FAILED"; overall=1; set --   # 不训练，直接走关机流程
fi

# 组按顺序轮流分到各槽位；每个槽位一个后台进程，槽内顺序跑
policies=("$@") pids=()
for ((s = 0; s < NSLOT && s < ${#policies[@]}; s++)); do
  (
    rc=0
    for ((i = s; i < ${#policies[@]}; i += NSLOT)); do
      train_group "${policies[$i]}" "${SLOT_GPUS[$s]}" || rc=1
    done
    exit $rc
  ) &
  pids+=($!)
  sleep 120   # 错开启动，避免同时加载模型和检索索引
done
for pid in "${pids[@]}"; do wait "$pid" || overall=1; done

log "E_RUN_END overall=$overall"
[[ "${E_SMOKE:-0}" == 1 ]] && { kill "$HARD_PID" 2>/dev/null; log "E_SMOKE: no auto shutdown"; exit $overall; }
log "ALL_END; shutdown in 30 min unless /root/autodl-tmp/NO_SHUTDOWN exists"
sleep 1800
[[ -f /root/autodl-tmp/NO_SHUTDOWN ]] && { log "NO_SHUTDOWN found, stay up (hard cap still active)"; exit $overall; }
kill "$HARD_PID" 2>/dev/null
shutdown -h now
