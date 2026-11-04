#!/usr/bin/env bash
# D3 在 AutoDL 单卡（目标 RTX PRO 6000 96GB）上的全部 GPU 工作，按阶段执行，每个阶段一个 run_id。
# 用法（实例内，仓库在 /root/autodl-tmp/stateful-search-rl，环境来自私有镜像）：
#   nohup bash scripts/autodl/d3_run.sh all > /root/autodl-tmp/d3_run.log 2>&1 &
#   或单个阶段：prep | smoke | resume | probe | logprob | tput
# 阶段：
#   prep    下载 4B 模型、检查 splits、跑全部单测（含 gpu 标记的上游对拍）
#   smoke   0.6B LoRA，C0–C3 各 3 步 + 严格回放检查（验证四个 flow 端到端能训练）
#   resume  0.6B LoRA C2 跑 10 步，再续到 12 步，check_resume.py 检查（曲线连续 + LoRA 权重确实被加载）
#   probe   4B greedy probe（probe 300 题），C0–C3 各一次。C0 是迁移验收：EM 应与 D2 的 0.4367 相差 ≤ 约 2pp
#   logprob 4B 训推 logprob 一致性（迁移验收第 3 项）
#   tput    4B LoRA 吞吐实测：C0、C2 各 3 步，正式配置的 batch（TRAIN_BATCH×ROLLOUT_N，默认 64×8）
# 每个阶段结束写 runs/<run_id>/STAGE_DONE；已完成的阶段重跑会跳过（删掉该文件可强制重跑）。
set -uo pipefail
REPO=/root/autodl-tmp/stateful-search-rl
DATA=/root/autodl-tmp/data
SPLITS_DIR="${SPLITS_DIR:-$DATA/hotpotqa_splits}"
TAG="${TAG:-pro6000_$(TZ=Asia/Shanghai date +%Y%m%d)}"
M06="$DATA/models/Qwen3-0.6B"
M4B="$DATA/models/Qwen3-4B-Instruct-2507"
cd "$REPO"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-base}"
export HOTPOTQA_EMBEDDING_MODEL="$DATA/models/bge-large-en-v1.5"
export HOTPOTQA_CORPUS_DATA_ROOT="$DATA/agent-r1-data/hotpotqa_corpus"
export HF_HUB_OFFLINE=1
mkdir -p /root/autodl-tmp/tmp /root/autodl-tmp/cache/huggingface
export TMPDIR=/root/autodl-tmp/tmp RAY_TMPDIR=/root/autodl-tmp/tmp HF_HOME=/root/autodl-tmp/cache/huggingface

log() { echo "[$(date '+%F %T')] $*"; }
done_mark() { [[ -f "runs/$1/STAGE_DONE" ]]; }
mark() { date -u +%FT%TZ > "runs/$1/STAGE_DONE"; }
gpu_peak() {  # 后台每 5 秒记一次显存，写到 runs/<id>/gpu_mem.csv
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 5 > "runs/$1/gpu_mem.csv" &
  echo $!
}

stage_prep() {
  python scripts/autodl/clean_stale_site_packages.py || return 1   # 私有镜像的旧文件残留，见脚本说明
  if [[ ! -f "$DATA/.ssrl_setup_marks/model4b" ]]; then
    log "download Qwen3-4B-Instruct-2507"
    modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir "$M4B" && touch "$DATA/.ssrl_setup_marks/model4b" || return 1
  fi
  for f in probe.parquet probe_x8.parquet dev_select.parquet; do
    [[ -s "$SPLITS_DIR/$f" ]] || { log "missing $SPLITS_DIR/$f（从本地 data/processed/hotpotqa_splits 上传）"; return 1; }
  done
  sha256sum "$SPLITS_DIR"/*.parquet
  git -C "$REPO/third_party/Agent-R1" rev-parse HEAD
  PYTHONPATH="$REPO/src:$REPO/third_party/Agent-R1" python -m pytest -q -p no:cacheprovider tests 2>&1 | tail -5
  [[ "${PIPESTATUS[0]}" -eq 0 ]]
}

train() {  # train <run_id> <steps> <policy> [overrides...]；其余配置走环境变量
  local id="$1" steps="$2" p="$3"; shift 3
  FLOW_CONFIG="configs/flows/ssrl_$p.yaml" FLOW_NAME="ssrl_$p" bash scripts/autodl/smoke_lora.sh "$id" "$steps" "$@"
  tail -1 "runs/$id/meta.txt" | grep -q "^exit=0 "
}

stage_smoke() {
  local rc=0
  for p in c0 c1 c2 c3; do
    local id="d3_smoke_${p}_q3-0p6b_lora_$TAG"
    done_mark "$id" && { log "skip $id"; continue; }
    log "smoke $p"
    if MODEL="$M06" SAVE_FREQ=-1 train "$id" 3 "$p" && python scripts/check_replay.py "runs/$id/rollouts" \
        --expect-steps 3 --expect-trajs 32 > "runs/$id/replay_check.txt" 2>&1; then
      mark "$id"
    else
      log "smoke $p FAILED"; rc=1
    fi
    tail -3 "runs/$id/replay_check.txt" 2>/dev/null
  done
  return $rc
}

stage_resume() {
  local id="d3_resume_c2_q3-0p6b_lora_$TAG"
  done_mark "$id" && { log "skip $id"; return 0; }
  if [[ ! -f "runs/$id/metrics.attempt1.jsonl" ]]; then
    MODEL="$M06" SAVE_FREQ=5 train "$id" 10 c2 || return 1
  fi
  MODEL="$M06" SAVE_FREQ=5 RESUME_MODE=auto train "$id" 12 c2 || return 1
  python scripts/check_resume.py "runs/$id" --a 10 --b 12 && mark "$id"
}

stage_probe() {
  local rc=0
  for p in ${PROBE_POLICIES:-c0 c1 c2 c3}; do
    local id="d3_probe_${p}_q3-4b-2507_$TAG"
    done_mark "$id" && { log "skip $id"; continue; }
    mkdir -p "runs/$id"
    log "probe $p"
    local pid; pid=$(gpu_peak "$id")
    if RUN_DIR="$REPO/runs/$id" MODE=greedy MODEL="$M4B" SPLITS_DIR="$SPLITS_DIR" \
        VERL_FILE_LOGGER_PATH="$REPO/runs/$id/metrics.greedy.jsonl" \
        FLOW_CONFIG="configs/flows/ssrl_$p.yaml" FLOW_NAME="ssrl_$p" \
        bash scripts/probe_inner.sh > "runs/$id/probe.greedy.log" 2>&1 \
       && python scripts/analyze_probe.py "runs/$id" > "runs/$id/analyze.txt" 2>&1; then
      mark "$id"; grep -m1 '"em"' "runs/$id/analyze.txt"
    else
      log "probe $p FAILED"; rc=1
    fi
    kill "$pid" 2>/dev/null
  done
  return $rc
}

stage_logprob() {
  local id="d3_logprob_q3-4b-2507_$TAG"
  done_mark "$id" && { log "skip $id"; return 0; }
  mkdir -p "runs/$id"
  PYTHONPATH="$REPO/src:$REPO/third_party/Agent-R1" VLLM_WORKER_MULTIPROC_METHOD=spawn \
    python -m ssrl.checks.logprob_consistency --split "$SPLITS_DIR/dev_select.parquet" \
    --corpus-dir "$HOTPOTQA_CORPUS_DATA_ROOT" --model "$M4B" \
    --out "runs/$id" > "runs/$id/logprob.log" 2>&1 && mark "$id"
}

stage_tput() {
  local rc=0
  for p in ${TPUT_POLICIES:-c0 c2}; do
    local id="d3_tput_${p}_q3-4b-2507_lora_$TAG"
    done_mark "$id" && { log "skip $id"; continue; }
    mkdir -p "runs/$id"
    log "tput $p"
    local pid; pid=$(gpu_peak "$id")
    if MODEL="$M4B" TRAIN_BATCH="${TRAIN_BATCH:-64}" ROLLOUT_N="${ROLLOUT_N:-8}" MAX_PROMPT=8192 MAX_RESPONSE=1024 \
        MICRO_BS="${MICRO_BS:-4}" OFFLOAD="${OFFLOAD:-False}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}" \
        AGENT_WORKERS="${AGENT_WORKERS:-8}" SAVE_FREQ=-1 train "$id" 3 "$p"; then
      mark "$id"
    else
      log "tput $p FAILED"; rc=1
    fi
    kill "$pid" 2>/dev/null
    python - "runs/$id" <<'EOF'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
for f in sorted(run.glob("metrics.attempt*.jsonl")):
    for line in f.open():
        r = json.loads(line); d = r["data"]
        keys = ["timing_s/step", "timing_s/gen", "timing_s/update_actor", "perf/max_memory_allocated_gb",
                "perf/throughput", "response_length/mean", "prompt_length/mean"]
        print(r["step"], {k: round(d[k], 2) for k in keys if k in d})
EOF
  done
  return $rc
}

STAGES="${1:-all}"
[[ "$STAGES" == all ]] && STAGES="prep smoke resume probe logprob tput"
overall=0
for s in $STAGES; do
  log "=== stage $s ==="
  if "stage_$s"; then log "=== stage $s OK ==="; else log "=== stage $s FAILED ==="; overall=1; [[ "$s" == prep ]] && break; fi
done
log "D3_RUN_END overall=$overall"
exit $overall
