#!/usr/bin/env bash
# D9：封存 test + OOD + 放宽轮数评测（实验卡 docs/experiment_cards/d9_sealed_eval.md，开跑前冻结）。
# 单卡 RTX PRO 6000（西北B，私有镜像），数据、模型、adapter、OOD 语料从文件存储 /root/autodl-fs 拷入。
# 用法（仓库根目录，nohup 后台跑）：
#   bash scripts/autodl/d9_eval.sh prep       # 环境、模型、切分、adapter、OOD 数据就位（含 manifest 哈希核对）
#   bash scripts/autodl/d9_eval.sh index      # 建 MuSiQue 语料 + 索引、2Wiki 索引（GPU），结果回存文件存储
#   bash scripts/autodl/d9_eval.sh devcheck   # 评测路径一致性检查：C0 step_80 在 dev_select 上应复现 0.526（±2pp）
#   bash scripts/autodl/d9_eval.sh main       # 四组 × {test, musique, 2wiki} × {5 步, 10 步}
#   bash scripts/autodl/d9_eval.sh baselines  # B0/B1 × {test, musique, 2wiki}
#   bash scripts/autodl/d9_eval.sh all        # 以上依次执行；devcheck 不通过就停（不碰封存数据）
# 每个评测 run 完成后写 STAGE_DONE，重跑会跳过已完成的。all 结束后 30 分钟关机（touch /root/autodl-tmp/NO_SHUTDOWN 取消）。
set -uo pipefail
REPO=/root/autodl-tmp/stateful-search-rl
DATA=/root/autodl-tmp/data
FS=/root/autodl-fs
SPLITS_DIR="$DATA/hotpotqa_splits"
OOD="$DATA/ood"                       # 本地：splits/、musique_corpus/、2wikimultihopqa_corpus/
TAG="${TAG:-pro6000_$(TZ=Asia/Shanghai date +%Y%m%d)}"
MODEL="$DATA/models/Qwen3-4B-Instruct-2507"
EMB="$DATA/models/bge-large-en-v1.5"
cd "$REPO"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-base}"
log() { echo "[$(date '+%F %T')] $*"; }

# 实验卡登记的选中 adapter（e0_e3_main 选择规则），不再改
declare -A SRC_RUN=([c0]=e0_c0_q3-4b-2507_lora_pro6000_20261106 [c1]=e1_c1_q3-4b-2507_lora_pro6000_20261106
                    [c2]=e2_c2_q3-4b-2507_lora_pro6000_20261106 [c3]=e3_c3_q3-4b-2507_lora_pro6000_20261106)
declare -A SEL_STEP=([c0]=80 [c1]=150 [c2]=150 [c3]=140)

val_file() { case "$1" in
  dev) echo "$SPLITS_DIR/dev_select.parquet" ;; test) echo "$SPLITS_DIR/test.parquet" ;;
  musique) echo "$OOD/splits/musique_500.parquet" ;; 2wiki) echo "$OOD/splits/2wiki_500.parquet" ;; esac; }
corpus_dir() { case "$1" in
  dev|test) echo "$DATA/agent-r1-data/hotpotqa_corpus" ;;
  musique) echo "$OOD/musique_corpus" ;; 2wiki) echo "$OOD/2wikimultihopqa_corpus" ;; esac; }

stage_prep() {
  # 与 e_preflight.sh 相同：先清理私有镜像的旧文件残留，再装 / 核对环境和数据（bge、语料、索引从文件存储拷）
  python scripts/autodl/clean_stale_site_packages.py || return 1
  bash scripts/autodl/setup_env.sh > /root/autodl-tmp/setup_env.log 2>&1 || { tail -20 /root/autodl-tmp/setup_env.log; return 1; }
  bash scripts/autodl/d3_run.sh prep || return 1                          # 4B 模型、HotpotQA 切分、单测
  cmp <(python -c "import json;print(json.load(open('data/manifests/hotpotqa_test.json'))['ids_sha256'])") \
      <(python - "$SPLITS_DIR/test.parquet" <<'EOF'
import hashlib, sys, pandas as pd
ids = pd.read_parquet(sys.argv[1])["extra_info"].map(lambda e: e["question_id"]).tolist()
print(hashlib.sha256("\n".join(ids).encode()).hexdigest())
EOF
) || { log "test.parquet 与 manifest 不一致"; return 1; }
  for p in c0 c1 c2 c3; do                                                 # 只拷选中的那一个 adapter
    local dst="runs/${SRC_RUN[$p]}/adapters/step_${SEL_STEP[$p]}"
    [[ -f "$dst/adapter_config.json" ]] || { mkdir -p "$(dirname "$dst")" && cp -a "$FS/ssrl-runs/${SRC_RUN[$p]}/adapters/step_${SEL_STEP[$p]}" "$dst"; } || return 1
    ls -la "$dst"
  done
  mkdir -p "$OOD/splits"
  python -I scripts/make_ood_splits.py "$FS/ssrl-data/ood/hotpotqa" "$OOD/splits" || return 1
  git diff --exit-code data/manifests/ood_*.json || { log "OOD 切分与已提交的 manifest 不一致"; return 1; }
  for d in musique_corpus 2wikimultihopqa_corpus; do                       # 已建好的语料 / 索引（index 阶段回存的）
    [[ -d "$FS/ssrl-data/ood/$d" ]] && { mkdir -p "$OOD/$d" && cp -an "$FS/ssrl-data/ood/$d/." "$OOD/$d/"; }
  done
  rm -f "$OOD"/*/hpqa_corpus.npy                                          # 编码缓存，运行时不用，省磁盘
  df -h /root/autodl-tmp | tail -1
}

stage_index() {
  local env=(env PYTHONPATH="$REPO/third_party/Agent-R1" HF_HUB_OFFLINE=1)
  # MuSiQue：语料已在本地用上游 build_retrieval_corpus.py 建好（125,450 段，见实验卡）并放在文件存储，这里只建索引
  if [[ ! -s "$OOD/musique_corpus/index.bin" ]]; then
    mkdir -p "$OOD/musique_corpus"
    [[ -s "$OOD/musique_corpus/hpqa_corpus.jsonl" ]] || cp -a "$FS/ssrl-data/ood/musique_corpus/hpqa_corpus.jsonl" "$OOD/musique_corpus/"
    "${env[@]}" python third_party/Agent-R1/recipes/hotpotqa/env/build_index.py --data_dir "$OOD/musique_corpus" \
      --corpus_path "$OOD/musique_corpus/hpqa_corpus.jsonl" --embedding_model "$EMB" --devices cuda:0 --batch_size 1024 || return 1
  fi
  # 2Wiki：上游发布的语料（约 590 万段），只建索引
  if [[ ! -s "$OOD/2wikimultihopqa_corpus/index.bin" ]]; then
    mkdir -p "$OOD/2wikimultihopqa_corpus"
    [[ -s "$OOD/2wikimultihopqa_corpus/hpqa_corpus.jsonl" ]] || cp -a "$FS/ssrl-data/ood/2wikimultihopqa_corpus/hpqa_corpus.jsonl" "$OOD/2wikimultihopqa_corpus/"
    "${env[@]}" python third_party/Agent-R1/recipes/hotpotqa/env/build_index.py --data_dir "$OOD/2wikimultihopqa_corpus" \
      --corpus_path "$OOD/2wikimultihopqa_corpus/hpqa_corpus.jsonl" --embedding_model "$EMB" --devices cuda:0 --batch_size 1024 || return 1
  fi
  rm -f "$OOD"/*/hpqa_corpus.npy
  for d in musique_corpus 2wikimultihopqa_corpus; do                       # 回存文件存储，换机器不用重建
    mkdir -p "$FS/ssrl-data/ood/$d" && cp -an "$OOD/$d/." "$FS/ssrl-data/ood/$d/"
    log "$d: $(wc -l < "$OOD/$d/hpqa_corpus.jsonl") passages, index $(du -sh "$OOD/$d/index.bin" | cut -f1)"
  done
}

eval_one() {  # eval_one <policy> <dataset> <x1:0|1>
  local p="$1" ds="$2" x1="$3"
  local suffix=""; [[ "$x1" == 1 ]] && suffix="_x1"
  local id="d9_${ds}${suffix}_${p}_q3-4b-2507_lora_$TAG"
  local run="runs/$id"
  [[ -f "$run/STAGE_DONE" ]] && { log "skip $id"; return 0; }
  mkdir -p "$run"
  local adapter="$REPO/runs/${SRC_RUN[$p]}/adapters/step_${SEL_STEP[$p]}"
  # 2Wiki 的 Flat 索引约 24GB，每个 agent worker 进程各加载一份：内存不够 250GB 时降到 2 个 worker（只影响速度）
  local workers=8 mem_g
  mem_g=$(( $(cat /sys/fs/cgroup/memory.max 2>/dev/null | grep -E '^[0-9]+$' || free -b | awk '/Mem/{print $2}') / 1024 / 1024 / 1024 ))
  [[ "$ds" == 2wiki && "$mem_g" -lt 250 ]] && workers=2
  workers="${D9_WORKERS:-$workers}"   # 崩溃重跑时手动降 worker（只影响速度，greedy 结果不变；记录在 meta.txt 和实验卡）
  log "=== eval $id adapter=step_${SEL_STEP[$p]} corpus=$(corpus_dir "$ds") workers=$workers mem=${mem_g}G ==="
  echo "d9 policy=$p dataset=$ds x1=$x1 adapter=$adapter val_file=$(val_file "$ds") agent_workers=$workers" >> "$run/meta.txt"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 >> "$run/gpu_mem.csv" &
  local gpu_pid=$!
  CUDA_VISIBLE_DEVICES=0 N_GPUS=1 HOTPOTQA_CORPUS_DATA_ROOT="$(corpus_dir "$ds")" \
  FLOW_CONFIG="configs/flows/ssrl_${p}${suffix}.yaml" FLOW_NAME="ssrl_${p}${suffix}" MODEL="$MODEL" \
  TRAIN_BATCH=8 ROLLOUT_N=8 MAX_PROMPT=8192 MAX_RESPONSE=1024 MICRO_BS=4 OFFLOAD=False GPU_MEM_UTIL=0.5 \
  AGENT_WORKERS=$workers SAVE_FREQ=-1 RESUME_MODE=disable \
    bash scripts/autodl/smoke_lora.sh "$id" 1 \
      data.val_files="[\"$(val_file "$ds")\"]" \
      actor_rollout_ref.model.lora_adapter_path="$adapter" \
      trainer.val_before_train=True \
      trainer.val_only=True \
      trainer.validation_data_dir="$REPO/$run/val" \
      actor_rollout_ref.rollout.val_kwargs.n=1 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=False \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
      actor_rollout_ref.rollout.val_kwargs.top_p=1.0 > /dev/null
  kill "$gpu_pid" 2>/dev/null
  if tail -1 "$run/meta.txt" | grep -q "^exit=0 " && grep -q "val-core" "$run"/metrics.attempt*.jsonl; then
    date -u +%FT%TZ > "$run/STAGE_DONE"
    log "=== eval $id OK: $(grep -ho '"val-core[^,]*' "$run"/metrics.attempt*.jsonl | tail -1) ==="
  else
    log "=== eval $id FAILED ==="; return 1
  fi
}

stage_devcheck() {
  eval_one c0 dev 0 || return 1
  local em
  em=$(grep -ho '"val-core[^"]*acc/mean@1": [0-9.]*' runs/d9_dev_c0_q3-4b-2507_lora_$TAG/metrics.attempt*.jsonl | tail -1 | awk '{print $NF}')
  python -c "import sys; em=float('$em'); ok=abs(em-0.526)<=0.02; print(f'devcheck EM={em:.3f} (训练时 0.526) ->', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)"
}

stage_main() {
  local rc=0
  for x1 in 0 1; do for ds in test musique 2wiki; do for p in c0 c1 c2 c3; do
    eval_one "$p" "$ds" "$x1" || rc=1
  done; done; done
  return $rc
}

stage_baselines() {
  local rc=0
  for ds in test musique 2wiki; do
    local id="d9_b0b1_${ds}_q3-4b-2507_$TAG"
    [[ -f "runs/$id/STAGE_DONE" ]] && { log "skip $id"; continue; }
    mkdir -p "runs/$id"
    log "=== baselines $id ==="
    # 检索器默认按 HF 名字加载 bge，离线实例要指向本地（与训练 / 评测时相同的权重）
    HF_HUB_OFFLINE=1 HOTPOTQA_EMBEDDING_MODEL="$EMB" python -I scripts/baselines_b0b1.py --split "$(val_file "$ds")" --corpus-dir "$(corpus_dir "$ds")" \
      --out "runs/$id" --model "$MODEL" > "runs/$id/run.log" 2>&1 \
      && date -u +%FT%TZ > "runs/$id/STAGE_DONE" && log "=== baselines $id OK ===" || { log "=== baselines $id FAILED ==="; rc=1; }
  done
  return $rc
}

case "${1:-}" in
  prep|index|devcheck|main|baselines) "stage_$1" ;;
  all)
    overall=0
    if stage_prep && stage_index && stage_devcheck; then
      stage_main || overall=1
      stage_baselines || overall=1
    else
      log "prep / index / devcheck FAILED：不跑封存数据"; overall=1
    fi
    log "D9_END overall=$overall; shutdown in 30 min unless /root/autodl-tmp/NO_SHUTDOWN exists"
    sleep 1800
    [[ -f /root/autodl-tmp/NO_SHUTDOWN ]] || shutdown -h now
    ;;
  *) sed -n 2,11p "$0"; exit 2 ;;
esac
