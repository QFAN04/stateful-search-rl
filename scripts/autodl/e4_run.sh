#!/usr/bin/env bash
# E4：C2 + 笔记过程奖励（λ=0.2），80 步低成本方案；对照 E2 的 step_80。实验卡 docs/experiment_cards/e4_e5_candidates.md（开跑前冻结）。
# 单卡 RTX PRO 6000（西北B，私有镜像）。数据、模型、E2 adapter、OOD 语料从文件存储 /root/autodl-fs 拷入。
# 用法（仓库根目录，nohup 后台跑）：
#   bash scripts/autodl/e4_run.sh prep    # 环境、数据、模型、OOD 语料（复用 d9_eval.sh prep）+ E2 step_80 adapter + supporting fact 标题表
#   bash scripts/autodl/e4_run.sh smoke   # 0.6B、4 步：笔记奖励确实进了 reward（每条轨迹 score = acc + 0.2·note_bonus）
#   bash scripts/autodl/e4_run.sh train   # 4B E4 训练 80 步，每 20 步存档 + dev 评测（开训前也评一次）
#   bash scripts/autodl/e4_run.sh eval    # E4 按 dev EM 选中的 adapter + E2 step_80，各在 test / MuSiQue / 2Wiki 上 5 步评测
#   bash scripts/autodl/e4_run.sh all     # 以上依次执行，任一阶段失败就停；结束后结果拷到文件存储，30 分钟后关机（touch /root/autodl-tmp/NO_SHUTDOWN 取消）
# all 另设 14 小时硬上限（无论成败到点关机）。
set -uo pipefail
REPO=/root/autodl-tmp/stateful-search-rl
DATA=/root/autodl-tmp/data
FS=/root/autodl-fs
SPLITS_DIR="$DATA/hotpotqa_splits"
OOD="$DATA/ood"
TAG="${TAG:-pro6000_$(TZ=Asia/Shanghai date +%Y%m%d)}"
M4B="$DATA/models/Qwen3-4B-Instruct-2507"
M06="$DATA/models/Qwen3-0.6B"
SF="$DATA/hotpotqa_sf_titles.json"
SF_SHA=67a20d694990321ac6a95d30723538cdbcf4bbf109f50cbc3b3410e8a2af8122   # scripts/build_supporting_facts.py 本地生成
E2_RUN=e2_c2_q3-4b-2507_lora_pro6000_20261106
E4_ID="e4_c2nb_q3-4b-2507_lora_$TAG"
STEPS=80 SAVE_FREQ=20 MAX_HOURS=14
cd "$REPO"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-base}"
export SSRL_SF_TITLES="$SF"
log() { echo "[$(date '+%F %T')] $*"; }
disk_free() { df -h /root/autodl-tmp | awk 'NR==2{print $4}'; }

stage_prep() {
  bash scripts/autodl/d9_eval.sh prep || return 1      # 环境、单测、4B 模型、HotpotQA 切分 + test 哈希、OOD 切分 + 语料 / 索引
  if [[ ! -f "$M06/config.json" ]]; then
    mkdir -p "$M06"
    { [[ -d "$FS/ssrl-data/models/Qwen3-0.6B" ]] && cp -a "$FS/ssrl-data/models/Qwen3-0.6B/." "$M06/"; }       || modelscope download --model Qwen/Qwen3-0.6B --local_dir "$M06" || return 1
  fi
  local a="runs/$E2_RUN/adapters/step_80"
  [[ -f "$a/adapter_config.json" ]] || { mkdir -p "$(dirname "$a")" && cp -a "$FS/ssrl-runs/$E2_RUN/adapters/step_80" "$a"; } || return 1
  [[ -s "$SF" ]] || cp -a "$FS/ssrl-data/hotpotqa_sf_titles.json" "$SF" || return 1
  echo "$SF_SHA  $SF" | sha256sum -c - || { log "supporting fact 标题表与本地生成的不一致"; return 1; }
  PYTHONPATH="$REPO/src:$REPO/third_party/Agent-R1" python -m pytest -q -p no:cacheprovider tests/test_note_bonus.py | tail -2
  ls -la "$a"; df -h /root/autodl-tmp | tail -1
}

evict_cache() {  # D9 经验：2Wiki 索引的页缓存会把容器内存顶到上限，加载索引前先清掉大文件的页缓存
  python - <<'PY'
import glob, os
for f in glob.glob('/root/autodl-tmp/data/ood/*/*') + glob.glob('/root/autodl-tmp/data/agent-r1-data/hotpotqa_corpus/*'):
    if os.path.isfile(f):
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
PY
}

stage_smoke() {
  local id="e4smoke_c2nb_q3-0p6b_lora_$TAG" run
  run="runs/$id"
  [[ -f "$run/STAGE_DONE" ]] && { log "skip $id"; return 0; }
  mkdir -p "$run"
  log "=== smoke $id ==="
  CUDA_VISIBLE_DEVICES=0 N_GPUS=1 FLOW_CONFIG="configs/flows/ssrl_c2_e4.yaml" FLOW_NAME="ssrl_c2_e4" MODEL="$M06" \
  TRAIN_BATCH=8 ROLLOUT_N=8 MAX_PROMPT=8192 MAX_RESPONSE=1024 MICRO_BS=4 OFFLOAD=False GPU_MEM_UTIL=0.5 \
  AGENT_WORKERS=8 SAVE_FREQ=-1 RESUME_MODE=disable \
    bash scripts/autodl/smoke_lora.sh "$id" 4 trainer.val_before_train=False > /dev/null
  tail -1 "$run/meta.txt" | grep -q "^exit=0 " || { log "smoke training FAILED"; return 1; }
  # 检查：每个 rollout 都带 note_bonus；终局 score = acc + 0.2·note_bonus；问题都能在标题表里查到
  python - "$run/rollouts" "$SF" <<'PY' || { log "smoke check FAILED"; return 1; }
import glob, json, sys
sf = json.load(open(sys.argv[2], encoding='utf-8'))
n = bad = pos = miss = 0
for f in glob.glob(sys.argv[1] + '/*.jsonl'):
    for line in open(f, encoding='utf-8'):
        r = json.loads(line); last = r['steps'][-1]
        assert 'note_bonus' in last, 'note_bonus 不在 reward_extra_info 里'
        q = r['steps'][0]['input'].split('### User Query')[1].split('###')[0].strip()
        miss += q not in sf
        exp = float(last['acc']) + 0.2 * float(last['note_bonus'])
        bad += abs(float(last.get('score', r['score'])) - exp) > 1e-6
        pos += float(last['note_bonus']) > 0
        n += 1
print(f'smoke rollouts={n} score_mismatch={bad} bonus>0={pos} question_not_in_sf={miss}')
sys.exit(0 if n > 0 and bad == 0 and miss == 0 else 1)
PY
  date -u +%FT%TZ > "$run/STAGE_DONE"; log "=== smoke OK ==="
}

keep_adapters() {  # 与 e_run.sh 相同：每次存档后把 LoRA adapter 复制出来
  local run="$1"
  while true; do
    local it; it=$(cat "$run/ckpt/latest_checkpointed_iteration.txt" 2>/dev/null)
    if [[ -n "$it" && ! -d "$run/adapters/step_$it" && -d "$run/ckpt/global_step_$it/actor/lora_adapter" ]]; then
      mkdir -p "$run/adapters"; rm -rf "$run/adapters/.tmp_$it"
      cp -a "$run/ckpt/global_step_$it/actor/lora_adapter" "$run/adapters/.tmp_$it" && mv "$run/adapters/.tmp_$it" "$run/adapters/step_$it" \
        && log "saved adapter step_$it; disk free $(disk_free)"
    fi
    sleep 60
  done
}

stage_train() {
  local run="runs/$E4_ID"
  [[ -f "$run/STAGE_DONE" ]] && { log "skip $E4_ID"; return 0; }
  mkdir -p "$run"
  log "=== train $E4_ID (disk free $(disk_free)) ==="
  nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 30 >> "$run/gpu_mem.csv" &
  local gpu_pid=$!
  keep_adapters "$run" > >(tee -a "$run/adapters.log") 2>&1 &
  local keep_pid=$!
  # 与 E2 完全相同的训练参数（e_run.sh），只换 flow（ssrl_c2_e4 = ssrl_c2 + 笔记奖励）和总步数（80）
  CUDA_VISIBLE_DEVICES=0 N_GPUS=1 FLOW_CONFIG="configs/flows/ssrl_c2_e4.yaml" FLOW_NAME="ssrl_c2_e4" MODEL="$M4B" \
  TRAIN_BATCH=64 ROLLOUT_N=8 MAX_PROMPT=8192 MAX_RESPONSE=1024 MICRO_BS=4 OFFLOAD=False GPU_MEM_UTIL=0.5 \
  AGENT_WORKERS=8 SAVE_FREQ=$SAVE_FREQ RESUME_MODE=auto \
    bash scripts/autodl/smoke_lora.sh "$E4_ID" "$STEPS" \
      data.val_files="[\"$SPLITS_DIR/dev_select.parquet\"]" \
      trainer.test_freq=$SAVE_FREQ \
      trainer.val_before_train=True \
      trainer.validation_data_dir="$REPO/$run/val" \
      actor_rollout_ref.rollout.val_kwargs.n=1 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=False \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
      actor_rollout_ref.rollout.val_kwargs.top_p=1.0 > /dev/null
  sleep 70
  kill "$gpu_pid" "$keep_pid" 2>/dev/null
  local s ok=0
  for s in 20 40 60 80; do [[ -d "$run/adapters/step_$s" ]] || { log "missing adapter step_$s"; ok=1; }; done
  if tail -1 "$run/meta.txt" | grep -q "^exit=0 " && [[ $ok == 0 ]]; then
    date -u +%FT%TZ > "$run/STAGE_DONE"; rm -rf "$run/ckpt"; log "=== train $E4_ID OK ==="
  else
    log "=== train $E4_ID FAILED（checkpoint 保留，可同命令续训）==="; return 1
  fi
}

select_step() {  # 选择规则（与主实验相同）：dev EM 最高的 step（不含 step 0），并列取更早
  python - "runs/$E4_ID" <<'PY'
import glob, json, sys
best = None
for f in sorted(glob.glob(sys.argv[1] + '/metrics.attempt*.jsonl')):
    for line in open(f):
        o = json.loads(line)
        acc = next((v for k, v in o['data'].items() if k.startswith('val-core') and k.endswith('/acc/mean@1')), None)
        if acc is not None and o['step'] > 0 and (best is None or acc > best[1] or (acc == best[1] and o['step'] < best[0])):
            best = (o['step'], acc)
print(best[0])
PY
}

eval_one() {  # eval_one <label> <adapter> <dataset>；评测走 D9 同一路径，flow 用 ssrl_c2（只算 EM，与 E2 / D9 口径一致）
  local label="$1" adapter="$2" ds="$3"
  local id="d9e4_${ds}_${label}_q3-4b-2507_lora_$TAG" run vf corpus workers=8
  run="runs/$id"
  [[ -f "$run/STAGE_DONE" ]] && { log "skip $id"; return 0; }
  case "$ds" in
    test) vf="$SPLITS_DIR/test.parquet"; corpus="$DATA/agent-r1-data/hotpotqa_corpus" ;;
    musique) vf="$OOD/splits/musique_500.parquet"; corpus="$OOD/musique_corpus" ;;
    2wiki) vf="$OOD/splits/2wiki_500.parquet"; corpus="$OOD/2wikimultihopqa_corpus"; workers=2; evict_cache ;;
  esac
  mkdir -p "$run"
  echo "e4eval label=$label adapter=$adapter dataset=$ds val_file=$vf agent_workers=$workers" >> "$run/meta.txt"
  log "=== eval $id ==="
  CUDA_VISIBLE_DEVICES=0 N_GPUS=1 HOTPOTQA_CORPUS_DATA_ROOT="$corpus" FLOW_CONFIG="configs/flows/ssrl_c2.yaml" FLOW_NAME="ssrl_c2" MODEL="$M4B" \
  TRAIN_BATCH=8 ROLLOUT_N=8 MAX_PROMPT=8192 MAX_RESPONSE=1024 MICRO_BS=4 OFFLOAD=False GPU_MEM_UTIL=0.5 \
  AGENT_WORKERS=$workers SAVE_FREQ=-1 RESUME_MODE=disable \
    bash scripts/autodl/smoke_lora.sh "$id" 1 \
      data.val_files="[\"$vf\"]" actor_rollout_ref.model.lora_adapter_path="$adapter" \
      trainer.val_before_train=True trainer.val_only=True trainer.validation_data_dir="$REPO/$run/val" \
      actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=False \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 > /dev/null
  if tail -1 "$run/meta.txt" | grep -q "^exit=0 " && grep -q "val-core" "$run"/metrics.attempt*.jsonl; then
    date -u +%FT%TZ > "$run/STAGE_DONE"
    log "=== eval $id OK: $(grep -ho '"val-core[^,]*acc/mean@1": [0-9.]*' "$run"/metrics.attempt*.jsonl | tail -1) ==="
  else
    log "=== eval $id FAILED ==="; return 1
  fi
}

stage_eval() {
  local s rc=0
  s=$(select_step) || return 1
  log "E4 selected step $s (dev EM rule)"; echo "selected_step=$s" > "runs/$E4_ID/SELECTED"
  for ds in test musique 2wiki; do
    eval_one e4s$s "$REPO/runs/$E4_ID/adapters/step_$s" "$ds" || rc=1
    eval_one e2s80 "$REPO/runs/$E2_RUN/adapters/step_80" "$ds" || rc=1
  done
  return $rc
}

sync_fs() {
  mkdir -p "$FS/ssrl-runs" && cp -a runs/e4_c2nb_* runs/e4smoke_* runs/d9e4_* "$FS/ssrl-runs/" 2>/dev/null; sync
  log "synced runs to $FS/ssrl-runs"
}

case "${1:-}" in
  prep|smoke|train|eval) "stage_$1" ;;
  all)
    ( sleep $(( MAX_HOURS * 3600 )); log "MAX_HOURS=$MAX_HOURS reached, shutdown"; sync_fs; shutdown -h now ) &
    HARD_PID=$!
    overall=0
    stage_prep && stage_smoke && stage_train && stage_eval || overall=1
    sync_fs
    log "E4_END overall=$overall; shutdown in 30 min unless /root/autodl-tmp/NO_SHUTDOWN exists"
    sleep 1800
    [[ -f /root/autodl-tmp/NO_SHUTDOWN ]] && { log "NO_SHUTDOWN found, stay up (hard cap still active)"; exit $overall; }
    kill "$HARD_PID" 2>/dev/null
    shutdown -h now
    ;;
  *) sed -n 2,10p "$0"; exit 2 ;;
esac
