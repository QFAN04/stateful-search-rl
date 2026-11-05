#!/usr/bin/env bash
# E0 开跑前在 2 卡机器上做一次的检查（每组 2 卡的正式配置第一次上机）。判断标准见 docs/experiment_cards/e0_e3_main.md「2 卡验收」。
# 用法（实例内）：nohup bash scripts/autodl/e_preflight.sh > /root/autodl-tmp/e_preflight.log 2>&1 &
# 阶段（任一失败继续往下跑，最后汇总；不自动关机，等结果判断后直接开跑 E0；硬上限 4 小时）：
#   setup   数据和模型从文件存储拷到数据盘（setup_env.sh，已完成的步骤跳过）+ d3_run.sh prep（清理、单测）
#   smoke   E_SMOKE=1 e_run.sh c0 c2：0.6B、2 卡、每组 4 步，第 2/4 步存档 + 评测 + adapter 保存；再做严格回放检查
#   resume  0.6B C2 2 卡：10 步 → 续到 12 步，check_resume.py
#   calib   4B C0 2 卡，32×8 和 64×8 各 3 步（不存档）：批大小校准 + 1 卡→2 卡提速 + 有效题数
set -uo pipefail
REPO=/root/autodl-tmp/stateful-search-rl
cd "$REPO"
DAY=$(TZ=Asia/Shanghai date +%Y%m%d)
log() { echo "[$(date '+%F %T')] $*"; }
( sleep $(( 4 * 3600 )); log "preflight hard cap 4h, shutdown"; shutdown -h now ) &
HARD_PID=$!
echo $HARD_PID > /root/autodl-tmp/preflight_cap.pid   # e_run.sh 正式开跑时会取消这个上限
export N_GPUS=2
rc=0
step() { log "=== $1 ==="; if "${@:2}"; then log "=== $1 OK ==="; else log "=== $1 FAILED ==="; rc=1; fi; }

stage_setup() {
  bash scripts/autodl/setup_env.sh > /root/autodl-tmp/setup_env.log 2>&1 || { tail -20 /root/autodl-tmp/setup_env.log; return 1; }
  bash scripts/autodl/d3_run.sh prep
}

stage_smoke() {
  local ok=0 p id
  E_SMOKE=1 TAG="2gpu_pro6000_$DAY" bash scripts/autodl/e_run.sh c0 c2 || ok=1
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate "${ENV_NAME:-base}"
  for p in c0 c2; do
    id="esmoke_${p}_q3-0p6b_lora_2gpu_pro6000_$DAY"
    python scripts/check_replay.py "runs/$id/rollouts" --expect-steps 4 --expect-trajs 64 > "runs/$id/replay_check.txt" 2>&1 || ok=1
    tail -2 "runs/$id/replay_check.txt"; ls "runs/$id/adapters" 2>/dev/null | tr '\n' ' '; echo
  done
  return $ok
}

stage_resume() {
  TAG="2gpu_pro6000_$DAY" bash scripts/autodl/d3_run.sh resume
}

stage_calib() {
  local ok=0 b
  for b in 32 64; do
    TRAIN_BATCH=$b TPUT_POLICIES=c0 TAG="b${b}_2gpu_pro6000_$DAY" bash scripts/autodl/d3_run.sh tput || ok=1
  done
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate "${ENV_NAME:-base}"
  python scripts/count_mixed_groups.py runs/d3_tput_c0_q3-4b-2507_lora_b{32,64}_2gpu_pro6000_$DAY
  for b in 32 64; do
    grep -o "timing_s/step:[0-9.]*\|timing_s/gen:[0-9.]*\|timing_s/update_actor:[0-9.]*\|training/rollout_probs_diff_mean:[0-9.]*" \
      "runs/d3_tput_c0_q3-4b-2507_lora_b${b}_2gpu_pro6000_$DAY/train.attempt1.log" | paste -sd' ' | sed "s/^/b$b: /"
  done
  return $ok
}

for s in ${STAGES:-setup smoke resume calib}; do
  step "$s" "stage_$s"
  [[ "$s" == setup && $rc -ne 0 ]] && break
done
log "PREFLIGHT_END rc=$rc（不自动关机；硬上限 4 小时）"
