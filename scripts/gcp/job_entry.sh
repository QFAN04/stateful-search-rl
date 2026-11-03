#!/usr/bin/env bash
# A100 每次开机自动执行（由 a100_startup.sh 以普通用户在后台启动，不依赖 SSH）。
# 1. 从 GCS 读任务定义 control/job.env（RUN_ID、JOB_CMD）；已有 DONE / FAILED 标记则直接关机
# 2. 从 GCS 拉最新代码包 code/repo.bundle 并更新仓库
# 3. 跑初始化（按步骤标记跳过已完成部分）和任务命令（任务本身必须可续跑）
# 4. 成功：写 DONE、同步、关机；失败：写 FAILED、同步、关机（避免轮询机反复拉起一个必然失败的任务）
# 被 Spot 抢占时本脚本直接中断，下次开机从第 1 步重来，由任务自身的续跑逻辑接上进度。
set -uo pipefail
B=gs://ssrl-runs-gqf-2026
REPO=~/work/stateful-search-rl
log() { echo "[$(date -u +%FT%TZ)] $*"; }
finish() {  # $1 = DONE | FAILED
  local dir="$REPO/runs/$RUN_ID"; mkdir -p "$dir"; cp ~/job.log "$dir/job.log" 2>/dev/null || true
  bash "$REPO/scripts/gcp/sync_to_gcs.sh" "$RUN_ID" "$REPO/runs" || log "WARN: final sync failed"
  echo "$1 $(date -u +%FT%TZ)" | gcloud storage cp - "$B/runs/$RUN_ID/$1"
  log "marker $1 written; powering off"; sudo poweroff
}

log "boot job start"
gcloud storage cp "$B/control/job.env" ~/job.env || { log "no job.env, idle"; exit 0; }
source ~/job.env   # RUN_ID, JOB_CMD
for m in DONE FAILED; do
  if gcloud storage objects describe "$B/runs/$RUN_ID/$m" >/dev/null 2>&1; then log "$m marker exists"; sudo poweroff; exit 0; fi
done

gcloud storage cp "$B/code/repo.bundle" ~/repo.bundle || { log "bundle download failed"; exit 1; }
if [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" fetch -q ~/repo.bundle main && git -C "$REPO" reset -q --hard FETCH_HEAD
else
  mkdir -p ~/work && git clone -q -b main ~/repo.bundle "$REPO"
fi
cp "$REPO/scripts/gcp/setup_vm.sh" "$REPO/scripts/gcp/bootstrap_d1.sh" ~/
log "repo at $(git -C "$REPO" rev-parse --short HEAD)"

bash ~/bootstrap_d1.sh > ~/bootstrap.log 2>&1 || { log "bootstrap failed"; tail -20 ~/bootstrap.log; finish FAILED; exit 1; }
cd "$REPO"
log "run: $JOB_CMD"
if bash -c "$JOB_CMD"; then finish DONE; else log "job failed rc=$?"; finish FAILED; fi
