#!/usr/bin/env bash
# 跨实例搬运未完成的 run（含 checkpoint），经 AutoDL 文件存储中转（同一地区，西北B）。
#   旧实例：先停训练（kill e_run.sh 和训练进程，或等它在存档后被中断），再 bash scripts/autodl/run_cache.sh save <run_id>
#   新实例：bash scripts/autodl/run_cache.sh load <run_id>，然后用同样的 TAG 重跑 e_run.sh <policy>，从最近 checkpoint 续训
# 每组固定单卡，checkpoint 是 world_size=1 的格式，换到任何一台 RTX PRO 6000 的任意一张卡上都能续训；路径在所有实例上相同。
# 只搬最近一个完整 checkpoint（够续训），其余步的 LoRA adapter 已在 runs/<id>/adapters/。
set -euo pipefail
REPO=/root/autodl-tmp/stateful-search-rl
CACHE=/root/autodl-fs/ssrl-runs
mode="$1" id="$2"
src="$REPO/runs/$id" dst="$CACHE/$id"
case "$mode" in
  save)
    it=$(cat "$src/ckpt/latest_checkpointed_iteration.txt")
    mkdir -p "$dst"
    # 除 ckpt 外的所有文件 + 最近一个 checkpoint 和 latest 指针
    tar -C "$src" --exclude=./ckpt -cf - . | tar -C "$dst" -xf -
    mkdir -p "$dst/ckpt" && cp -a "$src/ckpt/global_step_$it" "$src/ckpt/latest_checkpointed_iteration.txt" "$dst/ckpt/"
    sync
    echo "saved $id (latest step $it): $(du -sh "$dst" | cut -f1)"
    ;;
  load)
    [[ -e "$src" ]] && { echo "$src already exists; move it away first" >&2; exit 1; }
    mkdir -p "$src" && cp -a "$dst/." "$src/"
    echo "loaded $id (latest step $(cat "$src/ckpt/latest_checkpointed_iteration.txt")): $(du -sh "$src" | cut -f1)"
    ;;
  *) echo "usage: run_cache.sh save|load <run_id>" >&2; exit 2 ;;
esac
