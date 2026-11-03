#!/usr/bin/env bash
# D2 Pre-GRPO probe（C0 = recipe 默认 flow）：只做 validation，不训练。在容器内运行，由 scripts/probe_c0.sh 调用。
# MODE=greedy：probe.parquet（300 题）贪心解码；MODE=sample：probe_x8.parquet（每题 8 行）按训练采样参数各采 1 条。
# 不用 val_kwargs.n>1：Agent-R1 _validate 会让同题 n 条轨迹共用 uid，落盘时合并成一条（见 scripts/make_splits.py）。
# 必需环境变量：RUN_DIR MODE MODEL SPLITS_DIR VERL_FILE_LOGGER_PATH HOTPOTQA_CORPUS_DATA_ROOT
set -euxo pipefail
: "${RUN_DIR:?}" "${MODE:?}" "${MODEL:?}" "${SPLITS_DIR:?}" "${VERL_FILE_LOGGER_PATH:?}" "${HOTPOTQA_CORPUS_DATA_ROOT:?}"

export CUDA_VISIBLE_DEVICES=0
export HOTPOTQA_EMBEDDING_PER_WORKER_GPU=0
export HOTPOTQA_EMBEDDING_DEVICE=cpu
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1

MAX_PROMPT=8192
MAX_RESPONSE=1024
case "$MODE" in
  greedy) VAL_FILE="$SPLITS_DIR/probe.parquet";    DO_SAMPLE=False; TEMP=0.0 ;;
  sample) VAL_FILE="$SPLITS_DIR/probe_x8.parquet"; DO_SAMPLE=True;  TEMP=1.0 ;;
  *) echo "MODE must be greedy|sample" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/../third_party/Agent-R1"
PROJECT_DIR="$(pwd)"
export HOTPOTQA_DATA_ROOT="$PROJECT_DIR/data/corpus/hotpotqa"

python3 -m agent_r1.trainer.main_agent_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$HOTPOTQA_DATA_ROOT/train.parquet" \
    data.val_files="[\"$VAL_FILE\"]" \
    data.train_batch_size=8 \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=$MAX_RESPONSE \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL" \
    +actor_rollout_ref.model.override_config.max_position_embeddings=$((MAX_PROMPT + MAX_RESPONSE)) \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$DO_SAMPLE \
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMP \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$PROJECT_DIR/recipes/hotpotqa/base.yaml" \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.default_agent_flow=hotpotqa_agent \
    actor_rollout_ref.rollout.trace.backend=null \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.enable=False \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    custom_reward_function.path=recipes/hotpotqa/reward_fn.py \
    custom_reward_function.name=compute_score \
    trainer.logger='["console","file"]' \
    trainer.project_name=ssrl \
    trainer.experiment_name="$(basename "$RUN_DIR")_$MODE" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$RUN_DIR/ckpt_unused" \
    trainer.validation_data_dir="$RUN_DIR/val_$MODE" \
    "$@"
