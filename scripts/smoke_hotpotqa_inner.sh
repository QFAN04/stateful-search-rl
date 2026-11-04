#!/usr/bin/env bash
# D1 冒烟：Agent-R1 HotpotQA recipe 单卡 GRPO。在容器内运行，由 scripts/smoke_hotpotqa.sh 调用。
# 复制自 third_party/Agent-R1/examples/hotpotqa/run_grpo.sh（b124aa4），按 docs/notes/agent_r1_interfaces.md §8 改为单卡。
# 注意：veRL 0.7 的 vLLMHttpServer 会用 hf_config.max_position_embeddings 覆盖 rollout.max_model_len
# （verl/workers/rollout/vllm_rollout/vllm_async_server.py:198），所以用 model.override_config 把它压到 prompt+response，
# 否则 vLLM 按 Qwen3 默认 40960 预留 KV cache，L4 上放不下。
# 必需环境变量：RUN_DIR STEPS MODEL RESUME_MODE VERL_FILE_LOGGER_PATH HOTPOTQA_CORPUS_DATA_ROOT
# 可选（D3 起，默认值 = D1 冒烟时的设置）：FLOW_CONFIG（相对仓库根目录）FLOW_NAME N_GPUS MAX_PROMPT MAX_RESPONSE
#   MICRO_BS OFFLOAD GPU_MEM_UTIL AGENT_WORKERS SAVE_FREQ
set -euxo pipefail
: "${RUN_DIR:?}" "${STEPS:?}" "${MODEL:?}" "${RESUME_MODE:?}" "${VERL_FILE_LOGGER_PATH:?}" "${HOTPOTQA_CORPUS_DATA_ROOT:?}"

N_GPUS="${N_GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N_GPUS - 1)))}"
export HOTPOTQA_EMBEDDING_PER_WORKER_GPU=0
export HOTPOTQA_EMBEDDING_DEVICE=cpu
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1

ROLLOUT_N="${ROLLOUT_N:-4}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"          # prompt 数；轨迹数 = TRAIN_BATCH * ROLLOUT_N
MAX_PROMPT="${MAX_PROMPT:-4096}" MAX_RESPONSE="${MAX_RESPONSE:-1024}"
MICRO_BS="${MICRO_BS:-2}" OFFLOAD="${OFFLOAD:-True}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
AGENT_WORKERS="${AGENT_WORKERS:-2}" SAVE_FREQ="${SAVE_FREQ:-5}"

cd "$(dirname "$0")/../third_party/Agent-R1"
PROJECT_DIR="$(pwd)"
export HOTPOTQA_DATA_ROOT="$PROJECT_DIR/data/corpus/hotpotqa"   # recipe 写死的相对位置（软链接到 /data）
# flow：默认 recipe 原版；E0–E3 用 configs/flows/ssrl_c*.yaml（我们的 flow 在 src/ssrl，Ray worker 也要能 import）
FLOW_CONFIG="${FLOW_CONFIG:-$PROJECT_DIR/recipes/hotpotqa/base.yaml}"
[[ "$FLOW_CONFIG" = /* ]] || FLOW_CONFIG="$PROJECT_DIR/../../$FLOW_CONFIG"
FLOW_NAME="${FLOW_NAME:-hotpotqa_agent}"
PYPATH="$PROJECT_DIR/../../src:$PROJECT_DIR"
export PYTHONPATH="$PYPATH:${PYTHONPATH:-}"

python3 -m agent_r1.trainer.main_agent_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="$HOTPOTQA_DATA_ROOT/train.parquet" \
    data.val_files="[\"$HOTPOTQA_DATA_ROOT/validation.parquet\"]" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=$MAX_RESPONSE \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BS \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    +actor_rollout_ref.model.override_config.max_position_embeddings=$((MAX_PROMPT + MAX_RESPONSE)) \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$FLOW_CONFIG" \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_WORKERS \
    actor_rollout_ref.rollout.agent.default_agent_flow="$FLOW_NAME" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYPATH" \
    actor_rollout_ref.rollout.trace.backend=null \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    critic.enable=False \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.99 \
    reward_model.enable=False \
    custom_reward_function.path=recipes/hotpotqa/reward_fn.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name=ssrl \
    trainer.experiment_name="$(basename "$RUN_DIR")" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_training_steps="$STEPS" \
    trainer.total_epochs=1 \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.rollout_data_dir="$RUN_DIR/rollouts" \
    "$@"
