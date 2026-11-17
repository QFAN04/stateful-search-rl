# 实验卡：d3_autodl_envtest（AutoDL 环境 + 0.6B LoRA 冒烟）

> 环境验证，不是正式实验。目的：在 AutoDL 上不用 Docker 复刻训练镜像的版本组合，确认 LoRA 路径能跑通，同时验证 Blackwell（RTX 5090）兼容性。

## 设定
- run：`d3_autodl_envtest_q3-0p6b_lora_5090_20261104`（云归档 同名目录）
- 机器：AutoDL 西北区普通实例，1×RTX 5090 32GB（Blackwell，计算能力 12.0），驱动 595.71.05，16 核 / 92GB 内存；基础镜像 PyTorch 2.8 / Python 3.12 / CUDA 12.8 / Ubuntu 22.04
- 环境：`scripts/autodl/setup_env.sh` 装进 conda base：torch 2.9.0+cu129、vLLM 0.12.0、verl 0.7.0、transformers 4.57.3、ray 2.52.1、flash-attn 2.8.1，其余按 `env/autodl/constraints.txt`（由训练镜像 pip list 生成，去掉 opencv-python-headless 的版本锁：它要求 numpy≥2，与镜像里的 numpy 1.26.4 冲突，原镜像是强装的）。完整列表：run 目录 `pip-list-autodl.txt`
- 训练：`scripts/autodl/smoke_lora.sh` → `scripts/smoke_hotpotqa_inner.sh`，Qwen3-0.6B，LoRA rank 32 / alpha 32 / all-linear，lr 1e-5，`rollout.load_format=safetensors`，batch 8 题 × 4 条，10 步；其余同 D1 冒烟
- 与 D1（`d1_smoke_hotpotqa_q3-0p6b_20261103_0956`）的差别：全参 → LoRA（lr 1e-6 → 1e-5）、L4 → 5090、Docker 镜像 → conda 环境（版本相同）

## 结果
- 10 步跑完，退出码 0；严格回放检查 **PASS**（`--expect-steps 10 --expect-trajs 32`，320 条轨迹无异常，`replay_check.txt`）
- 每步耗时：第 1 步 93.0s（含编译预热），之后 30.1–33.1s（D1 在 L4 上全参约 46.3s/步）
- 峰值显存：allocated 18.2GB / reserved 19.4GB（vLLM `gpu_memory_utilization=0.5`）
- 训推一致性 `training/rollout_probs_diff_mean`：0.0059–0.0075（D1：0.0062–0.0071），同一量级
- 终止原因：answer 147、无 answer 无 tool_call 170、tool_call 解析失败 3（0.6B 格式能力弱，与 D1 情况相同，不作为结论）
- vLLM 注意力后端为 `auto`，未报 FlashAttention 版本错误；Gemini 建议的 `VLLM_FLASH_ATTN_VERSION=2` 不需要
- 结束时有一条 "DataLoader worker killed by signal" 的 traceback，出现在训练 100% 并保存 global_step_10 之后，属于退出清理；D1 attempt1 也有同样一条
- checkpoint 两个（step 5、10）共 6.3GB，留在实例数据盘，未回传

## 结论
- AutoDL + conda 环境可以替代 Docker 镜像；LoRA 路径（权重同步到 vLLM、训练、存档）在 0.6B 上能跑通；5090（Blackwell）兼容。
- 未覆盖：LoRA 断点续训、4B 吞吐、多卡、迁移验收（C0 probe EM 对比 0.4367、4B logprob 一致性）。

## 坑（已写进脚本）
- 无卡模式只有 0.5 核 / 2GB，pip 被 CPU 卡在约 1MB/s：装环境要有卡开机
- pip 用腾讯云 HTTPS 镜像；torch 大包用 curl 从阿里云 `pytorch-wheels/cu129` 断点续传
- ModelScope 下载 BGE 会同时拉 safetensors / pytorch_model.bin / onnx 三份，只需要 safetensors
- 非交互 SSH 没有 PATH：远程命令走 `bash -l` 或先 `source conda.sh`；远程 `pkill -f` 的模式不能出现在自己的命令行里（用 `bash -s` 从 stdin 传命令）
- 系统盘会被存进镜像：pip 缓存装完即清；TMPDIR / RAY_TMPDIR / HF_HOME 指向数据盘
