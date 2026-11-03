# Agent-R1 接口笔记（D1 用）

> 写于 2026-11-03。只读源码，未运行任何代码。所有"未确认"项须在 D1 VM 上实测后回填 `env/GPU_ENV.md`。
> 行号均指下文固定 commit；veRL 行号指 v0.7.0 tag。

## 0. 固定版本

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/AgentR1/Agent-R1 （默认分支 `main`；旧实现在 `legacy` 分支） |
| 固定 commit | `b124aa46534cbf2fb8bc8af11405774984c42ac7`（2026-07-21 19:39 +0800，"docs: announce OPD support (#103)"；`git ls-remote` 确认为远端 HEAD） |
| veRL 依赖 | **没有 submodule**，README 与 `docs/getting-started/installation-guide.md:7` 要求 `verl==0.7.0`（PyPI 有 0.7.0，2026-01-05 发布；也有 0.7.1，不要用） |
| veRL v0.7.0 tag | `f9c855f7cf04d603c9546bc01776c74806a879c1`（2026-01-05）。注意 tag 内 `verl/version/version` 写的是 `0.7.0.dev`，以 PyPI 包为准 |
| 上游动向 | PR #100 "adapt Agent-R1 to verl 0.8.0" 仍 open。合并后 `main` 会换到 0.8 API → **必须固定上面的 SHA，不要跟 main** |

Agent-R1 本身不需要安装：从仓库根目录用 `python3 -m agent_r1.trainer.main_agent_ppo` 启动，`recipes.*` 靠 cwd 导入。

## 1. step-level 架构：类与方法

### 1.1 层次（实际名称）

| 层 | 类 | 文件 |
|---|---|---|
| 抽象 flow | `AgentFlowBase`（`run()` 抽象方法，`_postprocess()` 负责 pad + 算分） | `agent_r1/agent_flow/agent_flow.py:129`、`:253`、`:265` |
| 每步数据 | `AgentFlowStep`（pydantic） | `agent_flow.py:64-90` |
| 通用环境循环 | `AgentEnvLoop`（`@register("agent_env_loop")`） | `agent_r1/agent_flow/agent_env_loop.py:21-169` |
| 环境接口 | `AgentEnv`：`reset(**kwargs) -> Observation`、`async step(action) -> (Observation, reward, done, info)` | `agent_r1/env/base.py:39-102` |
| 内置工具环境 | `ToolEnv`（`@AgentEnv.register("tool")`） | `agent_r1/env/envs/tool.py:10-113` |
| 观测/动作 | `Observation(text \| messages \| token_ids)`、`Action(text, token_ids)` | `env/base.py:6-36` |
| **HotpotQA recipe 实际用的** | `HotpotQAAgentFlow(AgentFlowBase)`（`@register("hotpotqa_agent")`）——**不走 AgentEnvLoop/ToolEnv** | `recipes/hotpotqa/hotpotqa_agent_flow.py:146-433` |
| Ray worker | `AgentFlowWorkerBase` / `AgentFlowWorker`、`AgentFlowManager` | `agent_flow.py:468`、`:782`、`:820` |

flow 的注册：`actor_rollout_ref.rollout.agent.agent_flow_config_path` 指向一个 YAML 列表，每项 `name` + `_target_` + 自定义 kwargs（`AgentFlowWorkerBase.__init__`，`agent_flow.py:498-503`）；每条轨迹都会 `hydra.utils.instantiate` 一个新的 flow 实例（`agent_flow.py:626`），所以 flow 实例 = 单条轨迹的状态容器。

### 1.2 observation 由谁构造

- `AgentEnvLoop` 路径：**环境**构造。`env.step()` 返回完整的下一轮输入 `Observation`，循环里 `_obs_to_prompt()`（`agent_env_loop.py:70-98`）把它变成 `prompt_ids`。`ToolEnv` 的默认行为是 append：把 assistant 原文和工具结果追加到 `self._messages`（`tool.py:79`、`:105`），注释里写明还不处理 thinking 内容（`tool.py:107`）。
- `HotpotQAAgentFlow` 路径：**flow 自己**构造。每步从结构化状态（`passages`、`history_actions`、`tool_feedback`）重新渲染 `[system, user]`（`_build_messages`，`hotpotqa_agent_flow.py:205-231`），超长时只缩短 passage 段（`_prompt_ids_within_budget`，`:233-268`）。**模型自己写的 `<analysis>` 不会出现在下一步的 prompt 里**。

结论：recipe 的默认上下文既不是我们的 C0（全量 append），也不是 C1；更接近"问题 + 历史 query + 去重累积的全部 passage"。我们的 C0 必须自己实现，不能直接把 recipe 当 C0。

### 1.3 每步样本与 logprob 对应

- 每次 LLM 调用 = 一个 `AgentFlowStep` = 训练 batch 里的一行（`hotpotqa_agent_flow.py:348-394`，`agent_env_loop.py:151-163`）。
- 行内：`prompt_ids` 左 pad 到 `rollout.prompt_length`，`response_ids` 右 pad 到 `rollout.response_length`（`agent_flow.py:269-291`）。观测不进 response，所以 `response_mask` 默认全 1（`:294`），不存在多轮拼接里的"工具 token mask"问题。
- `response_logprobs` 直接取 vLLM 返回的 `output.log_probs[:response_length]`，与 `response_ids` 逐 token 对齐，pad 为 0（`agent_flow.py:304-307`）。只有每一步都有 logprobs 时才进 batch 的 `rollout_log_probs`（`:692-693`）。
- **默认不回传 rollout logprob**：采样参数 `logprobs=config.calculate_log_probs`（`agent_flow.py:552`），veRL 默认 `actor_rollout_ref.rollout.calculate_log_probs=false`。做 `logprob_consistency.py` 必须设为 `True`。设了以后 trainer 会自动算 `training/rollout_probs_diff_{max,mean,std}`、`training/rollout_actor_probs_pearson_corr`（Agent-R1 `ray_trainer.py:1046-1051` 调 veRL `verl/utils/debug/metrics.py` 的 `calculate_debug_metrics`）——可以直接作为我们一致性检查的第一版对照。
- 轨迹标识：worker 侧给每条轨迹生成 `trajectory_uids`（uuid），每步记 `step_indices`（`agent_flow.py:660-661`、`:715-716`）；trainer 侧 `uid` 是 prompt 级 uuid（`ray_trainer.py:948`），经 `repeat(n)` + `sample_level_repeat(num_steps)` 扩到每一步（`:976-979`）。
- 奖励放在每步 response 最后一个有效 token（`agent_flow.py:674-678`）。`reward_score=None` 的步会调用 reward 函数（`_compute_score`，`:418-451`），否则用 flow 给的数。recipe 中间检索步给 `0.0`，最终步给 `None` → EM（`hotpotqa_agent_flow.py:352`、`:390`）。
- PPO mini-batch 按整条轨迹切（`trajectory_batching.py:19-57`，PR #98 修过），`ppo_mini_batch_size` 会乘以 `rollout.n`（`ray_trainer.py:388-389`）。

### 1.4 C1/C2/C3 挂载点建议

推荐：**在我们自己的 `src/ssrl/env_search.py` 写一个 `AgentFlowBase` 子类**（以 `HotpotQAAgentFlow` 为模板复制，不改上游），用自己的 YAML（`agent_flow_config_path`）注册，例如：

```yaml
- name: ssrl_search
  _target_: ssrl.env_search.SSRLSearchFlow
  context_policy: c2        # c0/c1/c2/c3
  max_steps: 6
  max_searches: 4
  top_k: 3
  corpus_data_dir: /data/agent-r1-data/hotpotqa_corpus
```

具体挂载：
1. `run()` 每步开头调用 `ContextPolicy.build(state) -> messages`，替代 `_build_messages` / `_make_anchor_obs`（`hotpotqa_agent_flow.py:205-231`、`:270-283`）。
2. 动作解析（`<search>/<note>/<answer>`）放在 `server_manager.generate` 之后、构造 `AgentFlowStep` 之前（对应 `:328-384`），更新外置 `State`。
3. 非终局步 `reward_score=0.0`；终局 `None` 交给 EM；超预算强制结束时最后一步给 `0.0` 并在 `extra_fields` 记 `truncated`。**非终局步千万别留 `None`**：否则每步都会跑一次 reward 函数，且 recipe 的 EM 在没有 `<answer>` 时会拿全文去比（`reward_fn.py:32-40`），GRPO 再把各步分数求和（`core_algos.py:372-390`）。
4. 每步把 `state` 快照（含 notes、doc_id、事件日志）放进 `extra_fields`，会被带进 `non_tensor_batch`（`agent_flow.py:745-760`），用于回放与诊断。

备选：写 `AgentEnv` 子类，把上下文策略放在 `step()` 返回的 `Observation(messages=...)` 里，再用 `AgentEnvLoop`。缺点：`AgentEnvLoop` 在 prompt 超长时直接 `break`（`agent_env_loop.py:120-129`），没有 recipe 那样的预算收缩；`max_steps` 来自全局 `rollout.agent.max_steps`；`AgentEnv` 注册表靠 import 副作用，Ray worker 里需要确保我们的模块已被导入。所以首选子类方案。

注意：我们的模块要能在 Ray worker 里 import（`PYTHONPATH` 包含 `src/`，worker 是否继承 driver 的环境变量：未确认，D1 验证）。

## 2. 优势如何广播到每一步

- 入口：`compute_advantage()`，`agent_r1/trainer/ppo/ray_trainer.py:220`；GRPO 分支在 `:285-297`，传 `index=uid`（同一 prompt 的 G 条轨迹为一组）和 `trajectory_uids`。
- 实现：`compute_grpo_outcome_advantage()`，`agent_r1/trainer/ppo/core_algos.py:260-345`：
  1. `_trajectory_total_scores`（`:372-390`）把每条轨迹所有步的 token 奖励求和 → 一个轨迹分；
  2. 按 `uid` 分组求均值/标准差（组内只有一条时 mean=0, std=1，`:316-325`）；
  3. `(score - mean) / (std + eps)`（`norm_adv_by_std_in_grpo=False` 时不除 std，即 Dr.GRPO），再赋给该轨迹的**每一步**并乘 `response_mask`（`:329-343`）。
- G 的定义：`actor_rollout_ref.rollout.n`（recipe 默认 8）。
- `step_advantage.py` 的校验可以直接对着上面三步写：同一 `trajectory_uids` 的所有行优势相同；同一 `uid` 组内优势和约为 0。
- loss 聚合：recipe 用 `loss_agg_mode=seq-mean-token-mean`，这里的 "seq" 是**每一步**，所以步数多的轨迹在 loss 里权重更大（未见按轨迹归一）。C0–C3 步数分布不同时，这是需要在报告里说明的混杂因素。

## 3. HotpotQA recipe

| 项 | 值 | 位置 |
|---|---|---|
| flow 配置 | `recipes/hotpotqa/base.yaml`：`max_steps: 5`、`max_parallel_calls: 4`、`force_first_search: true`（用问题原文先检索一次，不占步数）、`enable_tool_parse_feedback: true` | `base.yaml:1-14` |
| GRPO 启动脚本 | `examples/hotpotqa/run_grpo.sh` | 全文 |
| 模型 | `Qwen/Qwen3-4B-Instruct-2507`（`HOTPOTQA_MODEL_PATH`） | `run_grpo.sh:44` |
| G | `AGENT_R1_GRPO_ROLLOUT_N=8` | `:25` |
| batch | `train_batch_size = 256 / n = 32` 个 prompt；`ppo_mini_batch_size=32`（×n=256 条轨迹，一步一次更新）；`ppo_micro_batch_size_per_gpu=4`；`log_prob_micro_batch_size_per_gpu=1` | `:27-30`、`:95`、`:105-111` |
| 长度 | `max_prompt_length=10240`（每步重建 prompt）、`max_response_length=1024`（单步） | `:52-53` |
| 其他 | lr 1e-6，无 KL loss，actor/ref 参数和优化器 offload，vLLM TP=1，`gpu_memory_utilization=0.7`，`agent.num_workers=4`，`n_gpus_per_node=4`，`save_freq=100`，`test_freq=10`，`total_epochs=5`，logger 含 swanlab/mlflow，trace 后端 mlflow | `:101-139` |
| 奖励 | `custom_reward_function.path=recipes/hotpotqa/reward_fn.py`，`compute_score`：取最后一个 `<answer>...</answer>`（没有则用全文），SQuAD 式归一化（小写、去标点、去冠词、合并空白）后与 `ground_truth` 及 `extra_info.answers` 别名做**精确匹配**，返回 0/1 | `reward_fn.py:15-40`、`:94-120` |
| prompt / 工具 schema | `recipes/hotpotqa/prompts.py`（`<analysis>` + Hermes `<tool_call>` JSON + `<answer>`） | `prompts.py:8-70` |
| 预处理 | `recipes/hotpotqa/data_preprocess/process_hotpotqa.py`：HF `hotpot_qa`/`distractor` → `train.parquet`/`validation.parquet`（`data_source=hotpotqa_distractor`，prompt 只有问题），并从 train+validation 的 context 段落去重生成 `hpqa_corpus.jsonl`；`--include_cross_eval` 从 `corag/multihopqa` 生成 2Wiki/MuSiQue 验证集 | `:36-69`、`:186-199`、`:202-335` |
| 数据期望路径 | `data/corpus/hotpotqa/{train,validation}.parquet`（**脚本里写死相对仓库根目录**，`run_grpo.sh:55-56`）；检索文件 `data/corpus/hotpotqa_corpus/{hpqa_corpus.jsonl,index.bin}` | |

与我们 PLAN 的差异（实现自有 flow 时要改）：recipe 一步最多 4 个并行检索、top-k 写死为 5（`search_tool.py:279`、`:293`）、最多 5 步、单步 1024 token；PLAN 是一步一个动作、top-k=3、最多 4 次检索/6 步、单步 ≤512。

官方处理好的数据（ModelScope `Melmaphother/Agent-R1-data`）：HotpotQA train 90,447 行、validation 7,405 行，另有 2Wiki/MuSiQue 验证 parquet。

## 4. 检索器与语料

- 类型：**稠密检索，进程内加载，不是 HTTP 服务，没有端口**。`HotpotQASearchToolLegacy`（`recipes/hotpotqa/env/search_tool.py:153-331`）：FlagEmbedding `BAAI/bge-large-en-v1.5` 编码查询（带 "Represent this sentence for searching relevant passages: " 前缀）+ FAISS `Flat` 内积索引（`build_index.py:94-97`）。**不依赖 Search-R1 的 retriever / e5 / wiki-18。**
- 语料：HotpotQA distractor 设置下 train+validation 所有 context 段落去重，**509,308 条**（ModelScope README）。这是一个"封闭"语料：每道题的 gold 段落都在里面，不是完整 Wikipedia。
- 文件大小（ModelScope 文件列表）：`hpqa_corpus.jsonl` 258.1 MB、`index.bin` 1989.5 MB、`hpqa_corpus.npy` 1989.5 MB（只在建索引时用，运行时不加载，可以不下载）。
- 内存（估算，D1 实测）：每个加载者 ≈ FAISS 索引 ~1.9 GiB（509,308 × 1024 × 4 B）+ 语料 Python 字符串（jsonl 258 MB，内存里可能更大，未确认）+ bge-large 模型约 1.3 GB（fp32，3.35 亿参数）≈ **每份约 4 GB**。缓存是**进程内类变量**（`search_tool.py:167-171`），每个 Ray `AgentFlowWorker` 进程各加载一份 → 总量 ≈ 4 GB × `agent.num_workers`（recipe 是 4，默认 8）。单卡冒烟用 2 个 worker 约 8–10 GB 主机内存。
- GPU：默认 **CPU** 编码。`HOTPOTQA_EMBEDDING_PER_WORKER_GPU=1` 的设计需要框架传 `agent_flow_worker_index`，但当前 commit **框架从未传这个参数**（只在 `hotpotqa_agent_flow.py:196` 读取），而且 Ray agent worker 一般没有分配 GPU，`normalize_embedding_device` 会退回 CPU（`search_tool.py:67-74`）。所以实际是 CPU 编码。
- 吞吐隐患：`_do_search` 是同步调用（`hotpotqa_agent_flow.py:398-418`），在 async `run()` 里会阻塞该 worker 的事件循环，同一 worker 里的并发轨迹在检索时排队。D3 测吞吐时要注意。
- 启动方式：无需单独启动。设 `HOTPOTQA_CORPUS_DATA_ROOT=<绝对路径>`（相对路径会按 worker 的 cwd 解析，`search_tool.py:20-30`），首次检索时加载（日志会提示 "may take several minutes"）。
- OOD 语料：MuSiQue 有现成索引（139,416 条，`index.bin` 544.6 MB）；**2Wiki 只有 jsonl（5,902,082 条，2.9 GB），没有索引**，自建 bge-large Flat 索引约 24 GB 内存。另外 recipe 的跨集验证用的是同一个 `corpus_data_dir`，也就是拿 HotpotQA 语料去答 2Wiki/MuSiQue 问题——我们做 OOD 评测时要换成对应语料。

## 5. veRL 0.7 的 LoRA

来源：veRL v0.7.0 `docs/advance/ppo_lora.rst`，配置默认值见 `verl/trainer/config/_generated_ppo_trainer.yaml:297-301`。

FSDP + vLLM 路径（我们用的）：

```bash
actor_rollout_ref.model.lora_rank=32            # 0 = 关闭；文档建议 >=32
actor_rollout_ref.model.lora_alpha=32
actor_rollout_ref.model.target_modules=all-linear
actor_rollout_ref.rollout.load_format=safetensors   # 文档标为必需
actor_rollout_ref.actor.optim.lr=1e-5            # 文档建议比全参大一个数量级（全参 recipe 为 1e-6）
# 可选
actor_rollout_ref.model.lora_adapter_path=...    # 从已有 adapter 继续
actor_rollout_ref.model.use_shm=True
actor_rollout_ref.rollout.layered_summon=True    # 显存紧张时
```

注意事项：
- 只支持 FSDP/FSDP2 + vLLM（Megatron 用另一套 `model.lora.*` 键，与我们无关）。
- vLLM 端 `max_lora_rank` 会自动向上取到 8/16/32/64/128/256/320/512 之一（`verl/workers/rollout/vllm_rollout/utils.py:21-34`，`vllm_async_server.py:352-354`）。
- LoRA 时 ref policy = 关掉 adapter 的 actor（`fsdp_workers.py:1044-1048`，`:1010-1011`），不再单独加载 ref 模型，省显存。recipe 本来就没开 KL，影响不大。
- checkpoint 会额外存 `actor/lora_adapter/`（`fsdp_workers.py:1091-1092`）。
- **Agent-R1 用自己的 `DataParallelPPOActor` 替换了 veRL 的 actor**（`agent_r1/workers/fsdp_workers.py:24-38`），但包装的是同一个 `actor_module_fsdp`，adapter 开关在 worker 层处理，理论上兼容。仓库里没有任何 LoRA 示例或测试：**LoRA + Agent-R1 未确认，D3 前必须用 0.6B 跑通 5 步**。
- 有报告指 vLLM ≥0.12 移除了 LoRA extra vocab，导致其他框架（ms-swift PR #7245）访问 `lora_extra_vocab_size` 报错；veRL 0.7.0 的 vLLM rollout 代码里没有搜到这个属性，是否受影响未确认。

## 6. checkpoint / resume

Agent-R1 没有重写 `_save_checkpoint`/`_load_checkpoint`，用 veRL 的（`verl/trainer/ppo/ray_trainer.py:938-1062`）。`fit()` 开头调 `_load_checkpoint()`（Agent-R1 `ray_trainer.py:897`），满足 `save_freq` 条件时保存（`:1173-1179`）。

配置键（`trainer.*`，默认值见 `_generated_ppo_trainer.yaml:584-595`）：

| 键 | 默认 | 用法 |
|---|---|---|
| `trainer.save_freq` | -1 | 每 N 步存一次；>0 时最后一步也存 |
| `trainer.resume_mode` | `auto` | `auto`：找 `default_local_dir` 下最新的；`disable`；`resume_path`：配合下一项 |
| `trainer.resume_from_path` | null | 必须包含 `global_step_` |
| `trainer.default_local_dir` | `checkpoints/${project_name}/${experiment_name}` | 设成绝对路径 `runs/<run_id>/ckpt` |
| `trainer.max_actor_ckpt_to_keep` | null | 本地保留几个 |
| `trainer.del_local_ckpt_after_load` | false | |
| `trainer.total_training_steps` | null | 冒烟设 10 |
| `actor_rollout_ref.actor.checkpoint.save_contents` | `[model, optimizer, extra]` | 加 `hf_model` 会额外存完整 HF 权重 |
| `trainer.esi_redundant_time` | 0 | 火山 ESI 专用，与 GCP Spot 无关 |

目录结构：

```
<default_local_dir>/
  latest_checkpointed_iteration.txt        # 保存完成后才写入（async_save=false 时）
  global_step_<N>/
    data.pt                                 # StatefulDataLoader 状态（数据读取进度）
    actor/
      model_world_size_<W>_rank_<r>.pt
      optim_world_size_<W>_rank_<r>.pt
      extra_state_world_size_<W>_rank_<r>.pt  # lr_scheduler + RNG
      huggingface/                          # config + tokenizer（权重只有在 save_contents 含 hf_model 时才有）
      lora_adapter/                         # 仅 LoRA
```

对照我们的 checkpoint 要求（模型、optimizer、scheduler、RNG、数据进度、step）：veRL 都覆盖了（step 来自目录名）。

`--resume` 包装要点：
1. 固定 `trainer.default_local_dir`、`project_name`、`experiment_name`、`total_training_steps`，`resume_mode=auto`；启动前先从 GCS 把 `latest_checkpointed_iteration.txt` 和对应 `global_step_N/` 拉回本地。
2. 分片文件名带 world size，**恢复时 GPU 数必须一致**（单卡串行跑没问题）。
3. **veRL 的 file logger 用 `"w"` 模式打开**（`verl/utils/tracking.py:239-246`），恢复会覆盖指标文件。每次启动设不同的 `VERL_FILE_LOGGER_PATH`（例如 `runs/<run_id>/metrics.attempt<k>.jsonl`），之后再合并。
4. 存完一次就把 `global_step_N/` 同步到 `gs://ssrl-runs-gqf-2026/<run_id>/`；4B 全参的 optimizer 状态很大，同步时间和磁盘要在 D3 实测。
5. 轨迹落盘：`trainer.rollout_data_dir=<dir>` 会每步写 `<global_step>.jsonl`，按轨迹分组、内含每步的 input/output/score（`ray_trainer.py:142-197`、`:469-499`、`:1145-1147`）——这就是 Go/No-Go 要求的"轨迹可回放"的直接来源。验证集用 `trainer.validation_data_dir`。

## 7. 依赖版本与 Docker 镜像

veRL 0.7.0 `setup.py` 约束：`vllm>=0.8.5,<=0.12.0`、`tensordict>=0.8.0,<=0.10.0,!=0.9.0`、`ray[default]>=2.41.0`、`numpy<2.0.0`；sglang extra 固定 `torch==2.9.1`。v0.7.0 自己的 CI 用 `verlai/verl:vllm012.latest`（`.github/workflows/vllm.yml:77`）。

推荐镜像：**`verlai/verl:vllm012.latest`，按 digest 拉取**

| 项 | 值 | 来源 |
|---|---|---|
| digest（Docker Hub API） | `sha256:9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d` | hub.docker.com tags API，last_updated 2025-12-29，压缩约 13.5 GB |
| 基础镜像 | `nvidia/cuda:12.9.1-devel-ubuntu22.04` | `docker/Dockerfile.stable.vllm` |
| python | 3.12 | 同上 |
| torch | 2.9.0（cu129） | 同上 |
| vLLM | 0.12.0 | 同上 |
| flash-attn | 2.8.1 | 同上 |
| transformers | 构建时 `pip install --upgrade` 的最新版，**未确认**（D1 记录） | 同上 |
| ray | 随 vLLM 依赖安装，**未确认**（D1 记录） | |
| verl | 镜像里装了 0.6.0 又卸载，**镜像里没有 verl**，需要 `pip install --no-deps verl==0.7.0` | 同上 |

D1 在 VM 上 `docker pull` 后用 `docker inspect --format '{{index .RepoDigests 0}}'` 核对 digest，并把 `pip list` 中 torch/vllm/transformers/flash-attn/ray/tensordict/verl 版本写进 `env/GPU_ENV.md`。

备选（vllm012 跑不通时）：`verlai/verl:vllm011.latest`（`sha256:3ce56ff018516b28ab9c4f4fc09d3aa67589074495ace75e2674b720aa4d0e5d`）。Agent-R1 issue #83 的报告者用 verl 0.7.0 + python 3.12 + CUDA 12.8 + torch 2.8.0 + vLLM 0.11.0 + transformers 4.56.1 能跑起来（他报的 bug 已由 `make_json_safe` 修复，`ray_trainer.py:126-139`、`:907`、`:1220`）。

recipe 额外依赖：`faiss-cpu>=1.7.4`、`FlagEmbedding>=1.2.0`（`recipes/hotpotqa/requirements.txt`）。FlagEmbedding 会拉 transformers/sentence-transformers 等，**可能升级镜像里的 transformers**。安装时用 constraints 锁住现有版本（见 §10）。

## 8. 单张 A100 80GB 跑 10 步：需要改什么

不要直接改上游脚本：复制 `examples/hotpotqa/run_grpo.sh` 到我们的 `scripts/smoke_hotpotqa.sh` 再改（或用尾部 Hydra 覆盖；同一个键覆盖两次是否报错未实测，复制更稳）。需要改的点：

| 项 | recipe | 单卡冒烟建议 | 原因 |
|---|---|---|---|
| `CUDA_VISIBLE_DEVICES` / `trainer.n_gpus_per_node` | 0,1,2,3 / 4 | `0` / `1` | 单卡 |
| `HOTPOTQA_EMBEDDING_PER_WORKER_GPU` | 1 | 0（并设 `HOTPOTQA_EMBEDDING_DEVICE=cpu`） | 本来也不生效，显式写清楚 |
| `HOTPOTQA_CORPUS_DATA_ROOT` | 相对路径 | 绝对路径 | 避免 worker cwd 问题 |
| 模型 | Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-1.7B`（不行再换 0.6B） | |
| `+data.apply_chat_template_kwargs.enable_thinking=False` | 无 | 加上 | Qwen3-1.7B/0.6B 是混合思考模型，默认会输出 `<think>`，吃掉单步 1024 token；4B-Instruct-2507 没有这个问题（键来源 `agent_flow.py:162`，未实测） |
| `rollout.n` | 8 | 4 | 冒烟只验证能跑通 |
| `data.train_batch_size` | 32 | 8（即 `HOTPOTQA_GRPO_BASE_TRAIN_BATCH=32`） | |
| `actor.ppo_micro_batch_size_per_gpu` | 4 | 2 | 显存保守 |
| `rollout.log_prob_micro_batch_size_per_gpu` | 1 | 4 | 1 太慢；OOM 再调小 |
| `data.max_prompt_length` | 10240 | 4096（flow 会自动收缩 passage） | 加快冒烟 |
| `rollout.gpu_memory_utilization` | 0.7 | 0.5 | 训练和 vLLM 共卡；OOM 再降 |
| `rollout.agent.num_workers` | 4 | 2 | 每个 worker 一份 ~4 GB 检索内存 |
| `rollout.calculate_log_probs` | false | **True** | 拿 rollout logprob，用于一致性检查 |
| `trainer.total_training_steps` | — | 10 | |
| `trainer.val_before_train` / `test_freq` | True / 10 | False / -1（或 `data.val_max_samples=64`） | validation 有 7,405 条，太慢 |
| `HOTPOTQA_ENABLE_CROSS_EVAL` | auto | 0 | |
| `trainer.logger` | console, swanlab, mlflow | `["console","file"]` | 不依赖外部服务 |
| `rollout.trace.backend` | mlflow | `null` | 同上 |
| `trainer.save_freq` | 100 | 5 | 冒烟时顺便验证保存与 resume |
| `trainer.rollout_data_dir` / `default_local_dir` | — | `runs/<run_id>/rollouts`、`runs/<run_id>/ckpt` | 轨迹回放 + checkpoint |

保留 recipe 自带的 param/optimizer offload（1.7B 不需要，但不影响正确性）。以上数值都没跑过，是起点，不是结论。

## 9. 已知问题与风险

GitHub issues（截至 2026-11-03）：
- #101（open）：HotpotQA 训练时出现 `Failed to decode tool call: 'arguments'`。维护者答复：小模型输出的 tool call 格式错误是正常现象，会被捕获，不影响训练。冒烟时看到这条日志不算失败。
- #102（open）：checkpoint 位置问题，答复与 §6 一致；有用户已在 verl 0.8.0 + sglang 上跑（非 main）。
- #100（open PR）：迁移到 verl 0.8.0，会删除 legacy FSDP actor 重写、改 reward 配置命名空间。合并后与本笔记不兼容 → 固定 SHA。
- #108/#110（open）：`Observation` 未校验"只设一个字段"，同时设多个时按 token_ids > messages > text 顺序静默取一个（`agent_env_loop.py:87-98`）。
- #83（closed）：file logger 遇到 numpy 标量会崩溃，已修。

会威胁 Go/No-Go 或后续计划的风险：
1. **环境栈**：vllm012 镜像 + 手动装 `verl==0.7.0` + FlagEmbedding 依赖可能互相冲突；镜像基于 CUDA 12.9，宿主机驱动需足够新（`nvidia-smi` 先看，过旧时用 vllm011 镜像或升级驱动；具体最低驱动版本以 NVIDIA 文档为准，未确认）。缓解：按 digest 拉取 + constraints 安装 + 备选镜像。
2. **recipe ≠ 我们的实验设计**：recipe 的上下文不是 C0；top-k 写死 5、一步并行 4 个检索、同步检索阻塞事件循环；loss 按"步"聚合导致步数多的轨迹权重大。需要自写 flow（§1.4），而这部分只有 D1 跑通 recipe 后才能开始。另外 Agent-R1 技术报告已在 GSM8K 上比较过 append / sliding-window / LLM 摘要三种上下文策略（`docs/experiments.md:46-54`，结论是 sliding-window 最好、LLM 摘要在小模型上更差）——与我们的 C0/C1/C2 有重叠，报告里要讲清差异（多跳检索任务、笔记是被 RL 直接优化的动作、帕累托分析），必要时回头调整定位。
3. **检索内存与吞吐**：每个 agent worker 一份 ~4 GB（估算），CPU 编码 + 同步调用；正式训练时 `num_workers` 和 G×batch 一放大，可能成为瓶颈。2Wiki OOD 没有现成索引（590 万段，Flat 索引约 24 GB）。
4. **LoRA 路径未被 Agent-R1 测试过**：单卡期间大概率锁定 LoRA（CLAUDE.md 已定决策），必须 D3 前验证。
5. **resume 细节**：file logger 覆盖写；checkpoint 与 GPU 数绑定；Spot 抢占发生在保存中途时，`latest_checkpointed_iteration.txt` 仍指向上一个完整 checkpoint（保存完才写），这一点是安全的。

## 10. D1 执行清单（全新 GCP Deep Learning VM，1×A100 80GB）

以下命令按顺序执行，都没有实测，遇到错误时记录原文。`<...>` 由执行者填写。

```bash
# 0. 主机检查
nvidia-smi                      # 记录驱动版本、CUDA 版本
docker --version && docker info | grep -i runtime   # 确认有 nvidia runtime（DLVM 是否预装：未确认）
df -h /; free -g; nproc         # 磁盘建议 ≥200 GB（镜像约 30 GB 解压后，未确认；数据 ~2.5 GB；checkpoint 另算）

# 1. 拉镜像并记录 digest
docker pull verlai/verl@sha256:9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d
docker tag  verlai/verl@sha256:9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d ssrl/verl:vllm012

# 2. 代码与数据目录
mkdir -p ~/work ~/data ~/hf && cd ~/work
git clone https://github.com/<user>/stateful-search-rl && cd stateful-search-rl
git submodule add https://github.com/AgentR1/Agent-R1 third_party/Agent-R1   # 若尚未添加
git -C third_party/Agent-R1 checkout b124aa46534cbf2fb8bc8af11405774984c42ac7

# 3. 启动容器（参数来自 veRL docker/README.md）
docker run -d --name ssrl --runtime=nvidia --gpus all --net=host --shm-size=32g \
  -v ~/work:/work -v ~/data:/data -v ~/hf:/root/.cache/huggingface \
  ssrl/verl:vllm012 sleep infinity
docker exec -it ssrl bash

# ---- 以下在容器内 ----
# 4. 装 verl 0.7.0 与 recipe 依赖，锁住镜像原有版本
pip freeze | grep -iE '^(torch|vllm|transformers|flash.attn|ray|tensordict|numpy)==' > /tmp/pins.txt
pip install --no-deps verl==0.7.0
pip install -c /tmp/pins.txt "faiss-cpu>=1.7.4" "FlagEmbedding>=1.2.0" modelscope
python -c "import verl, vllm, torch, transformers, ray, flash_attn; print(verl.__version__, vllm.__version__, torch.__version__, transformers.__version__, ray.__version__, flash_attn.__version__)"
pip list > /work/stateful-search-rl/runs/env_pip_list.txt   # 写入 env/GPU_ENV.md

# 5. 下载 HotpotQA 数据与检索文件（只要 hotpotqa 和 hotpotqa_corpus；--include 的写法未实测，失败就用 git lfs 方式整仓下载约 19 GB）
modelscope download --dataset Melmaphother/Agent-R1-data \
  --include 'hotpotqa/*' 'hotpotqa_corpus/hpqa_corpus.jsonl' 'hotpotqa_corpus/index.bin' \
  --local_dir /data/agent-r1-data
cd /work/stateful-search-rl/third_party/Agent-R1
mkdir -p data/corpus && ln -s /data/agent-r1-data/hotpotqa data/corpus/hotpotqa   # 脚本写死这个相对路径
export HOTPOTQA_CORPUS_DATA_ROOT=/data/agent-r1-data/hotpotqa_corpus

# 6. 预下载模型
huggingface-cli download Qwen/Qwen3-1.7B     # 记录 revision hash

# 7.（可选，5 分钟内）先跑 GSM8K 单步 sanity（README Stage 1）验证训练栈；同样要改成单卡

# 8. HotpotQA 冒烟 10 步（脚本由 §8 的表复制修改，放在我们仓库 scripts/smoke_hotpotqa.sh）
export RUN_ID=d1_smoke_hotpotqa_grpo_q3-1p7b_$(date +%Y%m%d_%H%M)
export RUN_DIR=/work/stateful-search-rl/runs/$RUN_ID && mkdir -p $RUN_DIR
export VERL_FILE_LOGGER_PATH=$RUN_DIR/metrics.attempt1.jsonl
bash /work/stateful-search-rl/scripts/smoke_hotpotqa.sh \
  trainer.default_local_dir=$RUN_DIR/ckpt trainer.rollout_data_dir=$RUN_DIR/rollouts \
  trainer.experiment_name=$RUN_ID 2>&1 | tee $RUN_DIR/train.log
# 另开终端：watch -n 5 nvidia-smi；free -g（记录峰值显存和主机内存）

# 9. Go/No-Go 检查
ls $RUN_DIR/rollouts/            # 应有 1.jsonl ... 10.jsonl
python -I -c "import json,sys; e=[json.loads(l) for l in open(sys.argv[1])]; t=e[0]; print(t['num_steps'], [s['step_index'] for s in t['steps']], t['score']); print(t['steps'][0]['input'][:2000]); print(t['steps'][0]['output'][:1000])" $RUN_DIR/rollouts/10.jsonl
grep -E "rollout_probs_diff_mean|critic/score/mean|response_length/mean" $RUN_DIR/train.log | tail
cat $RUN_DIR/ckpt/latest_checkpointed_iteration.txt   # 应为 10

# 10. 顺便验证 resume：用同样参数、total_training_steps=12、新的 VERL_FILE_LOGGER_PATH 再启动，确认日志里出现 "Resuming from .../global_step_10" 且从第 11 步开始

# 11. 收尾
gsutil -m rsync -r $RUN_DIR gs://ssrl-runs-gqf-2026/$RUN_ID
# 在 env/GPU_ENV.md 记录：镜像 digest、pip 版本、驱动、$/h、每步耗时、峰值显存、主机内存
# 在 docs/PROGRESS.md 追加当天一行；然后关机
sudo shutdown -h now
```

Go 的判据（PLAN §8）：10 步无报错，且 `rollouts/*.jsonl` 里能按 `trajectory_uid` 读出每一步的完整 prompt 和 response。否则当天按 CLAUDE.md 切换到 Search-R1。
