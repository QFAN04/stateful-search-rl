# 实验卡：d1_smoke_hotpotqa（D1 Go/No-Go 冒烟）

> 开跑前填写「设定」，跑完填写「结果」。数字只能来自 runs/<run_id>/ 日志。

## 设定（开跑前冻结）
- 目的 / 假设：验证 Agent-R1（veRL 0.7.0）HotpotQA recipe 能在单张 A100 80GB 上用小模型跑通 GRPO 10 步，且轨迹可回放、checkpoint 可续跑。这是环境冒烟，不是实验结果，不报告任何准确率结论。
- 相对哪个 run 改了什么：首个 run。相对上游 `examples/hotpotqa/run_grpo.sh` 的改动见 `scripts/smoke_hotpotqa_inner.sh` 头注释和 `docs/notes/agent_r1_interfaces.md` §8（单卡、小 batch、关验证、file logger、打开 `calculate_log_probs`、轨迹落盘）。
- 代码 commit：（开跑时由 `runs/<run_id>/meta.txt` 记录）   上游 SHA：Agent-R1 `b124aa46534cbf2fb8bc8af11405774984c42ac7`，verl==0.7.0   镜像：`verlai/verl@sha256:9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d`（vllm012）
- 模型 / revision：Qwen/Qwen3-1.7B（`enable_thinking=False`；revision 见 HF cache snapshot，跑后回填）；不行再换 0.6B
- 数据 split 与 manifest hash：Agent-R1 官方处理的 HotpotQA train.parquet（ModelScope `Melmaphother/Agent-R1-data`）；冒烟不涉及 dev-select / test 划分
- 关键超参：GRPO，`rollout.n=4`，`train_batch_size=8`（32 条轨迹/步），lr 1e-6 全参，`max_prompt_length=4096`，`max_response_length=1024`，recipe 默认 `max_steps=5`、top-k=5，`gpu_memory_utilization=0.5`，`num_workers=2`，`save_freq=5`，`total_training_steps=10`
- 硬件 / 预计时长 / 预计费用：**GCP us-central1-a 按需 g2-standard-16（1×L4 24GB，16 vCPU，64GB）**，VM `ssrl-l4`。原计划的 us-central1 Spot A100 在 11-03 两次拿到后都在 2 分钟内被抢占，经用户同意 D1 改用按需 L4（D1 只验证环境栈与 recipe 能否跑通，结论与 GPU 型号无关；本 run 的吞吐 / 显存数字不用于 D3 决策）。时长与 $/h 跑后实测回填
  - 因 L4 只有 24GB：若 1.7B 在 `gpu_memory_utilization=0.5` 下 OOM，先降到 0.4，再换 Qwen3-0.6B（改动记录在结果里）
- checkpoint 选择规则：不适用（冒烟）
- 停止规则：任一步报错且 30 分钟内无法定位则停；10 步完成即停。

### Go / No-Go 判据（PLAN §8 D1）
1. 10 步训练无报错完成（`latest_checkpointed_iteration.txt` = 10）。
2. `rollouts/*.jsonl` 能按 `trajectory_uid` 读出每一步完整的 prompt 与 response（轨迹可回放）。
3. 用 `--resume` 再启动到 12 步，日志显示从 global_step_10 恢复并从第 11 步继续。
4. 记录 `training/rollout_probs_diff_*`（训推 logprob 差）作为一致性检查的第一份数据（不设阈值，仅记录）。

Go：1+2 满足（3 不满足则记为待修问题，不阻塞 D2）。No-Go：1 或 2 当天无法满足 → 按 CLAUDE.md 切换 Search-R1。

## 结果（跑完填写）
有效 run：`d1_smoke_hotpotqa_q3-0p6b_20261103_0956`（早期云归档）。代码 commit `76183656`，模型 **Qwen/Qwen3-0.6B**，额外覆盖 `ppo_micro_batch_size_per_gpu=1`（见 meta.txt）。
- **代码版本更正（codex D1 审查发现）**：meta.txt 里的 `76183656` 是 VM 上仓库的 HEAD，但实际运行的两个脚本是之后单独 scp 上去的，`76183656` 不能还原这次运行。实际运行的脚本等同于 commit `a20c197` 中的 `scripts/smoke_hotpotqa.sh`（含 `7e58ffc` 的 attempt 计数修复）和 `scripts/smoke_hotpotqa_inner.sh`（含 `+actor_rollout_ref.model.override_config.max_position_embeddings=5120`）。之后的启动器已改为在 meta.txt 中记录脚本 sha256 和未提交改动。

- 实际时长 / 费用：训练 10 步 8 分 32 秒（attempt1，09:56–10:07 UTC），续跑 2 步（attempt2，10:20 起）。`ssrl-l4` 开机 08:10:44–10:27:57 UTC（约 2 小时 17 分，其中约 1 小时因启动脚本 bug 空转，见下）。`ssrl-a100` 两次 Spot 合计开机约 3 分钟。$/h 以 GCP 账单回填。
- 主指标（来自 `metrics.attempt{1,2}.jsonl`、`replay_check.txt`）：
  - Go 判据 1：10 步完成，`latest_checkpointed_iteration.txt` = 10（global_step_5 / 10 已存）。✅
  - Go 判据 2：用加严后的 `scripts/check_replay.py`（检查文件齐全、每文件 32 条、uid 唯一、`step_index == range(num_steps)`、input 非空，有问题退出码非零）在本地对全部 12 个 rollout 文件复查：384 条轨迹 `PASS`（`runs/<run_id>/replay_check_v2.txt`）。✅ 范围限定：这是**逐步文本可审阅**——文件内容是 `skip_special_tokens` 解码后的文本，不能用于 token 级重建或 logprob 重算；我们自己的状态快照（notes、doc_id）不会自动落盘，D3 实现 C2 时需显式写入。
  - Go 判据 3：`--resume` 启动，日志 `Load from checkpoint folder: .../global_step_10`，指标只有 step 11、12，最终 checkpoint 12。✅
  - 判据 4（仅记录）：veRL 调试指标 `training/rollout_probs_diff_mean` 12 步范围 0.0062–0.0071，`training/rollout_probs_diff_max` 0.108–0.547。它是 veRL `calculate_debug_metrics` 的输出，具体公式（是否取指数、mask 口径）尚未对照 veRL 源码核实，**不等于 PLAN 要求的逐 token |logp_rollout − logp_actor| 一致性检查**，后者在 D2 由 `checks/logprob_consistency.py` 显式计算。
  - 单步耗时：第 1 步 69.9 s，第 2–12 步平均 46.3 s（L4、0.6B、32 条轨迹/步）；`perf/max_memory_allocated_gb` 峰值 17.9。
  - 行为（replay_check_v2.txt，12 步共 384 条）：375 条 1 步、8 条 2 步、1 条 3 步；每步平均得分 0.000–0.344。**按最后一步输出分类的终止原因：201 条（52%）既无 `<answer>` 也无 `<tool_call>`，176 条带 `<answer>`，7 条有未能解析的 `<tool_call>`。** recipe 的 flow 在没解析到工具调用时就结束并评分，不要求合法答案。所以"一步结束"主要是 0.6B 的**格式失败**，其次才是"自动首搜后直接作答"（`force_first_search=true` 会在第 1 步前用问题原文检索一次，不计步）。
- 异常 / 中断 / resume 记录：
  1. 首次启动（1.7B，`d1_smoke_hotpotqa_q3-1p7b_20261103_0839`）：`smoke_hotpotqa.sh` 的 attempt 计数用 `ls glob | wc -l`，无匹配时在 `pipefail + set -e` 下无声退出，训练未启动；监控用 `pgrep -f` 匹配到自身 ssh 命令，误报 ALIVE，约 1 小时后才发现。已改用 `find` 计数、监控改用 `grep '[m]ain_agent_ppo'`。
  2. vLLM KV cache 不足：veRL 0.7 `vLLMHttpServer` 用 `hf_config.max_position_embeddings`（Qwen3 为 40960）覆盖 `rollout.max_model_len`（`vllm_async_server.py:198`）。改为 `+actor_rollout_ref.model.override_config.max_position_embeddings=5120`。4B / A100 上同样需要。
  3. 1.7B 全参在 L4 上 actor 更新 OOM（rollout 已跑通）→ 按预案换 0.6B + micro batch 1。
  4. 训练结束后的清理阶段出现 `DataLoader worker ... killed by signal: Killed`（10/10 之后），dmesg 无 OOM 记录，不影响结果。
- 结论与下一步：**Go（范围限定）**。Agent-R1 + veRL 0.7 栈、recipe、逐步文本回放、本地 checkpoint 续跑在单卡上可用，保留 Agent-R1，推进 D2。尚未验证：云归档 灾后恢复（从 云归档 拉回再续跑）、模型 / optimizer / RNG 恢复的逐项正确性，这些放到 D3 的 `test_resume.py`。下一步（D2）：
  - 4B probe（C0 prompt）重点看**步数分布和终止原因**（格式失败 / 作答 / 工具调用）：若 4B 也几乎一步结束，C0–C3 的上下文策略没有作用空间，需要四组统一关闭 `force_first_search` 和 / 或在 prompt 中要求先检索（属于实验设计变更，需用户确认）。
  - 4B 在单卡上的显存：1.7B 全参在 24GB 上已 OOM；4B 必须在 A100 80GB 上跑。`max_position_embeddings` 覆盖值要按 prompt+response 预算计算（不能把 5120 原样带到 8k 上下文），并在 D2 一致性检查里覆盖接近上限长度的样本。

### codex（gpt-6-astra）D1 只读审查的处理
采纳并已修复：检查器加严并对全部 rollout 复查；"一步结束"原因改正（格式失败为主）；meta 中代码版本更正，启动器记录脚本 sha256 与未提交改动；`--resume` 找不到 checkpoint 时停止、目标已完成时直接退出；最终 云归档 同步失败时显式报错；`delete_vm.sh` 只删启动盘并先列出实例；接口笔记中过时的 C0 / 预算说明加标注；PLAN 中 B1 的 top-k 改为 5。
暂缓（D3 正式训练前处理，记入 HANDOFF）：checkpoint 完整后再发布到 云归档 的原子同步；VM 生命周期绑定（记录实际 zone、任务结束自动停机）。
未采纳（对当前单人、单卡冒烟属于过度要求）：attempt 编号加锁防并发；把本卡改为"有条件 Go"之外再追加全部原始日志的逐项核验。
