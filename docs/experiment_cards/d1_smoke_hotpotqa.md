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
- 实际时长 / 费用：
- 主指标（附日志路径）：
- 异常 / 中断 / resume 记录：
- 结论与下一步：
