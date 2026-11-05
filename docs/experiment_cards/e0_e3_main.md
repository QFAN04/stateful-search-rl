# 实验卡：e0_e3_main（E0–E3 主实验：C0–C3 四种上下文策略的 GRPO 训练）

> 开跑前填写「设定」并冻结；四组共用本卡，**组间只有 context policy 不同**。数字只能来自 runs/<run_id>/ 日志。

## 设定（E0 开跑前冻结，2026-11-05）
- 问题：在多跳搜索 Agent 的 GRPO 训练中，上下文策略（C0 append / C1 window / C2 模型自写笔记 / C3 规则压缩）是否影响 dev EM 与每步 prompt 长度。
- run_id：`e{0,1,2,3}_c{0,1,2,3}_q3-4b-2507_lora_pro6000_<首次开跑日期>`；驱动脚本 `scripts/autodl/e_run.sh <policy...>`
- 代码：D3 验收通过的版本（`docs/experiment_cards/d3_policies_tput.md`）；flow `configs/flows/ssrl_c{0..3}.yaml`；上游 Agent-R1 `b124aa4`；私有镜像 `image-f5cf94c23b`
- 硬件：AutoDL 1×RTX PRO 6000 96GB。四组同卡型、同镜像、同卡数（单卡）。
- 模型：Qwen3-4B-Instruct-2507；LoRA rank 32 / alpha 32 / all-linear，lr 1e-5（锁定，与 D3 相同）
- 训练：GRPO，**每步 32 题 × G=8 = 256 条轨迹，150 步**（用户 2026-11-05 选定；PLAN §7 起始超参是每步 64–128 题，按 D3 吞吐和预算下调）。一步一次更新（mini batch = 32 题）。max_prompt 8192、单步 response ≤1024，micro batch 4，不 offload，vLLM 显存 0.5，8 个 agent worker。
- 其余沿用 recipe 默认，四组一致：不加 KL（`use_kl_loss=False`、`use_kl_in_reward=False`；PLAN §7 写的 KL 0.001 以 CLAUDE.md「以 recipe 默认为准」为准不加）、`loss_agg_mode=seq-mean-token-mean`、`norm_adv_by_std_in_grpo=True`、奖励只用 EM、5 步 / ≤4 并行检索 / top-k=5 / 自动首搜。训练数据为 recipe 的 HotpotQA train，seed 和数据顺序四组相同。
- 策略参数：C1 `window_k=1`；C2 每步最多 3 条 note、单条 ≤300 字符；C3 每步最多 6 句、重叠 ≥2（均为 D3 定下的值）。

### checkpoint 与 dev 评测（PLAN §5，事先写定）
- 开训前（step 0）和每 20 步（20, 40, …, 140）及最后一步（150）在 dev_select 500 题上 greedy 评测（veRL 训练内 validation，`val_kwargs` greedy、n=1），生成结果存 `runs/<id>/val/`。
- 每 20 步存 checkpoint，完整 checkpoint 只保留最近 2 个（用于续训）；每次存档后把 LoRA adapter 复制到 `runs/<id>/adapters/step_<N>/`，全部保留（用于最终评测）。
- **选择规则**：取 dev EM 最高的 step（不含 step 0）；并列取更早的一步。封存 test 只在四组都训完、配置冻结后，用各组选中的 adapter 跑一次（D9）。

### 判断标准（事先写定）
- 每组训练有效性（PLAN D4–D5 验收）：前 40 步内 `critic/score/mean`（训练 EM）有上升趋势；全程无 NaN / inf；抽查 step 20、step 100 各 5 条轨迹格式正常。不满足时先停下与用户讨论，**不在组内改超参**。
- 主比较：四组选中 checkpoint 的 dev EM 和每步平均 prompt token 数；按检索轮数（1 轮 / ≥2 轮）分层报告（PLAN 2026-11-04 补充）。dev 只用来选 checkpoint 和看趋势，结论以 D9 封存 test 为准。
- C2 专门记录：有 note 调用的轨迹比例随训练的变化。若训练到 60 步时 C2 note 比例降到 < 1%，按 PLAN 止损规则讨论（结论保留为诊断，不调参追分）。
- 步数分布（`num_steps/mean`）和 response 长度作为混杂因素报告。

### 运行规范
- 中断续训：同样的命令重跑，`RESUME_MODE=auto` 从最近 checkpoint 继续（D3 已验证 LoRA 续训）；跨日期用 `TAG` 保持 run_id。
- 脚本结束 30 分钟后自动关机，另有 `MAX_HOURS` 硬上限（默认每组 12 小时 + 2）。
- 结果（不含完整 checkpoint，含 adapters）`autodl pull` 回本地，再上传 `gs://ssrl-runs-gqf-2026/runs/`。
- 磁盘：4B 完整 checkpoint 大小未实测（按 0.6B 推算 8–16GB / 个），保留 2 个加数据和模型约 40–50GB，**50GB 数据盘不够，开跑前扩到 100GB**。每组结束且 adapter 齐全（20…140、150）后脚本删除该组完整 checkpoint，多组连跑不累积；adapter 不齐时该组记为失败并保留 checkpoint。第一次存档（step 20）后核对 checkpoint 和 adapter 大小（`runs/<id>/adapters.log`），记到下方结果。
- 预计时长 / 费用（按 D3 吞吐外推，**未实测**）：训练每步约 3–3.5 分钟 × 150 步约 8–9 小时，加 9 次 dev 评测（每次估计 5–10 分钟），每组约 9–10 小时，按 ¥6.98/时约 ¥65–70；四组约 ¥260–280。

## 结果（每组跑完填写）

| 组 | run_id | 实际时长 / 费用 | 每步耗时 | 峰值显存 | checkpoint / adapter 大小 | 选中 step | 选中 dev EM | 异常 |
|---|---|---|---|---|---|---|---|---|
| E0 C0 | | | | | | | | |
| E1 C1 | | | | | | | | |
| E2 C2 | | | | | | | | |
| E3 C3 | | | | | | | | |
