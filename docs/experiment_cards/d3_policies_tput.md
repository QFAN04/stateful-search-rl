# 实验卡：d3_policies_tput（C0–C3 端到端验证、迁移验收、4B 吞吐）

> D3 的 GPU 部分。开跑前填写「设定」，跑完填写「结果」。数字只能来自 runs/<run_id>/ 日志。
> 不是正式实验：不训练到收敛，只验证代码、环境和吞吐，为 E0–E3 定配置。

## 设定（开跑前冻结）
- 代码：`src/ssrl/context_policies.py`（★ C0–C3 渲染）、`src/ssrl/env_search.py::SSRLSearchFlow`、`configs/flows/ssrl_c{0,1,2,3}.yaml`；本地单测 46 项通过（`tests/test_context_policies.py` 含 C0 与 recipe 逐字一致的对拍；`tests/test_step_advantage.py`），另有 1 项 gpu 测试（与上游 `compute_grpo_outcome_advantage` 对拍）在实例上跑。
- 上游：Agent-R1 `b124aa4`；环境：AutoDL 私有镜像 `image-f5cf94c23b`（版本见 `env/GPU_ENV.md`）
- 硬件：AutoDL 1×RTX PRO 6000 96GB（Blackwell）。四组正式实验将用同一卡型。
- 驱动脚本：`scripts/autodl/d3_run.sh all`，各阶段 run_id 后缀 `pro6000_<日期>`
- 策略参数（四组统一，除 policy 本身）：recipe 默认（5 步、≤4 并行检索、top-k=5、自动首搜、单步 ≤1024 token）；C1 `window_k=1`；C2 每步最多 3 条 note、单条 ≤300 字符，只能引用本步展示的文档编号；C3 每步最多 6 句、与"问题 + 历史 query"内容词重叠 ≥2
- LoRA：rank 32 / alpha 32 / all-linear，lr 1e-5（与 `d3_autodl_envtest` 相同）

| 阶段 | 内容 | 通过标准（事先写定） |
|---|---|---|
| prep | 下载 4B；全部单测（含 gpu 对拍） | 全绿 |
| smoke | 0.6B LoRA，C0–C3 各 3 步（8 题 × 4 条） | 四组都跑完，`check_replay.py --expect-steps 3 --expect-trajs 32` 通过 |
| resume | 0.6B LoRA C2：10 步 → 续到 12 步 | `check_resume.py` 通过：步号连续（1–10、11–12），日志有 global_step_10 加载记录，LoRA B 矩阵变化比 < 0.5 |
| probe | 4B greedy，probe 300 题，C0–C3 各一次 | **迁移验收**：C0 EM 与 D2 的 0.4367 相差 ≤ 2pp（我们的 C0 与 recipe prompt 逐字一致，greedy 下应几乎相同）。C1–C3 只记录，不设门槛 |
| logprob | 4B，dev_select 前 64 题 | \|Δlogp\| 均值与 D2（0.0101）同一量级（≤ 0.03） |
| tput | 4B LoRA，C0 与 C2 各 3 步，64 题 × G=8，max_prompt 8192，不 offload，micro batch 4 | 只记录：每步耗时（gen / update 分开）、峰值显存、平均步数 |

- probe 额外记录（用于决定 E0–E3 前是否还要改动，**改动只能在 E0 开跑前、四组同时生效**）：
  - 终止原因中非 answer 的比例：任何一组 > 20%（即格式合法率 < 80%，PLAN D2 的标准），先与用户讨论是否加轻量 SFT（四组同源同规模）。
  - C2 有 note 调用的轨迹比例 `frac_traj_with_note`：greedy 下为 0 也不直接否定 C2（RL 采样 temperature 1.0 可能会探索到），但要在 tput 的 C2 采样轨迹里再看一次；采样下仍接近 0 时与用户讨论（选项：改 C2 指令措辞，或按 PLAN 止损规则把 C2 降为扩展）。
  - 每步 prompt 字符数 `prompt_chars_per_step_mean`：C1 的 `window_k` 按"平均 prompt 长度最接近 C2"选（PLAN §3.2），当前 k=1；若 k=2 明显更接近，再跑一次 C1 probe 确认后改。
- E0–E3 的 checkpoint 选择规则（PLAN §5，事先写定）：每 20 步在 dev_select 500 题上 greedy 快评，取 EM 最高的 checkpoint；并列取更早的一步。封存 test 只在四组配置冻结后跑一次。
- 停止规则：prep 失败立刻停；其他阶段失败不影响后续阶段（脚本继续跑），汇总后再决定。
- 预计时长 / 费用：环境恢复（数据盘下载数据和模型）约 30 分钟，smoke + resume 约 25 分钟，probe 4 × 约 10 分钟，logprob 约 5 分钟，tput 约 30–60 分钟；合计约 2.5–3.5 小时，按 ¥6.98/时约 ¥18–25。实例设自动关机。

## 结果（跑完填写）
- 实际时长 / 费用：
- 各阶段结果（附 run_id）：
- 异常 / 中断：
- 结论与下一步（E0–E3 配置：每步题数、总步数、卡数）：
