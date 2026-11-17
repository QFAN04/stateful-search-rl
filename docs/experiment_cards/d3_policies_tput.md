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
- 实际时长 / 费用：实例约 00:40–03:46（北京时间，2026-11-05），约 3.1 小时，按 ¥6.98/时约 ¥22。其中数据和模型下载约 1 小时（ModelScope 1–3MB/s）；D3 各阶段 02:03–03:43 共 1 小时 40 分钟。所有 run 目录（不含 checkpoint）已回传本地和 早期云归档。
- 硬件：RTX PRO 6000 Blackwell Server Edition 96GB，驱动 580.95.05，22 核，1TB 内存
- 各阶段结果（`scripts/autodl/d3_run.sh` 退出码 0，所有阶段 OK）：

| 阶段 | run_id | 结果 | 判定 |
|---|---|---|---|
| prep | — | 单测全部通过（含 gpu 对拍 `test_matches_agent_r1_upstream`） | 通过 |
| smoke | `d3_smoke_c{0,1,2,3}_q3-0p6b_lora_pro6000_20261105` | 四组各 3 步完成，`check_replay.py` 均 PASS（96 条轨迹 / 组，bad=0） | 通过 |
| resume | `d3_resume_c2_q3-0p6b_lora_pro6000_20261105` | 步号 1–10 → 11–12；日志有从 global_step_10 加载的记录；lora_B 196 个张量，\|B_12−B_10\|/\|B_10\| = 0.228 | 通过 |
| probe | `d3_probe_c{0,1,2,3}_q3-4b-2507_pro6000_20261105` | C0 greedy EM **0.4333**（D2 recipe 0.4367，差 −0.34pp） | **迁移验收通过** |
| logprob | `d3_logprob_q3-4b-2507_pro6000_20261105` | \|Δlogp\| 均值 0.0096、p99 0.216（D2：0.0101、0.215），964 token / 64 序列 | 通过 |
| tput | `d3_tput_{c0,c2}_q3-4b-2507_lora_pro6000_20261105` | 见下表 | 只记录 |

- 4B greedy probe 四组对比（probe 300 题，均未训练）：

| 策略 | EM | 非 answer 终止 | 多轮检索占比 | 第 2 步起平均 prompt 字符数 | 有 note 的轨迹 |
|---|---|---|---|---|---|
| C0 | 0.4333 | 5.0% | 26.0% | 7148–7806 | 0 |
| C1（k=1） | 0.4200 | 11.3% | 26.7% | 5109–5353 | 0 |
| C2 | 0.4467 | 6.3% | 22.3% | 5972–6405 | 9.7% |
| C3 | 0.4567 | 3.0% | 23.3% | 5868–6138 | 0 |

  - 格式合法率：四组都 ≥ 80%（最低 C1 88.7%），**不需要 SFT**。
  - C2 笔记：greedy 下 9.7% 的轨迹调用了 note；tput 的 C2 采样轨迹（temperature 1.0，1536 条）中 11.0% 有 note。不是 0，**C2 可以进入 RL**，不改指令。
  - C1 的 k：C2 第 1 步比 C0 多约 790 字符的固定开销（note 工具说明和定义：5814 vs 5021）。扣掉后，C2 第 2 步起约 5200–5600 字符，与 C1（k=1）的 5100–5350 最接近；k=2 会接近 C0。**保持 k=1**。
  - C1 有 32 条轨迹用满 5 步（C0 14 条）：窗口外的检索记录看不到，模型重复检索。这是 C1 的预期代价，记录下来供报告分析。
- 吞吐（4B LoRA rank 32，64 题 × G=8 = 512 条轨迹 / 步，max_prompt 8192，不 offload，micro batch 4，8 个 agent worker）：

| 策略 | 步 | 每步耗时 | 生成 gen | old_log_prob | update_actor | 平均步数 |
|---|---|---|---|---|---|---|
| C0 | 1 / 2 / 3 | 427 / 350 / 394 s | 161 / 132 / 144 s | 58 / 46 / 54 s | 208 / 171 / 195 s | 1.4 / 1.2 / 1.3 |
| C2 | 1 / 2 / 3 | 481 / 386 / 407 s | 177 / 140 / 141 s | 66 / 53 / 58 s | 236 / 191 / 207 s | 1.6 / 1.3 / 1.4 |

  - 峰值显存：PyTorch 分配 73.0GB；nvidia-smi 观测 C0 65.8GB、C2 66.4GB（5 秒采样），96GB 卡有余量。主机内存约 300GB（8 个 worker 各加载一份检索索引）。
  - 每步约 6–7 分钟，其中约一半在 update_actor。
- 磁盘：0.6B LoRA 的 checkpoint 每个 3.2GB（其中 `model_world_size_1_rank_0.pt` 2.9GB 存的是完整模型，LoRA adapter 只有 78MB）；`max_actor_ckpt_to_keep=2` 没有清掉第一次运行留下的 global_step_5。推算 4B 每个 checkpoint 约 8–16GB（未实测），E0–E3 需要控制保留个数或扩容数据盘。
- 异常 / 中断：
  - 私有镜像里残留基础镜像的旧文件（torch 2.8 的 `flex_attention.py` 等 321 个），导致 `import vllm` 报 "duplicate template name"。原因是保存镜像时没有保留 pip 的删除操作。已写 `scripts/autodl/clean_stale_site_packages.py`，`d3_run.sh` 的 prep 阶段会先执行它。
  - bge 下载同时拉了 safetensors / bin / onnx 三份，中途改为只下 safetensors（`setup_env.sh` 已修）。
  - 子模块从 Windows 打包后是 CRLF 换行，在实例上 `git checkout -- .` 恢复了 LF，代码内容不变。
- 结论与下一步：
  - D3 验收全部通过：C0–C3 端到端可训练、LoRA 续训可靠、迁移验收通过（D2 的基线和 probe 结论沿用）、单测全绿。
  - E0–E3 配置待用户定（按实测外推，**未实测**）：64 题 × 8 条时每步约 6.5 分钟，单卡跑 150 步约 16 小时 / 组，四组约 65 小时，约 ¥455；32 题 × 8 条时每步约减半，150 步约 8–9 小时 / 组，四组约 ¥250。还要加上每 20 步一次 dev_select 500 题快评的时间。
  - 可能的提速点（未测）：update_actor 占一半时间，可试 micro batch 8 或动态 batch。只能在 E0 开跑前定一次，四组统一。
