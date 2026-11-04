# 实验卡：d2_env_probe（环境配置预实验：多轮检索占比）

> 选环境配置用的预实验，只在 dev_select 的 probe 300 题上跑，不碰封存 test。判断标准在开跑前写定。

## 设定（开跑前冻结）
- 目的：上下文策略（C0–C3）只在一条轨迹有 ≥2 轮检索时才会让各组 prompt 不同。recipe 默认配置下（自动首搜开、top-k=5），Qwen3-4B 只有约 25% 的轨迹是多轮（D2 C0 probe：greedy 24.7%、sample 25.9%），主实验的组间差异可能被稀释。本实验测两种改动能否提高多轮占比，再决定 E0–E3 的统一环境配置。
- 对照：`d2_probe_c0_q3-4b-2507_20261103_1100`（recipe 默认：`force_first_search=true`、top-k=5）
- 变体（每个只改一个变量）：
  - **FFS-off**：`configs/flows/hotpotqa_ffs_off.yaml`，`force_first_search=false`，其余同 recipe
  - **k3**：`configs/flows/hotpotqa_k3.yaml`，`ssrl.env_search.SSRLHotpotQAFlow`，检索 top-k 5 → 3，其余同 recipe
- 模型 / 数据 / 解码：与对照完全相同（Qwen3-4B-Instruct-2507；probe 300 题；greedy + sample 8 条 / 题，temperature 1.0；max_prompt 8192、max_response 1024）
- 硬件：`ssrl-a100`（Spot），经轮询流程自动运行；训练镜像同 `env/GPU_ENV.md`
- 指标（`scripts/analyze_probe.py`，FFS-off 需加 `--ffs-off`）：多轮检索占比 `frac_multi_round`（检索轮数 = 自动首搜 + 模型发起检索的步数）、greedy EM、sample pass@1/8 与 mixed 比例、终止原因、平均步数

### 判断标准（开跑前写定）
1. 采用某个变体的条件：sample 的多轮占比比对照至少高 **10 个百分点**，且 greedy EM 比对照下降 **不超过 3 个百分点**（对照 0.4367），且格式失败（非 answer 终止）不超过 10%。
2. 两个变体都满足时，选多轮占比更高的；都不满足时保持 recipe 默认，主实验分析按检索轮数分层，并在报告中说明上下文策略的作用范围。
3. 不叠加两个改动（叠加需要再跑一组，且变量不再单一）。
4. 结果交用户确认后，写入 CLAUDE.md「已定决策」，E0–E3 四组统一采用。

## 结果（跑完填写）
任务 `d2_envprobe_20261104`（`DONE`，轮询日志 19:45 UTC 记录完成）；变体 run：`d2_envprobe_20261104_ffsoff`、`d2_envprobe_20261104_k3`（各自 `probe_summary.json`）。FFS-off 期间 A100 多次被抢占后由轮询流程续跑（meta.txt 记录了 4 次以上启动）。

| 配置 | greedy EM | 多轮占比 greedy / sample | sample pass@1 / pass@8 | mixed | sample 非 answer 终止 |
|---|---|---|---|---|---|
| 对照：FFS on，k=5 | 0.4367 | 0.247 / 0.259 | 0.4304 / 0.5333 | 0.2133 | 0.040 |
| FFS-off | 0.4033 | 0.277 / 0.283 | 0.3904 / 0.5400 | 0.3433 | 0.086 |
| k3 | 0.4567 | 0.280 / 0.285 | 0.4275 / 0.5367 | 0.2234 | 0.068 |

- 检索轮数分布（sample，2400 条）：对照 1 轮 1779 / ≥2 轮 621；FFS-off 0 轮 131 / 1 轮 1589 / ≥2 轮 680；k3 1 轮 1715 / ≥2 轮 685。
- **按事先写定的标准：两个变体都不满足**（多轮占比只比对照高 2.4 / 2.6 个百分点，远低于 10 个百分点的门槛；FFS-off 的 greedy EM 还下降了 3.3 个百分点）。**结论：保持 recipe 默认配置**；主实验分析按检索轮数分层，并在报告中说明上下文策略约作用于 1/4 的轨迹。
- 附带观察（不在判断标准内，仅记录）：FFS-off 的 mixed 比例从 21.3% 升到 34.3%（第一步要自己决定查什么，同题 8 条结果分化更大），GRPO 可学空间更大，但 pass@1 降到 0.39，且约 5% 的轨迹完全不检索（0 轮）。若之后需要更多有效梯度，可作为备选，须单独评估。
- 2026-11-04 用户确认保持 recipe 默认，已写入 CLAUDE.md「已定决策」和 PLAN §3.1。
