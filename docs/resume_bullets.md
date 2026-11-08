# 简历 bullet（真实数字，2026-11-08）

> 每个数字的出处在括号里；面试时能讲清楚的版本见 `docs/REPORT.md`。按 PLAN §1.3 的原则：说「基于 Agent-R1 实现了…」，不说「从零搭了多轮 RL 框架」。

## 中文

**StatefulSearch-RL：多跳搜索 Agent 上下文管理的强化学习研究**（Qwen3-4B，Agent-R1 / veRL，GRPO）

- 基于 Agent-R1（veRL）step-level 框架，为多跳搜索 Agent 实现可插拔上下文管理层（全量累积 / 滑动窗口 / 模型自写证据笔记 / 规则压缩），在 Qwen3-4B + LoRA 上完成四组控制变量 GRPO 对照（各 150 步，约 70 卡时）；规则压缩与全量累积 EM 持平（HotpotQA 封存测试 0.550 vs 0.542，2Wiki 0.664 vs 0.656），每步上下文缩短 7.5–20%，在 2Wiki 上每题总 token 少 11%。
- 用配对 McNemar 检验、按检索轮数分层和失败模式分析定位策略差距来源：差距几乎全部来自跨轮题——滑窗与笔记策略在 2Wiki 上低 11–17pp（p < 1e-7），原因是丢失旧证据后重复检索、步数耗尽；放宽步数上限（5 → 10）不能弥补。
- 验证「RL 能否学会自己记笔记」：只用结果奖励时，RL 没有学会——有笔记的轨迹比例从 10.6% 降至 4.4%；用双 LLM judge（DeepSeek / Gemini，忠实度判定 kappa 0.92）评估笔记质量，RL 后更贴原文（0.72 → 0.81）但引用支撑文档的比例下降（0.86 → 0.75）；据此设计了笔记过程奖励的后续实验。
- 工程与正确性：训推 logprob 一致性检查（|Δlogp| 均值 0.010、p99 0.215）、逐步优势广播对拍、严格轨迹回放与续训一致性测试；定位并修复 Agent-R1 多卡路径的 3 个问题（填充 micro-batch 崩溃、跨卡指标归约、rollout 记录混入填充行），已向上游提交 PR；实测单组 2 卡仅提速 1.19 倍（rollout 受长尾轨迹限制，Amdahl 上限约 1.4 倍），改为单卡多组并行，GPU 成本约降 40%（按实测吞吐）。
- （面试时主动说明：单 seed；「持平」指配对检验不显著；token 节省是每步口径，每题总量只在 2Wiki 上更省，因为规则压缩会多搜几轮。）

## English

**StatefulSearch-RL: RL for context management in multi-hop search agents** (Qwen3-4B, Agent-R1 / veRL, GRPO)

- Built a pluggable context-management layer (full append / sliding window / model-written evidence notes / rule-based compression) on Agent-R1's step-level framework and ran a four-arm controlled GRPO study on Qwen3-4B + LoRA (150 steps each, ~70 GPU-hours); rule-based compression matched full append in EM (sealed HotpotQA test 0.550 vs 0.542; 2WikiMultihopQA 0.664 vs 0.656) while shortening per-step context by 7.5–20%, and cut total prompt tokens per question by 11% on 2Wiki.
- Traced where the policies differ using paired McNemar tests, round-stratified analysis and failure-mode breakdowns: the gap comes almost entirely from multi-round questions; the window and notes arms lose 11–17 points on 2Wiki (p < 1e-7) because dropped evidence leads to repeated searches and exhausted step budgets, which a larger step budget (5 → 10) does not fix.
- Showed that outcome-only RL does not teach the model to curate evidence: note-writing trajectories fell from 10.6% to 4.4%; with two LLM judges (DeepSeek / Gemini, kappa 0.92 on faithfulness), notes became more faithful (0.72 → 0.81) but cited gold supporting-fact documents less often (0.86 → 0.75), motivating a follow-up process-reward experiment.
- Correctness and engineering: train–inference logprob consistency check (mean |Δlogp| 0.010, p99 0.215), step-level advantage tests, strict trajectory replay and resume checks; found and fixed three multi-GPU bugs in Agent-R1 (upstream PRs submitted); measured only 1.19× speedup from 2 GPUs per run (rollout bound by long-tail trajectories), switched to one GPU per run in parallel, cutting GPU cost by ~40% (from measured throughput).
