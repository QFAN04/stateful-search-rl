# 实验卡：d2_logprob_consistency（训推 logprob 一致性，训练镜像）

## 设定
- 目的：PLAN §6「训推一致性」第一份数据。Agent-R1 每步样本直接使用 vLLM 返回的 token ids（不重新分词），剩余不一致来自推理引擎与训练前向的数值差异；本检查测同一组 token 上 vLLM 采样 logprob 与 HF bf16 前向重算 logprob 的差。
- 代码：`src/ssrl/checks/logprob_consistency.py`（统计函数有单测 `tests/test_logprob.py`）
- 环境：A100 80GB（`ssrl-a100`，Spot，经轮询流程自动运行），**训练镜像** `verlai/verl@sha256:9576682f…`（vLLM 0.12.0 / torch 2.9.0 / transformers 4.57.3）；vLLM 与 HF 均 bf16
- 数据：dev_select 前 64 题，B1 式 prompt（问题 + recipe 检索器 top-5 段落），temperature 1.0、top_p 1.0、max_tokens 512

## 结果
run_id `d2_logprob_q3-4b-2507_20261103`（早期云归档中的 `logprob_consistency.json`；A100 15:42–15:45 UTC，`DONE 2026-11-03T15:45:36Z`）

| 指标 | 值 |
|---|---|
| 有效 token 数 / 序列数 | 969 / 64 |
| \|Δlogp\| 均值 | 0.0101 |
| \|Δlogp\| 中位数 | 1.2e-7 |
| \|Δlogp\| p99 | 0.215 |
| \|Δlogp\| 最大值 | 2.20 |
| Δlogp 有符号均值（rollout − HF） | 0.0016 |
| \|Δp\| 均值 | 0.0037 |
| 按序列平均 \|Δlogp\| 的最大值 | 0.244 |

- 解读：一半以上 token 两边完全一致；差异集中在少数 token（p99 0.21、最大 2.2），没有系统性偏差（有符号均值接近 0）。量级与 bf16 推理/训练引擎不一致的常见水平相符，**未发现需要处理的问题**；但目前不设通过阈值，只作为基线记录。
- 局限：B1 prompt 要求短答案，平均每条只有约 15 个 token，样本偏小，且不是 agent flow 的真实输出分布（`<analysis>` + 工具调用，单步约 100–1000 token）。D3 训练开跑时用 veRL 自带的 `training/rollout_probs_diff_*` 和更长的真实 rollout 再看一次。
- 遗留：A100 上 `job.log` 未能复制进 run 目录（目录由容器以 root 创建）；不影响结果。
