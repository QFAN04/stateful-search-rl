# 实验卡：d2_baselines_b0b1（只推理基线，dev_select）

> 数字只能来自 runs/<run_id>/ 日志。封存 test 到 D9 与 E0–E3 一起只跑一次。

## 设定
- 目的：给 E0–E3 一个"不训练"的参照：B0 不检索直接作答；B1 用问题原文检索一次（与 recipe 自动首搜同一检索器、top-5）后一次作答。
- 代码：`scripts/baselines_b0b1.py`（commit 见 git 历史，2026-11-03；Colab 上的副本加了 `VLLM_WORKER_MULTIPROC_METHOD=spawn`，与提交版一致）
- 模型：Qwen/Qwen3-4B-Instruct-2507，贪心解码，max_tokens 1024，max_model_len 9216
- 数据：`data/manifests/hotpotqa_dev_select.json`（500 题；输出题目 ID 与 manifest 顺序逐一核对一致）
- 评分：`recipes/hotpotqa/reward_fn.compute_score`（与训练相同的 EM：取最后一个 `<answer>`，SQuAD 式归一化 + 别名）
- 硬件 / 环境：Colab A100-SXM4-80GB（高内存），vLLM 0.33.0、torch 2.14.0、transformers 5.20.2、FlagEmbedding 1.4.2（`pip_versions.txt`）。**与训练镜像（vLLM 0.12 / torch 2.9 / transformers 4.57）不同**；只推理的 EM 受影响应很小，但 D9 封存 test 上的 B0/B1 建议在训练镜像里重跑以统一环境。

## 结果
run_id `d2_b0b1_dev_q3-4b-2507_20261103`（早期云归档，`summary.json`）

| 基线 | EM | 带 `<answer>` 比例 | 生成耗时 |
|---|---|---|---|
| B0 不检索 | 0.128 | 0.982 | 3.3 s |
| B1 检索一次 | 0.312 | 0.986 | 12.1 s（检索 81.6 s，CPU 编码） |

- 参照：同一模型在 C0（recipe agent flow，最多 5 步）的 probe（300 题，dev_select 前 300）greedy EM 0.4367。两者题集不同（500 vs 300），不能直接相减；D9 在同一封存 test 上统一比较。
- 冒烟 run `d2_b0b1_smoke20`（20 题）只用于检查流程，不报告。
- 时长 / 费用：Colab 会话约 1 小时（多数时间在装依赖和从 ModelScope 下载 2GB 索引），约 8 CU。
