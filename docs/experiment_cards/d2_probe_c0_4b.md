# 实验卡：d2_probe_c0_4b（D2 Pre-GRPO probe，C0）

> 开跑前填写「设定」，跑完填写「结果」。数字只能来自 runs/<run_id>/ 日志。

## 设定（开跑前冻结）
- 目的 / 假设：训练前测 Qwen3-4B-Instruct-2507 在 C0（recipe 默认 flow）下的起点：pass@1（greedy）、pass@k（k=1/2/4/8，采样）、mixed 组比例（8 条中既有对又有错的题占比 = GRPO 可学空间），以及**步数分布和终止原因**——D1 中 0.6B 有 52% 的轨迹以"既无答案也无工具调用"结束，需要确认 4B 是否有多步检索行为，决定是否四组统一关闭 `force_first_search`。不做任何训练。
- 相对哪个 run 改了什么：相对 D1 冒烟：模型 0.6B → 4B-Instruct-2507；只做 validation（`trainer.val_only=True`）；数据为 probe 300 题。
- 代码 commit / 上游 SHA / 镜像：以 `runs/<run_id>/meta.txt` 为准（含脚本 sha256）；Agent-R1 `b124aa4`；镜像同 `env/GPU_ENV.md`
- 模型 / revision：Qwen/Qwen3-4B-Instruct-2507（revision 跑后从 HF cache 回填）
- 数据 split 与 manifest hash：`data/manifests/hotpotqa_probe.json`（300 题，dev_select 前 300，seed 20261103，ids_sha256 `cfed0edb…d2d9`）；sample 模式用每题复制 8 行的 `probe_x8.parquet`
- 关键超参：recipe 默认 flow（5 步、一步 ≤4 个并行检索、top-k=5、`force_first_search=true`）；`max_prompt_length=8192`、`max_response_length=1024`、`max_position_embeddings` 覆盖为 9216；greedy：`do_sample=False`；sample：`temperature=1.0, top_p=1.0`（与训练采样一致），每题 8 条
- 硬件 / 预计时长 / 预计费用：us-central1-c Spot A100 80GB（`ssrl-a100`）；用户规定：今天被抢占两次就停，等扩容
- checkpoint 选择规则：不适用
- 停止规则：两种模式跑完即停；Spot 被抢占两次即停

### 要报告的量（全部由 `scripts/analyze_probe.py` 从 `val_greedy/`、`val_sample/` 计算）
- greedy EM（= pass@1）；sample 的 pass@1/2/4/8（无偏估计）与 mean@8
- mixed 组比例（8 条中 1–7 条正确的题）、all-correct / all-wrong 比例
- 步数分布；终止原因（answer / tool_call_unparsed / no_answer_no_tool）；每条轨迹检索次数

## 结果（跑完填写）
- 实际时长 / 费用：
- 主指标（附日志路径）：
- 异常 / 中断 / resume 记录：
- 结论与下一步：
