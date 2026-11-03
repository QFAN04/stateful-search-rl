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
**2026-11-03 第二次尝试（自动抢占流程，`ssrl-poller` + A100 开机任务）：完成。** `DONE 2026-11-03T14:11:50Z`；汇总 `runs/d2_probe_c0_q3-4b-2507_20261103_1100/probe_summary.json`（`scripts/analyze_probe.py`）。
- sample（`val_sample/0.jsonl`，300 题 × 8 条，temperature 1.0）：
  - pass@1 = 0.4304，pass@2 = 0.4737，pass@4 = 0.5072，pass@8 = 0.5333
  - 8 条全错 46.7%，全对 32.0%，**mixed（有对有错）21.3%**——GRPO 中有非零优势的题约五分之一
  - 步数分布（2400 条）：1 步 1779、2 步 433、3 步 72、4 步 29、5 步 87；有额外检索的轨迹 25.9%；终止原因 answer 96.0%、tool_call_unparsed 2.9%、no_answer_no_tool 1.1%
  - 注意：validation 落盘的 `input` 是 prompt+response 拼接解码，分析脚本最初按整段 input 分组，得到错误的"每题 1 条"，已改为按问题文本分组后重算
- 结论（供用户决定）：C0 起点 EM ≈ 0.43；mixed 21% 表示 RL 有可学空间但不大；约 74% 的轨迹一步作答，**上下文策略只在约 1/4 的轨迹里起作用**。若希望 C0–C3 差异可测，可考虑四组统一关闭 `force_first_search`（第一步必须自己发起检索），代价是改变 recipe 默认行为、需要重跑一次 probe 对比。
- 时长：A100 三次开机（13:12–13:16、13:24–13:38、13:48–14:12 UTC）后完成；greedy 生成不到 1 分钟，每次开机初始化（Ray / FSDP / vLLM / 4B 加载）约 4 分钟。
- greedy（`runs/d2_probe_c0_q3-4b-2507_20261103_1100/val_greedy/0.jsonl`，300 题，A100 13:24–13:38 UTC 这次开机跑完并同步）：
  - EM = 0.4367（131/300）
  - 步数分布：1 步 226、2 步 46、3 步 13、4 步 4、5 步 11（约 25% 的轨迹有额外检索，平均每条额外检索 0.45 次）
  - 终止原因：answer 289、tool_call_unparsed 9、no_answer_no_tool 2——与 D1 的 0.6B 不同，4B 几乎没有格式失败
  - 初步含义：4B 有一部分多步检索行为，上下文策略有作用空间，但 3/4 的题一步作答；是否关闭 `force_first_search` 等 sample 结果（pass@k、mixed 比例）出来后再定

**2026-11-03 第一次尝试（手动）：**
**2026-11-03 第一次尝试：未完成，无结果数字。** run_id `d2_probe_c0_q3-4b-2507_20261103_1100`（VM 上目录已建，GCS 上无产物）。
- 实际时长 / 费用：`ssrl-a100`（us-central1-c Spot）三次开机：10:59–11:20（约 21 分钟，完成全部初始化：Docker、镜像、verl 依赖、数据、划分、Qwen3-1.7B；4B 在 probe 启动时下载）、11:47–11:51（约 4 分钟）、12:15–12:20（约 6 分钟），每次都被抢占。greedy probe 都已启动，但 vLLM 服务起来后不久即被收回，`val_greedy/` 从未写出。
- 异常 / 中断 / resume 记录：按用户规则（连续两次开机都不到 15 分钟即停手）于 12:21 UTC 停止，VM 保持停机，磁盘保留（初始化标记都在，下次开机直接进入 probe）。另：A100 磁盘在 07:28 抢占时正处于 apt 安装中，留下损坏的 apt 缓存，`setup_vm.sh` 已加修复步骤。
- 结论与下一步：us-central1 Spot A100 在 11-03 无法提供 ≥15 分钟的稳定窗口。等待扩容（11-05 15:22 北京时间后可申请，建议同时申请按需 A100 80GB）；也可考虑在按需 L4 上先跑本 probe（4B 推理 + 9216 上下文，显存是否够待验证）。
