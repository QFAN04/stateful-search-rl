# 实验卡：d10_rq2_note_quality（RQ2：RL 训练前后 C2 笔记质量）

> 开跑前填写「设定」并冻结；只读已有轨迹 + 外部 judge，不训练、不用 GPU。数字只能来自 `runs/d10_rq2_note_quality_<日期>/`。

## 设定（2026-11-08，开跑前冻结）
- 问题（PLAN RQ2）：RL 训练前后，C2 写下的证据笔记质量是否提升？（D9 已知：笔记**数量**从约 12% 的轨迹降到约 3–4%；这里看**质量**。）
- 数据（全部来自 `runs/e2_c2_q3-4b-2507_lora_pro6000_20261106/`）：
  - dev（dev_select 500 题，greedy）：step 0（训练前，基座 + C2 prompt）vs step 150（训练后），并报告 20…140 各步的数量。
  - 训练 rollouts：第 1–10 步 vs 第 141–150 步（每步 512 条轨迹，采样）。
  - 另报 D9 封存 test 上 C2 的笔记（`runs/d9_test_c2_*`），只作参考。
- 抽取：每次 `note` 工具调用（轨迹各步 output 中的 `<tool_call>`，`name == "note"`），取 `text`、`source`（引用的文档编号），以及该步 prompt 中编号为 `source` 的段落原文。解析失败或引用了不存在编号的记为「无效笔记」，计数不送 judge。
- 指标：
  1. **自动**：引用段落是否属于 HotpotQA 官方 supporting facts 的标题（dev 题可查，来源 HF `hotpotqa/hotpot_qa` distractor validation，按行与 Agent-R1 validation 对齐，7405/7405 题问题文本一致；train rollouts 对应 HF train split，同样按问题文本匹配）；笔记是否包含标准答案（规范化后子串）。
  2. **Gemini judge**（Vertex AI `gemini-3.8-flash`，thinking level medium，temperature 0，JSON 输出），每条笔记给出：
     - `faithful`：笔记内容是否被引用段落支持（yes / partial / no；no = 幻觉或与段落矛盾）；
     - `relevant`：笔记是否记录了回答该问题所需的信息（yes / no）。
     judge 输入只有问题、标准答案、引用段落、笔记文本；不告诉 judge 来自训练前还是训练后（顺序打乱）。
- 抽样：每组笔记全部送 judge；若某组超过 400 条，随机抽 400 条（seed 20261108）。
- 判读：比较训练前后 `faithful=yes` 比例、`relevant=yes` 比例、supporting-fact 命中率，给 95% Wilson 区间；样本量小（预计训练后 dev 只有几十条），结论以区间为准。
- judge 校验：随机抽 30 条，由 Claude 人工复核 judge 判定，记录一致率；一致率 < 80% 时结论降级为「仅供参考」。
- 预计费用：约 1000 次调用，Vertex 赠金内（< $2）。

## 设定变更（2026-11-08，DeepSeek 开跑前写定，用户确认）
- 起因：Gemini judge 判到 630 / 751 条时（顺序已打乱，缺失随机），Google 以「疑似违反服务条款」限制了项目 `llm-rl-lab-2026` 的 Gemini 访问（先是 token 过期 401，补判时 403）；已停止调用，由项目所有者申诉。已判的 630 条保留在 `judged_gemini.jsonl`。
- 变更：**全部 751 条用 DeepSeek 重判**（`deepseek-flash` = DeepSeek-V4.1-Flash，reasoning_effort high，temperature 0，JSON 模式；prompt、输入字段、抽样集合与 Gemini 完全相同），结果写 `judged_deepseek.jsonl`。
- 主结果以 DeepSeek（覆盖全部 751 条）为准；Gemini 630 条作为第二个 judge，报告两者在共同 630 条上的一致率和 Cohen's kappa（faithful 三分类、relevant 二分类）。两个 judge 结论方向不一致时，在结论里写明并降级。
- 人工复核：仍按原设定随机抽 30 条，Claude 复核，报告与两个 judge 各自的一致率。

## 结果（跑完填写）
- 数据量：
- 主结果：
- judge 复核一致率：
- 结论：
