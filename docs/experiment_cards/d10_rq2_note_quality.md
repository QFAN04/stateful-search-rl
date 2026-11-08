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

## 结果（2026-11-08，`runs/d10_rq2_note_quality_20261108/`：`notes.jsonl`、`judged_deepseek.jsonl`、`judged_gemini.jsonl`、`summary.json`、`audit_claude.json`）
- 数据量：dev step 0 / 20 / 40 / 60 / 80 / 100 / 120 / 140 / 150 的笔记数 = 58 / 48 / 21 / 24 / 18 / 14 / 26 / 26 / 14（每次 500 题）；训练第 1–10 步 646 条（5120 条轨迹），第 141–150 步 236 条；D9 test 43 条。全部笔记格式有效、引用的段落都能找到（valid ≈ 100%），全部题目都匹配到官方答案和 supporting facts。送 judge 751 条（训练 1–10 步按设定抽 400 条）。
- 主结果（DeepSeek，全部 751 条；括号为 95% Wilson 区间；supporting fact 与「含答案」是自动指标，与 judge 无关）：

  | 来源 | 条数 | faithful=yes | faithful=no（幻觉） | relevant=yes | 引用官方 supporting fact | 笔记含标准答案 |
  |---|---|---|---|---|---|---|
  | 训练 1–10 步 | 400 | 0.718（0.671–0.759） | 0.022 | 0.930（0.901–0.951） | 0.856（0.827–0.881） | 0.615 |
  | 训练 141–150 步 | 236 | **0.805**（0.750–0.851） | 0.025 | **0.852**（0.801–0.891） | **0.746**（0.687–0.797） | 0.542 |
  | dev step 0（训练前） | 58 | 0.741（0.616–0.837） | 0.017 | 0.948 | 0.793 | 0.621 |
  | dev step 150（训练后） | 14 | 0.714（0.454–0.883） | 0.000 | 1.000 | 0.857 | 0.500 |
  | D9 test（训练后） | 43 | 0.767 | 0.070 | 0.907 | 0.721 | 0.465 |

  Gemini（630 条）给出的数值几乎相同：训练 1–10 步 faithful 0.723、relevant 0.912；141–150 步 faithful 0.814、relevant 0.794。
- judge 一致性：Gemini 与 DeepSeek 在共同 630 条上 faithful 一致率 0.968（kappa 0.918）、relevant 0.956（kappa 0.773）。人工复核 30 条（Claude 在看 judge 结果前写定标签，`audit_claude.json` 12:53 写入）：与 DeepSeek faithful 一致 0.900、relevant 0.933；与 Gemini 0.867、1.000，均高于 80% 阈值。分歧全是 judge 比人工更严（把「lays claim to being the oldest」写成「the oldest」之类的轻微夸大判为 partial），方向一致。
- 结论：
  1. **RL 没有提高笔记的整体质量，主要改变的是数量。** 笔记本来就基本忠实（训练前明确幻觉约 2%）、相关（约 93%）。训练后：忠实度上升（0.72 → 0.81，区间不重叠），但相关性下降（0.93 → 0.85）、引用官方 supporting fact 的比例下降（0.86 → 0.75，区间不重叠）、笔记直接含答案的比例下降（0.62 → 0.54）。两个 judge 方向一致。
  2. 同时笔记数量大幅下降（dev 58 → 14 条 / 500 题；训练 646 → 236 条 / 5120 条轨迹）。即只用 EM 奖励时，RL 让模型「少写、写得更保守（更贴原文但更少命中关键证据）」，而不是学会提炼关键证据。
  3. 与 D9 的行为结论一致：C2 的问题是笔记太少、没能保住跨轮证据，而不是笔记在胡写。支持 E4（笔记奖励按 supporting fact 命中给分）的设计方向。
  4. 局限：dev 训练后只有 14 条，单看 dev 无法下结论，主要依据训练 rollouts 的对比；「relevant」由 LLM 判断，含主观成分，但两 judge 与人工复核一致。
- 费用：DeepSeek ¥0.27（余额 12.54 → 12.27）；Gemini 630 次调用在赠金内。
