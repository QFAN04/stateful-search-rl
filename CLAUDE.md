# CLAUDE.md — stateful-search-rl（主项目）

在多跳搜索 Agent 的 RL 训练中，把**上下文管理**作为实验变量：RL 能否教会模型自己提炼证据，在更少 token 下保持或提升准确率？

- **新会话先读 `docs/HANDOFF.md`**（规划会话的决策理由、已否决的方案、待办清单）。
- 完整计划：`docs/PLAN.md`（v3，2026-11-02 定稿）。与本文件冲突时以本文件「已定决策」为准。
- 姊妹项目：`D:/AI-Projects/agentic-verifier-code`（副项目，并行两周）。
- 当前状态：**规划完成，尚未运行任何实验。**

## 已定决策（变更需用户确认）
- 底座：**Agent-R1（基于 veRL 0.7.0，step-level MDP）**，作为 git submodule 放在 `third_party/`，固定 SHA。
  D1 Go/No-Go：HotpotQA recipe 用小模型跑 10 步无报错且轨迹可回放；否则当天切换到 Search-R1，并自行实现逐轮样本。
- 模型：Qwen3-4B-Instruct-2507（部署时核验 revision）；冒烟用 Qwen3-1.7B / 0.6B。
- 数据：HotpotQA train 子集；dev-select 500（选 checkpoint）；test 1000（封存）；OOD：MuSiQue 500、2Wiki 500。
- 对照：C0 append / C1 sliding window / C2 模型自写证据笔记 / C3 规则压缩。奖励只用 EM。四组超参、步数、G、检索 top-k、最大轮数一致；如需 SFT，四组同源同规模。
- 全参还是 LoRA：D3 实测吞吐后决定，四组必须一致。

## 关键模块（★ = 用户必须亲自读懂）
- ★ `src/ssrl/context_policies.py`：C0–C3 的 observation 构造
- ★ `src/ssrl/checks/logprob_consistency.py`：rollout 与 trainer 的 logprob 一致性
- ★ `src/ssrl/checks/step_advantage.py`：组内优势广播到每一步的校验
- `src/ssrl/actions.py`、`state.py`、`env_search.py`、`rewards.py`、`diagnostics.py`、`eval/`

## 审查节点（GPT codex 只读审查）
D1 环境跑通 → D3 核心模块合并 → D4 正式开跑前 → D13 报告定稿。

## 14 天节奏（详见 docs/PLAN.md §8）
D1 环境 + Go/No-Go → D2 probe + 一致性检查 → D3 策略实现 + 单测 + 吞吐 → D4–D8 E0–E3 训练 → D9 封存测试 + OOD → D10–12 分析 → D13–14 报告。
止损：D3 时 C2 跑不通 → C2 降为扩展；D8 时 E2 崩溃 → 保留诊断作为结论，不再调参追分。

## 红线（任何 agent 都必须遵守）
1. **任何数字必须来自 `runs/<run_id>/` 下的真实日志。** README、报告、简历里的每个数字都标注 run_id。禁止预填、估计或"预期结果"写成结果。
2. **一次只改一个变量。** 每个正式 run 开跑前在 `docs/experiment_cards/` 写好实验卡。
3. **封存测试集只在配置冻结后跑一次。** checkpoint 只按 dev 选。
4. **不在本地 Windows 跑训练，也不执行模型生成的代码。** 本地只做编辑、单测、数据处理、分析。
5. 上游代码放在 `third_party/`，固定 SHA；必须修改时用 patch 文件记录，不直接散改。
6. 不要为了让测试通过而改测试或放宽断言；先找原因。

## 运行规范（GCP Spot / Colab CLI / TC2 通用）
- 所有训练、生成、评测脚本支持 `--resume`：从最近 checkpoint 或已完成分片继续。
- checkpoint 包含：模型、optimizer、scheduler、RNG 状态、数据读取进度、step。`tests/test_resume.py` 验证恢复后曲线连续。
- 每 10–20 步或 30 分钟保存一次，并同步到 GCS 桶。
- 任务结束（无论成败）：同步 `runs/<run_id>/` 日志与结果 → 关机 / 销毁会话。
- 每次开机记录 $/h、吞吐、峰值显存到实验卡。

## 环境
- 本地（Windows）：conda 环境 `llm-rl`（两个项目共用，`env/environment-local.yml`），无 torch。
- GPU：Docker 镜像（veRL 0.7.x），D1 固定版本，记录在 `env/GPU_ENV.md`。
- 代码流转：本地编辑 → push GitHub → GPU 机器 `git pull` 后运行。
- GCP：与另一个项目共用同一个 GCP 项目；本项目用独立桶 `gs://ssrl-runs-gqf-2026`（us-central1），VM 打标签 `project=ssrl`。

## 协作分工（用户已确认；遵循 multi-model-collaboration 协议）
- **跑通实验、拿到日志是第一优先级。** Claude 主导全局并编写核心代码。
- GPT（codex）：只读独立审查，只在约定节点介入（见下方「审查节点」），发现的问题由 Claude 修复。
- Gemini（通过 Antigravity CLI `agy` 调用）/ DeepSeek：只接与核心文件不相交、规格清楚的并行任务；DeepSeek API 用作教师模型 / judge。
- 同一文件同一时间只能有一个 agent 修改；任何外部 agent 说"完成"后，主控必须自己看 diff、跑测试。
- 用户本人必须读懂关键路径代码（标注 ★ 的模块），并能讲清对应单测在测什么。

## 与用户协作
- 用中文交流。
- 用户的需求或观点不要下意识附和，先判断再回答，但不要过度防御。
