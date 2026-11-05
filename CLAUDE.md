# CLAUDE.md — stateful-search-rl（主项目）

在多跳搜索 Agent 的 RL 训练中，把**上下文管理**作为实验变量：RL 能否教会模型自己提炼证据，在更少 token 下保持或提升准确率？

- **新会话先读 `docs/HANDOFF.md`**（规划会话的决策理由、已否决的方案、待办清单）。
- 完整计划：`docs/PLAN.md`（v3，2026-11-02 定稿）。与本文件冲突时以本文件「已定决策」为准。
- 姊妹项目：`D:/AI-Projects/agentic-verifier-code`（副项目，并行两周）。
- 当前状态：**D1、D2 完成（见 `docs/PROGRESS.md`）；下一步 D3：AutoDL 环境测试 + C1–C3 实现。**

## 已定决策（变更需用户确认）
- 底座：**Agent-R1（基于 veRL 0.7.0，step-level MDP）**，作为 git submodule 放在 `third_party/`，固定 SHA。
  D1 Go/No-Go：HotpotQA recipe 用小模型跑 10 步无报错且轨迹可回放；否则当天切换到 Search-R1，并自行实现逐轮样本。
- 模型：Qwen3-4B-Instruct-2507（部署时核验 revision）；冒烟用 Qwen3-1.7B / 0.6B。
- 数据：HotpotQA train 子集；dev-select 500（选 checkpoint）；test 1000（封存）；OOD：MuSiQue 500、2Wiki 500。
- 对照：C0 append / C1 sliding window / C2 模型自写证据笔记 / C3 规则压缩。奖励只用 EM。四组超参、步数、G、检索 top-k、最大轮数一致；如需 SFT，四组同源同规模。
- **以 recipe 默认为准（2026-11-03）**：动作协议、预算（5 步、一步 ≤4 个并行检索、top-k=5、单步 ≤1024 token）、EM 奖励都沿用 Agent-R1 HotpotQA recipe；C0 = recipe 默认 flow；C1–C3 只替换每步 prompt 的渲染函数（自写 `AgentFlowBase` 子类，不改上游）；C2 笔记是额外的 `note` 工具。loss 按步聚合不改，步数分布作为混杂因素报告。详见 PLAN §3.1–3.2。
- **环境配置保持 recipe 默认（2026-11-04）**：`force_first_search=true`、top-k=5，E0–E3 统一。预实验中 FFS-off / top-k=3 都没达标（见 `docs/experiment_cards/d2_env_probe.md`）。4B 约 25% 轨迹为多轮检索，主实验分析按检索轮数分层，报告中说明上下文策略的作用范围。
- **训练方式锁定 LoRA（2026-11-04 用户确认）**：整个实验矩阵（E0–E5）都用 LoRA，之后即使有了更多或更大的卡也不改成全参。理由：4B 全参需要 80GB 卡加多卡分片，花费约为 LoRA 的 2–3 倍；RL 的信号稀疏，LoRA 足够；四组同为 LoRA，组间比较仍然公平。LoRA 超参（rank、alpha、作用层、lr）只定一次，四组统一；如需试调，只用 C0 跑短程，并在报告中说明。
- **算力平台改为 AutoDL（2026-11-04 用户确认）**，取代原"单卡期间规则"（GCP `GPUS_ALL_REGIONS=1`）：
  - E0–E3 必须在**同一平台、同一 GPU 型号、同一环境（私有镜像）、同一卡数**上跑完，不跨平台或混用型号；允许多卡。
  - 卡型在 D3 环境测试后定，倾向 RTX PRO 6000 96GB（Pro 实例，可用 `autodl` CLI 自动创建）；不用 4090-48G / 3090-48G 等改装显存的消费卡跑正式实验。
  - **卡型和卡数已定（2026-11-06 用户确认，取代 11-05 的「每组 2 卡」）**：RTX PRO 6000 96GB，**每组固定 1 张卡，多组在同一台机器上并行**（双卡机两组、四卡机四组）。理由：实测 2 卡只比 1 卡快 1.19 倍（rollout 受长尾轨迹限制），每组 1 卡便宜约 40%；单组 2 卡的实测保留为工程记录（实验卡「单组 2 卡记录」）。checkpoint 按单卡保存，任意一张卡都能续训；换机器用克隆实例或西北B文件存储（`scripts/autodl/run_cache.sh`）。详见 `docs/experiment_cards/e0_e3_main.md`。
  - 正式训练前必须通过迁移验收（`docs/notes/compute_fallback.md` §6：0.6B 冒烟 + 严格回放、C0 greedy EM 与 0.4367 相差 ≤ 约 2pp、logprob 一致性同量级），通过后 D2 的基线和 probe 结论可以沿用。
  - 每个实例都设 `--ttl` 自动关机；创建多卡实例或单次预计花费超过 ¥20 前，先向用户确认。
  - Colab 仍只用于主项目的 B0/B1 和评测，以及副项目；GCP 保留桶和现有脚本，GCP 配额若批下也不再用于 E0–E3（避免四组跨平台）。

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

## 进度规则：每个工作日都要有可见产出
- 目的：避免"先搭完整架构 + 每个功能都跑全局测试"导致长期没有真实结果。
- **按"项目工作日"计算，不按日历日。** 用户因个人原因（课程、考试、休息、等 GCP 配额等）没有推进的日子不计入，也不催促。
- **每个工作日结束时，必须有一个能看到的新产出**：一条真实轨迹、一张曲线、一个评测数字、一个通过的关键单测、一份实验卡结果等。只搭环境、写框架、写文档，不算产出。
- **连续 2 个工作日没有可见产出：停下来，不要继续往下推。** 向用户说明卡在哪里，给出选项（砍范围 / 换路 / 用退路方案），由用户决定。
- 每个工作日结束时，在 `docs/PROGRESS.md` 追加一行：`日期 | 第几个工作日 | 可见产出（附 run_id 或文件路径）| 卡点 | 明天计划`。用户随时问"今天的可见产出是什么"，都要能对照这个文件回答。
- 判断进度看阶段性成果是否达成（PLAN §8 的验收标准），不看写了多少代码、多少测试。测试范围以 PLAN §4 的 `tests/` 清单为准（★ 关键模块 + `test_resume.py` 等运行保障）；清单之外不为小功能补测试，不追求全覆盖，不为每个小功能跑全局测试。

## 运行规范（AutoDL / GCP Spot / Colab CLI / TC2 通用）
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
- AutoDL：本机已装第三方 CLI `autodl`（`@minato-aqukin/autodl-cli` 0.3.1，已审查：无安装脚本，token 只发往 `api.autodl.com`）。官方 API 只能建 Pro 实例（PRO6000、H800 等 7 种规格，最多 4 卡），且没有可靠库存接口，只能直接尝试创建。境内访问不了 GCS：结果先 `autodl pull` 回本地，再由本地上传到 GCS；HF / GitHub 走 `source /etc/network_turbo`。系统盘 30GB 且只有系统盘进镜像：环境装系统盘，模型和数据放数据盘或文件存储。实例连续关机 15 天会被释放。

## 协作分工（用户已确认；遵循 multi-model-collaboration 协议）
- **跑通实验、拿到日志是第一优先级。** Claude 主导全局并编写核心代码。
- GPT（codex）：只读独立审查，只在约定节点介入（见下方「审查节点」），发现的问题由 Claude 修复。
- Gemini（通过 Antigravity CLI `agy` 调用）/ DeepSeek：只接与核心文件不相交、规格清楚的并行任务；DeepSeek API 用作教师模型 / judge。
- 同一文件同一时间只能有一个 agent 修改；任何外部 agent 说"完成"后，主控必须自己看 diff、跑测试。
- 用户本人必须读懂关键路径代码（标注 ★ 的模块），并能讲清对应单测在测什么。

## 与用户协作
- 用中文交流。
- 用户的需求或观点不要下意识附和，先判断再回答，但不要过度防御。
