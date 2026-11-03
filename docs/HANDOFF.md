# 会话交接：2026-11-02 规划会话 → Claude Code

> 来源：claude.ai 上与 Claude 的规划会话（2026-11-02）。本文件记录**做了什么、为什么这么定、还剩什么**。
> 规则以 `CLAUDE.md` 为准，细节以 `docs/PLAN.md` 为准；本文件解释它们背后的理由，避免后续会话重新走回头路。

## 1. 背景
- 用户目标：找 Agent + LLM 后训练方向的（日常）实习，需要两个**不是玩具**的简历项目。主项目 stateful-search-rl，副项目 agentic-verifier-code，**两周内并行完成**。
- 参考材料：用户收集的几份他人简历项目（Search-R1 / Code-R1 / RTL-RLVR 复现；Qwen3-4B GSM8K LoRA+GRPO；"SFT 初始化分布对搜索 Agent RL 的影响"；τ²-bench 零售多轮 Tool-Use Agent），以及 GPT 写的 v1 计划书和 Gemini 写的 v2 计划书。
- 用户偏好：中文；不要下意识附和，先判断，但不过度防御；重视原理，必须自己读懂核心代码。

## 2. 对旧方案的结论（不要退回去）
- **GPT v1**：实验纪律对（成本匹配、mask、隐藏测试隔离、不预设提升），但按"写论文"规划，周期一个多月、自建模块太多。保留它的纪律，不要保留它的规模。
- **Gemini v2**：节奏对、选现成底座对、离线执行缓存是好主意。但有几处必须避开：
  1. **预填了大量编造的数字**（EM 42.2%、token −73%、吞吐 ×2.3、FAR 42.6→24.2% 等），还写进了简历 bullet 和面试话术。**全部作废，绝不能出现在任何产物中。**
  2. 以为"丢弃历史文档"只是改约 90 行环境代码。实际上上下文每轮不同就必须做逐轮样本，否则重算的 logprob 是错的。这是选 Agent-R1 的原因。
  3. 一边说零方差组梯度严格为 0，一边说过滤后"收敛快 35%"，自相矛盾。动态采样（DAPO）是已知技术，只能说"采用"，不能说成创新。
  4. verifier 的"组内强制包含正负样本"把不同候选放进同一个 GRPO 组，破坏了 per-prompt baseline。
  5. 教师蒸馏给教师看隐藏标签，会导致倒推合理化。
  6. 奖励一次叠加 4 项，结果无法归因；单卡 A40 跑 7B 全参 GRPO 不现实；上下文长度的"峰值"按求和来算，算错了。
- **参考简历的教训**："dev EM 1.4%→81.2%（训练批次内峰值）"这种口径会被追问，我们只报封存测试；它敢写"方案无增益因此停止"是加分项，我们也照写负结果。

## 3. 最终决策及理由
- **主项目研究问题**：把上下文管理（C0 append / C1 滑窗 / C2 模型自写证据笔记 / C3 规则压缩）作为 RL 实验变量，看准确率–token 帕累托关系，以及 RL 是否学会了提炼证据。
  - 理由：问题是真实的（MemAgent、harness-1、RAGEN 都在做），和参考简历的"Search-R1 复现"有区分度，工程上有深度（逐轮样本、训推一致性，正是 τ²-bench 那份简历最有分量的部分），负结果也有意义。
  - 曾考虑过换成 τ²-bench：否决。原因是需要 LLM 扮演用户模拟器，环境更重，两周内风险太大。改为把它的工程点（逐决策样本、优势共享、训推 logprob 一致性）吸收进主项目。
- **主项目底座：Agent-R1**（基于 veRL 0.7.0，step-level MDP，由环境决定每步 observation，自带 HotpotQA recipe）。2026-11-02 查证时：veRL 原生 agent loop 和 Search-R1 都是"一条轨迹一个拼接序列 + 工具 token mask"，不支持逐轮重写上下文。Agent-R1 新版用户少，所以 D1 设 Go/No-Go，失败就退回 Search-R1。
- **副项目**：执行增强的生成式 verifier。用离线执行缓存，训练纯文本单轮 GRPO（veRL 0.7，和主项目同一个镜像）。不依赖 AgentV-RL，因为它面向数学任务。与参考简历"1.5B 标量 verifier 门控失败"的区别在于：我们做生成式、带执行证据、目标是选择和拦截。核心洞察点是**题内 AUROC**，用来区分真实区分能力和题目难度先验。
  - v3.1（用户回看 GPT v1 / Gemini v2 后补齐）：v2 对公开测试全过的候选几乎没有执行信号，所以加入**离线探针输入**（DeepSeek 只生成输入、不生成期望输出，证据是崩溃 / 超时 / 输出一致性），保留 v1"自己构造测试"的能力但避开期望输出不可靠的问题；加入 **MBR-exec 强基线**、未见 solver 评测、配对 bootstrap、Python 时限倍数、标签噪声说明；删除门控自修复；项目对外名称去掉 "agentic"（verifier 不调用工具）。
  - 同日审查后的修正：verifier 只看**本候选自己**的探针结果，候选间一致性不喂给 verifier（否则 N<8 的 BoN 会看到子集外候选），只由 MBR-exec 在子集内计算、H1 检验互补（曾考虑"固定参照池"方案，因副项目应保持简单、要多采样而放弃）；消融 A1b 完整重训、A1a 只做到 SFT；探针生成器 / 校验器代码只在沙箱运行；`gen_probes.py` 列为 ★。
- **模型**：两个项目统一用 Qwen3-4B-Instruct-2507。不选 Qwen3.5-4B，因为它是 Mamba 混合架构，在 veRL / vLLM 上有适配风险。副项目的 solver 是 Qwen2.5-Coder-7B-Instruct + Qwen3-4B 混合。
- **两周并行**：主项目 D1 启动，副项目 D2 启动，D14 一起收尾。副项目前一周以 CPU 工作为主，和主项目错峰使用 GPU。

## 4. 算力（用户已确认）
- GCP（额度 396 美元）：**一个 GCP 项目**供两个项目共用，申请 A100 80GB × 4（按需和 Spot 的配额都申请，2 张也能开工），外加 GPUS_ALL_REGIONS=4、CPUS≈96、L4×2 作为备用。账户必须先升级为付费账户，试用账户不能挂 GPU。两个项目分别用存储桶 gs://ssrl-runs-gqf-2026 和 gs://avc-runs-gqf-2026（GCP 项目 llm-rl-lab-2026，us-central1），VM 打标签区分。
- Colab（211 CU）：本机有 Colab CLI，单会话最长约 12 小时。用于开发、probe、副项目的生成和训练；跑完同步结果后销毁会话。
- TC2 A40：候补（需要连学校网络，单作业 6 小时上限）。
- 所有脚本都要支持 `--resume`，并按 checkpoint 规范保存（见 CLAUDE.md）。

## 5. 协作方式（用户已确认）
- 协议文件：`C:\Users\GQF\.gemini\config\skills\multi-model-collaboration\SKILL.md`（软链接自 `C:\Users\GQF\.claude\skills\...`），定义 codex / dsh 等 CLI 的调用方式和防冲突规则。Gemini 通过 Antigravity CLI（`agy`，`C:\Users\GQF\.local\bin\agy.exe`）调用。
- 用户采纳了 Gemini 的提醒"分工太碎、频繁切换 CLI 成本高"。**跑通实验、拿到日志第一；Claude 主导全局并写核心代码**；GPT（codex）只在约定节点做只读审查；Gemini / DeepSeek 只接互不相交的并行杂活；DeepSeek API 用作教师模型和 judge。可用模型以开工时查到的为准（GPT 6.1 sol / 6 astra，Gemini 4 Argon / 3.8 Flash，DeepSeek Flash）。

## 6. 本地环境
- 新建 conda 环境 `llm-rl`（Python 3.12，**不装 torch**），两个项目共用：`conda env create -f env/environment-local.yml`。只用于单测、数据处理和画图。不复用 cs336，避免污染课程环境。
- gcloud SDK：PATH 上只保留 `C:\Users\GQF\AppData\Local\Google\Cloud SDK\google-cloud-sdk`（自带 bundledpython）。`CLOUDSDK_PYTHON` 已删除，`D:\GoogleCloudSDK` 已从 PATH 移除（2026-11-02）。
- 训练环境：GPU 机器上的 veRL 0.7.x Docker 镜像，D1 固定版本，记录在 `env/GPU_ENV.md`。

## 7. 当前状态与待办
已完成（规划会话中）：
- [x] 两份 v3 计划书（`docs/PLAN.md`）
- [x] 两个仓库的骨架、CLAUDE.md、实验卡模板、本地环境配置文件

用户侧待办（开工前）：
- [x] GCP：账户已升级为付费；2026-11-02 配额申请结果：GPUS_ALL_REGIONS=1，CPUS_ALL_REGIONS=64；Spot A100 80GB 仅 us-central1、asia-southeast1 各 1 张；L4 各区域 1 张。新项目有启动限制，计划在产生 GPU 计费且满 48h 后再申请 us-central1 的 A100 扩容
- [ ] GCP 配额扩容（D4 正式训练前需要 ≥2 张 A100，否则按 PLAN §11「GCP 配额被拒或延迟」的退路：LoRA + 缩小训练集与步数）
  - **首次 GPU 计费开始：2026-11-03 07:21:49 UTC（北京时间 15:21）**，VM `ssrl-a100`（us-central1-c，Spot a2-ultragpu-1g）。
  - **满 48 小时 = 2026-11-05 07:22 UTC（北京时间 11-05 15:22）后提醒用户申请扩容**：`GPUS_ALL_REGIONS=4`、us-central1 `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS=4`。SSD_TOTAL_GB 已于 11-03 提到 1024GB。
  - 11-03 建机时 us-central1-a / -c 都出现过 Spot STOCKOUT，用 `scripts/gcp/retry_create.sh` 在两个 zone 间轮流重试拿到。
- [x] 修正 `CLOUDSDK_PYTHON`（2026-11-02 已删除该用户变量，PATH 上的 gcloud 用自带 bundledpython），`gcloud auth login`，项目 `llm-rl-lab-2026`
- [x] `conda env create -f env/environment-local.yml`（Python 3.12），两个仓库分别 `pip install -e .`，`pytest` 通过（2026-11-02）
- [x] 两个仓库分别 `git init` 并做首次提交（2026-11-02）
- [x] GitHub 私有仓库已建并 push（QFAN04/stateful-search-rl、QFAN04/agentic-verifier-code），提交身份 QIFAN004@e.ntu.edu.sg（仓库级配置）
- [ ] 前置阅读：Agent-R1 的 step-level 设计、Search-R1 的 `search_r1/llm_agent/generation.py`、veRL 的 GRPO 配置、MemAgent（BytedTsinghua-SIA 版）

Claude Code 开工后第一步（主项目 D1，详见 PLAN §8）：
1. 检查上面的用户侧待办是否都已完成（GCP 配额未到位时，用 Colab A100 / G4 单卡开发）。
2. 用子 agent 读 Agent-R1 源码（AgentEnvLoop / ToolEnv / HotpotQA recipe / 检索服务），回报接口位置。
3. 添加 submodule 并固定 SHA；拉起检索服务；用 Qwen3-1.7B 或 0.6B 跑 recipe 10 步。
4. Go/No-Go 判定，写入实验卡；之后由 codex 做 D1 审查。

## 8. 尚未决定、需要时再问用户的事项
- 全参还是 LoRA（D3 实测吞吐后决定）
- 是否需要 SFT 冷启动（D2 看 probe 的格式合法率，阈值 80%）
- 检索语料与索引（D1 优先用 Agent-R1 自带的，否则用 Search-R1 的 e5 + wiki-18）
- GCP 实际批下的卡数（区域已定：us-central1 为主力，与两个 GCS 桶同区）。D1 结论：us-central1 Spot A100 极不稳定（11-03 两次 2 分钟内被抢占），扩容时建议同时申请**按需** A100 80GB 配额
- D1 发现（2026-11-03，见 D1 实验卡）：recipe `force_first_search=true` + 小模型 → 轨迹几乎都是 1 步，上下文策略无作用空间。D2 看 4B 的步数分布后决定是否四组统一关闭自动首搜 / 在 prompt 中要求先检索（实验设计变更，需用户确认）
- 副项目 D11：是否做 A3（N=8 时把候选间一致数给 verifier，只推理）；若 A3 显示帮助很大且有余力，是否上"固定参照池"方案。背景与代价见副项目 PLAN §3.2（H1 已定为必做）
