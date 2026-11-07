# 实验卡：d9_sealed_eval（D9 封存测试 + OOD + 放宽轮数）

> 开跑前填写「设定」并冻结（提交到 git 后才开 GPU）；封存 test 每个配置只跑一次。数字只能来自 runs/<run_id>/ 日志。

## 设定（2026-11-08 用户确认冻结，之后不再改）

### 目的
- 主结论：E0–E3 四组（C0 append / C1 window / C2 自写笔记 / C3 规则压缩）各自选中的 adapter，在 HotpotQA 封存 test 1000 题上的 EM 和每题 prompt token 数（EM–token 帕累托），按检索轮数（1 轮 / ≥2 轮）分层。
- 难度外推（PLAN X1，事先登记）：MuSiQue 500、2Wiki 500（只评测，不训练）；以及把每题最大步数从 5 放宽到 10。
- 参照：B0（无检索）/ B1（单次检索）在同样的题上各跑一次。

### 被评测的模型（按 e0_e3_main 的选择规则，已定，不再改）
| 组 | run_id | 选中 adapter | dev EM |
|---|---|---|---|
| C0 | `e0_c0_q3-4b-2507_lora_pro6000_20261106` | `adapters/step_80` | 0.526 |
| C1 | `e1_c1_q3-4b-2507_lora_pro6000_20261106` | `adapters/step_150` | 0.524 |
| C2 | `e2_c2_q3-4b-2507_lora_pro6000_20261106` | `adapters/step_150` | 0.512 |
| C3 | `e3_c3_q3-4b-2507_lora_pro6000_20261106` | `adapters/step_140` | 0.524 |
- 基座：Qwen3-4B-Instruct-2507（与训练同一份本地权重）。B0/B1 用基座（不加 adapter），与 D2 相同的 `scripts/baselines_b0b1.py`。
- adapter 存放：AutoDL 西北B文件存储 `/root/autodl-fs/ssrl-runs/<run_id>/adapters/`（2026-11-07 从原实例拷贝，逐文件名+大小核对一致）。

### 数据（manifest 已提交）
| 名称 | 题数 | 来源 | manifest |
|---|---|---|---|
| HotpotQA test | 1000 | 官方 dev 中与 dev_select 互斥（D1 切分，封存至今未用） | `data/manifests/hotpotqa_test.json` |
| MuSiQue | 500 | Agent-R1-data `musique_validation.parquet`（MuSiQue-Ans dev，2417 题）随机 500，seed 20261103 | `data/manifests/ood_musique_500.json` |
| 2Wiki | 500 | Agent-R1-data `2wikimultihopqa_validation.parquet`（dev，12576 题）随机 500，seed 20261103 | `data/manifests/ood_2wiki_500.json` |
- 切分脚本：`scripts/make_splits.py`（HotpotQA）、`scripts/make_ood_splits.py`（OOD）。OOD parquet 字段与训练用 HotpotQA 相同（`prompt` / `reward_model.ground_truth`），直接作为 `data.val_files`。

### 检索（每个数据集用自己的段落库，与 HotpotQA 设定一致）
- 检索器、编码模型、top-k、索引类型全部与训练相同：bge-large-en-v1.5、query instruction 同上游默认、top-k=5、FAISS `Flat` 内积（精确检索）。只换 `HOTPOTQA_CORPUS_DATA_ROOT` 指向的段落库。
- HotpotQA：训练用的 `agent-r1-data/hotpotqa_corpus`。
- MuSiQue：上游 `build_retrieval_corpus.py --dataset musique`（默认含 ans + full 的 train/dev，去重），输入为 MuSiQue 官方 jsonl（HF `bdsaglam/musique`：`musique_{ans,full}_v1.0_{train,dev}.jsonl`，sha256 前 8 位 83a75b1e / 15fa6379 / b1cd998f / 8cab31d5），2026-11-07 在本地建好：**125,450 段**，`hpqa_corpus.jsonl` sha256 `05c2a8d6…`。上游发布的 MuSiQue 索引（约 13.9 万向量）缺对应文本文件、段数也对不上，不使用；用上游 `build_index.py` 自己建索引。
- 2Wiki：上游发布的 `2wikimultihopqa_corpus/hpqa_corpus.jsonl`（ModelScope `Melmaphother/Agent-R1-data`，3,015,844,304 字节，sha256 `50b6aa52…` 与发布方一致，**5,902,082 段**），上游 `build_index.py` 建索引（PRO 6000 上做，约 0.5–1 卡时）。
- 建库产物、段落数、索引大小记到本卡结果；建好的语料和索引存文件存储 `/root/autodl-fs/ssrl-data/ood/`。

### 评测方式（与训练时 dev 评测完全同一路径）
- 用训练脚本的 validation 路径（`scripts/autodl/smoke_lora.sh` → `agent_r1.trainer.main_agent_ppo`）跑 `trainer.val_only=True`，`actor_rollout_ref.model.lora_adapter_path=<选中 adapter>`；flow `configs/flows/ssrl_c{0..3}.yaml`（与训练相同）；`val_kwargs` greedy、n=1、temperature 0、top_p 1；max_prompt 8192、单步 response ≤1024；生成结果存 `runs/<run_id>/val/`，指标进 `metrics.attempt*.jsonl`。
- 评分：主指标 EM = 训练奖励 `recipes/hotpotqa/reward_fn.compute_score`（只对 `ground_truth`），与训练和 dev 口径一致。辅助（离线从 val dump 算，不影响主指标）：token F1；OOD 另报「命中任一答案别名」的 EM（MuSiQue 500 题中 136 题、2Wiki 500 题中 276 题有多个别名）。
- 记录：EM、`prompt_tokens`、`num_rounds`、`num_tool_steps`、`num_notes`（C2），按检索轮数 1 / ≥2 分层。
- 放宽轮数（X1）：flow 只把 `max_steps` 5 → 10（新 yaml `configs/flows/ssrl_c{0..3}_x1.yaml`，其余键不变），在 test、MuSiQue、2Wiki 上各跑一次。超长 prompt 沿用 flow 现有的段落截断（记录截断告警条数）。

### 运行顺序与验收（事先写定）
1. 环境：私有镜像新开单卡 RTX PRO 6000（西北B，数据盘 ≥ 120GB：模型和数据约 23GB，2Wiki 语料 + 编码缓存 + 索引约 51GB），数据、模型、adapter、OOD 语料从文件存储拷入（`scripts/autodl/d9_eval.sh prep`，含 test / OOD manifest 哈希核对）。
2. **评测路径一致性检查（不碰封存数据）**：C0 `step_80` 用本评测路径在 dev_select 500 题上跑一次，EM 应与训练时记录的 0.526 相差 ≤ 2pp（与迁移验收同一阈值；vLLM greedy 在不同批次组合下不保证逐位相同）。不通过：停下查原因（adapter 加载、LoRA 同步到 vLLM），**不跑 test**。
3. 建 MuSiQue、2Wiki 索引，各用 5 个问题做检索冒烟（答案段落能否在 top-5 里找到，只看是否正常工作，不做筛选）。
4. 主评测：四组 × {test, MuSiQue, 2Wiki} × {5 步, 10 步}，共 24 次；B0/B1 × {test, MuSiQue, 2Wiki}。每个配置只跑一次；失败（崩溃、中断）可重跑，重跑记录在下方，不因结果好坏重跑。
5. 全部跑完：结果拉回本地，`runs/` 上传 GCS，实例关机。

### 硬件 / 预计时长 / 费用
- 单卡 RTX PRO 6000（与训练同卡型、同镜像）。2Wiki 的 Flat 索引约 24GB，检索工具在每个 agent worker 进程各加载一份：评测 2Wiki 时若实例内存 < 250GB，把 agent worker 从 8 降到 2（只影响速度，greedy 结果不变），记录在结果里。
- 预计：dev 一致性检查约 0.2 卡时；建索引约 0.5–1 卡时；主评测约 1 万题（5 步）+ 1 万题（10 步，轨迹更长）约 2–3 卡时；合计约 3.5–4 卡时，约 ¥30（按单卡 ¥7/h 估，以账单为准）。

### 停止规则
- 第 2 步不通过：停下与用户讨论，不跑封存数据。
- 任一 OOD 索引建不成或检索冒烟异常：先跑 HotpotQA test，再处理 OOD，不改检索设定迁就结果。

## 结果（跑完填写）
- 实际时长 / 费用：
- 环境：新实例（西北B 文件存储，SSH 网关 westd）RTX PRO 6000 Blackwell 96GB，驱动 580.95.05（训练机 595.71.05），同私有镜像；容器内存上限 110GB → 按设定 2Wiki 用 2 个 agent worker。仓库 6d0127d（本卡冻结版本），Agent-R1 b124aa4。adapter sha256 前 16 位：C0 step_80 `00915422e0f72916`、C1 step_150 `d18f79773c9be641`、C2 step_150 `93b3cb5d2f21ae41`、C3 step_140 `6c4dfb3d02982ec6`。
- dev 一致性检查（`d9_dev_c0_q3-4b-2507_lora_pro6000_20261108`）：EM **0.508**（训练时 0.526，差 1.8pp，≤ 2pp → **PASS**，按规则继续）。逐题对比 500 题：输出逐字相同 159 题；对→错 10 题、错→对 1 题；平均步数相同（1.422 / 1.422）。基座无 adapter 时为 0.446，adapter 确实加载。
  - 判断（2026-11-08 01:3x，用户休息中，咨询 gpt-6-astra 后由 Claude 决定）：继续按冻结规则跑完，不改配置。理由：① step_80 是 8 次 dev 评测中最高的一次，复测向均值回落（选择偏差）可以解释翻转偏向一侧；② vLLM greedy 在不同批次组合 / 驱动下不保证逐位一致；③ 四组用同一新路径评测。astra 提醒：10:1 不能单凭此定因，也不能证明新旧路径等价；不同上下文策略受数值漂移的影响不一定相同，组间小幅差距要保守解释。报告中写明：dev 0.526 → 0.508（预设 ±2pp 内），逐字一致 159/500，翻转 10:1，跨环境等价性未证实。
- 检索冒烟（`d9_retrieval_smoke_pro6000_20261108/smoke.log`，CPU 上用上游检索工具；冻结的第 3 步原本没写进 `d9_eval.sh`，在 OOD 评测开始前手工补做）：两个索引的向量数与语料行数一致（MuSiQue 125,450 / 125,450；2Wiki 5,902,082 / 5,902,082）；各 5 题 top-1 段落都与问题实体相关；答案字符串出现在 top-5 中：MuSiQue 3/5、2Wiki 4/5（多跳题单次检索找不全是预期的，只用于确认检索正常，不做筛选）。另见 2Wiki 有一题 ground_truth 为 "at"（疑似数据集自身的答案截断），记录不处理。
- 建库：MuSiQue 125,450 段、索引 491MB；2Wiki 5,902,082 段、索引 23GB（`build_index.py` 在 PRO 6000 上，01:13 / 01:17 完成，两库共约 50 分钟）；已回存文件存储。
- 主结果表：
- 异常 / 重跑记录：
- 结论与下一步：
