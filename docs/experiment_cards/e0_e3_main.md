# 实验卡：e0_e3_main（E0–E3 主实验：C0–C3 四种上下文策略的 GRPO 训练）

> 开跑前填写「设定」并冻结；四组共用本卡，**组间只有 context policy 不同**。数字只能来自 runs/<run_id>/ 日志。

## 设定（E0 开跑前冻结，2026-11-05）
- 问题：在多跳搜索 Agent 的 GRPO 训练中，上下文策略（C0 append / C1 window / C2 模型自写笔记 / C3 规则压缩）是否影响 dev EM 与每步 prompt 长度。
- run_id：`e{0,1,2,3}_c{0,1,2,3}_q3-4b-2507_lora_pro6000_<首次开跑日期>`；驱动脚本 `scripts/autodl/e_run.sh <policy...>`
- 代码：D3 验收通过的版本（`docs/experiment_cards/d3_policies_tput.md`）；flow `configs/flows/ssrl_c{0..3}.yaml`；上游 Agent-R1 `b124aa4`；私有镜像 `image-f5cf94c23b`
- 硬件：AutoDL RTX PRO 6000 96GB。四组同卡型、同镜像、同卡数：**每组固定 1 张卡，多组在同一台机器上并行**（双卡机两组、四卡机四组；2026-11-06 用户确认，取代 2026-11-05 的「每组 2 卡」）。
  - 改动理由（实测，见下方「单组 2 卡记录」）：2 卡只比 1 卡快 1.19 倍。rollout 占每步约一半，受最慢一条轨迹（4–5 轮串行）限制，加 vLLM 副本不缩短；按费用算，每组 1 卡便宜约 40%，四组并行时总时长也更短。
  - 四组都在新配置下从头训练；按每组 2 卡跑到第 32 步的 E0 只作为工程记录，不进入四组比较。
  - checkpoint 按单卡保存，暂停后可在任何一台 RTX PRO 6000 的任意一张卡上续训。
- 模型：Qwen3-4B-Instruct-2507；LoRA rank 32 / alpha 32 / all-linear，lr 1e-5（锁定，与 D3 相同）
- 训练：GRPO（每组 1 卡），**每步 64 题 × G=8，150 步**（2026-11-06 批大小校准定，见下方结果；校准在 2 卡上做，单卡 64×8 每步约 372 秒（D3 实测），按用户事先偏好仍用 64）。一步一次更新（mini batch = 每步题数）。`e_run.sh` 里 `TRAIN_BATCH_E=64`，四组统一。max_prompt 8192、单步 response ≤1024，micro batch 4，不 offload，vLLM 显存 0.5，8 个 agent worker。

### 2 卡验收与批大小校准（E0 开跑前，在第一台 2 卡机器上做一次：`scripts/autodl/e_preflight.sh`；2026-11-06 已完成，之后正式配置改为每组 1 卡，本节保留为记录）
- 2 卡验收（事先写定，全部通过才开跑 E0）：
  - 冒烟：`E_SMOKE=1 e_run.sh c0 c2`（0.6B，2 卡，每组 4 步，第 2、4 步存档 + probe 300 题 greedy 评测），`check_replay.py --expect-steps 4 --expect-trajs 64` 通过，`adapters/step_2`、`step_4` 都在。
  - 续训：0.6B C2 2 卡，10 步 → 续到 12 步，`check_resume.py` 通过（同 D3 标准）。
  - 训推一致：4B 2 卡吞吐 run 里 `training/rollout_probs_diff_mean` 与 D3 单卡（0.0061–0.0065）同一量级（≤ 0.02）。
- 批大小校准：
  - 动机：GRPO 中同一题的 8 条轨迹全对或全错时优势为 0，不提供梯度。D3 吞吐（单卡 64×8，各 3 步）实测每步"有效题"（有对有错）：C0 9 / 13 / 13 题（18.2%），C2 12 / 17 / 17 题（24.0%）（`scripts/count_mixed_groups.py`，`d3_tput_{c0,c2}_q3-4b-2507_lora_pro6000_20261105`）。按此比例，每步 32 题只有约 6–8 题有效，64 题约 12–15 题。
  - 做法：4B C0、2 卡，32×8 和 64×8 各跑 3 步（不存档），run_id `d3_tput_c0_q3-4b-2507_lora_b{32,64}_2gpu_pro6000_<日期>`。取第 2–3 步平均每步耗时 t32、t64，并统计有效题数。
  - 决策规则（事先写定）：每小时有效题数之比 ≈ t64 / (2·t32)。若 t32 > 0.6·t64（32 题省不到 40% 时间，按每小时有效题数算 64 题效率高出 20% 以上）：用 64 题；否则把两种配置的实测每步耗时和四组总费用报给用户，由用户决定（用户已表示为实验效果可以接受 64 题的花费）。
  - 同时记录 1 卡→2 卡提速：D3 单卡 64×8 C0 第 2–3 步平均 372 秒 / 2 卡 t64。
- 结果填在本卡末尾，E0 开跑前写定，之后不再改。
- 其余沿用 recipe 默认，四组一致：不加 KL（`use_kl_loss=False`、`use_kl_in_reward=False`；PLAN §7 写的 KL 0.001 以 CLAUDE.md「以 recipe 默认为准」为准不加）、`loss_agg_mode=seq-mean-token-mean`、`norm_adv_by_std_in_grpo=True`、奖励只用 EM、5 步 / ≤4 并行检索 / top-k=5 / 自动首搜。训练数据为 recipe 的 HotpotQA train，seed 和数据顺序四组相同。
- 策略参数：C1 `window_k=1`；C2 每步最多 3 条 note、单条 ≤300 字符；C3 每步最多 6 句、重叠 ≥2（均为 D3 定下的值）。

### checkpoint 与 dev 评测（PLAN §5，事先写定）
- 开训前（step 0）和每 20 步（20, 40, …, 140）及最后一步（150）在 dev_select 500 题上 greedy 评测（veRL 训练内 validation，`val_kwargs` greedy、n=1），生成结果存 `runs/<id>/val/`。
- 每 20 步存 checkpoint，完整 checkpoint 只保留最近 2 个（用于续训）；每次存档后把 LoRA adapter 复制到 `runs/<id>/adapters/step_<N>/`，全部保留（用于最终评测）。
- **选择规则**：取 dev EM 最高的 step（不含 step 0）；并列取更早的一步。封存 test 只在四组都训完、配置冻结后，用各组选中的 adapter 跑一次（D9）。

### 判断标准（事先写定）
- 每组训练有效性（PLAN D4–D5 验收）：前 40 步内 `critic/score/mean`（训练 EM）有上升趋势；全程无 NaN / inf；抽查 step 20、step 100 各 5 条轨迹格式正常。不满足时先停下与用户讨论，**不在组内改超参**。
- 主比较：四组选中 checkpoint 的 dev EM 和每步平均 prompt token 数；按检索轮数（1 轮 / ≥2 轮）分层报告（PLAN 2026-11-04 补充）。dev 只用来选 checkpoint 和看趋势，结论以 D9 封存 test 为准。
- C2 专门记录：有 note 调用的轨迹比例随训练的变化。若训练到 60 步时 C2 note 比例降到 < 1%，按 PLAN 止损规则讨论（结论保留为诊断，不调参追分）。
- 步数分布（`num_steps/mean`）和 response 长度作为混杂因素报告。

### 运行规范
- 中断续训：同样的命令重跑，`RESUME_MODE=auto` 从最近 checkpoint 继续（D3 已验证 LoRA 续训）；跨日期用 `TAG` 保持 run_id。
- 暂停与换机器（每组 checkpoint 按单卡保存，任何一台 RTX PRO 6000 的任意一张卡都能续训）：
  - 暂停：`touch /root/autodl-tmp/PAUSE`，训练在下一次存档（每 20 步）且 adapter 复制完后停止，最多多跑 19 步，不丢已训练的步数。
  - 原机器有空卡：直接开机，删掉 PAUSE，用同一 `TAG` 重跑 `e_run.sh <policy>`。
  - 换机器（同地区西北B）首选控制台的「克隆实例」，**勾选数据盘**：系统盘（环境）和数据盘（仓库、数据、模型、runs 含 checkpoint）整体复制到新实例（卡数够放下要续训的组即可），原实例不受影响。先 PAUSE 让训练在存档后停下、原实例关机，再克隆，避免复制到写了一半的文件。「跨实例拷贝数据」只搬数据盘，也可用。目标数据盘 ≥ 150GB。
  - 第一次开跑 E0 不用克隆 D3 实例：用私有镜像新开，数据从文件存储拷（约 4 分钟），prep 会做同样的旧文件清理。这样每组的环境都能从「镜像 + 脚本」复现，也不带 D3 数据盘上的旧 run 和临时文件。
  - 备用（控制台功能不可用时）：原实例用**无卡模式**开机执行 `scripts/autodl/run_cache.sh save <run_id>`（runs 目录 + 最近一个 checkpoint 存到文件存储 `/root/autodl-fs/ssrl-runs/`），新实例 `run_cache.sh load <run_id>` 后用同一 `TAG` 重跑。路径在所有实例上相同，卡号不必相同。
  - 跨地区：文件存储不通，需用 AutoDL 的跨实例拷贝或经本地中转（4B checkpoint 实测 18GB / 个，慢），尽量只在西北B开机。
  - 换机器后在实验卡记录新机器的驱动版本（`meta.txt` 自动记录），续训后的前 2 步检查 reward 和 `rollout_probs_diff_mean` 与暂停前衔接。
- 每台新机器开跑前先做冒烟：`E_SMOKE=1 bash scripts/autodl/e_run.sh <要跑的 policy...>`（0.6B，每组 1 卡、各组并行，每组 4 步），确认训练、存档、adapter 保存正常，同机多组并行无冲突（Ray、vLLM 端口、检索、主机内存）。冒烟 run_id 前缀 `esmoke_`，不参与分析。
- 脚本结束 30 分钟后自动关机，另有 `MAX_HOURS` 硬上限（见下方预计时长）。
- 结果（不含完整 checkpoint，含 adapters）`autodl pull` 回本地，再上传 `gs://ssrl-runs-gqf-2026/runs/`。
- 磁盘：4B 完整 checkpoint 实测 18GB / 个（2 卡 E0 step 20），每组保留 2 个约 36GB；数据和模型约 23GB。**双卡机（两组并行）150GB 够用（约 72GB checkpoint + 23GB）；四卡机（四组并行，约 144GB checkpoint）扩到 250GB**。每组结束且 adapter 齐全（20…140、150）后脚本删除该组完整 checkpoint，多组连跑不累积；adapter 不齐时该组记为失败并保留 checkpoint。第一次存档（step 20）后核对 checkpoint 和 adapter 大小（`runs/<id>/adapters.log`），记到下方结果。
- 预计时长 / 费用（每组 1 卡，2026-11-06 修订；单卡每小时按 2 卡价格折半约 ¥7，以账单为准）：
  - 单卡 64×8 每步约 372 秒（D3 实测，单组独占机器；同机两组并行时的速度由开跑后前几步实测补记）。150 步约 15.5 小时 + 9 次 dev 评测和存档，每组约 **16–16.5 小时、约 ¥115**；四组约 ¥460。
  - 双卡机两组并行：两轮共约 33 小时；四卡机四组并行：约 16.5 小时。
  - `MAX_HOURS` 硬上限：每组 22 小时，按每个槽位要顺序跑的组数累加，另加 2 小时（双卡机跑四组为 46 小时）。
  - 已花费：2 卡预检约 1.5 小时；单组 2 卡 E0（停在第 32 步）约 3.1 小时。

## 2 卡验收与批大小校准结果（E0 开跑前填写）
（2026-11-06 凌晨，用户睡觉期间按其授权由 Claude 判定；日志 `/root/autodl-tmp/e_preflight*.log`，各 run 见下）
- 机器 / 驱动：AutoDL 西北B，2× RTX PRO 6000 Blackwell Server Edition 96GB，驱动 595.71.05，私有镜像 `image-f5cf94c23b`，数据盘 150GB，主机内存 1007GB（训练中可用约 846GB）。
- 2 卡首次上机发现 Agent-R1 多卡路径的 3 个问题（单卡不触发），合并为一个补丁 `third_party/patches/agent-r1_skip_rollout_corr_on_padding_microbatch.patch`（说明见 `third_party/README.md`），只影响诊断指标和 rollouts 日志，不改 loss / 梯度 / 优势：
  1. 全填充 micro batch 使 veRL rollout-corr 指标断言失败（第 3 步崩溃）；
  2. 跳过后各卡指标条数不等，`reduce_metrics` 报错；
  3. 为整除卡数补的复制行被写进 `rollouts/*.jsonl`（回放检查 `step_index=[0, 0]`）。
  另：私有镜像旧文件残留需在 `setup_env.sh` 前清理（`e_preflight.sh` 已改）。
- 冒烟：`esmoke_c0_q3-0p6b_lora_2gpu_pro6000_20261106` 4 步完成，step_2 / step_4 adapter 在（78MB，0.6B checkpoint 3.2GB），回放 PASS；C2 训练完成、adapter 在，回放仅因问题 3 失败。补丁后两组重跑（03:26–03:42）：C0、C2 回放均 PASS，step_2 / step_4 adapter 都在。
- 续训：`d3_resume_c2_q3-0p6b_lora_2gpu_pro6000_20261106` 10 步 → 12 步，`check_resume.py` RESUME_CHECK PASS（从 global_step_10 加载，lora_B 相对变化 0.259）。
- 训推一致：4B 2 卡 `rollout_probs_diff_mean` b32 0.0059–0.0067、b64 0.0061–0.0065，与 D3 单卡（0.0061–0.0065）同量级，≤ 0.02，通过。峰值显存 80.3GB / 卡。
- t32 / t64（2 卡，第 2–3 步平均）：t32 = (228.3 + 180.8)/2 = **204.5 秒**；t64 = (299.6 + 327.4)/2 = **313.5 秒**（`d3_tput_c0_q3-4b-2507_lora_b{32,64}_2gpu_pro6000_20261106`）。
- 每步有效题数（32 / 64，3 步）：32 题 7 / 9 / 4（20.8%）；64 题 16 / 11 / 15（21.9%）。
- 1 卡→2 卡提速（372 秒 / t64）：**1.19 倍**（预估 1.6–1.8 倍偏高）。分项：update_actor 171–195 → 116–132 秒（约 1.45 倍），old_log_prob 46–54 → 35–39 秒；gen 132–144 → 148–155 秒（未变快）。gen 中最慢一条轨迹（4–5 轮）约 117–120 秒，其余约 30 秒为 vLLM 唤醒 / 权重同步 / 休眠开销：rollout 受长尾轨迹的串行延迟限制，增加 vLLM 副本不缩短。缩短长尾需改 rollout 语义或上游显存管理，不在 E0 前做。
- 决定：**每步 64 题**。规则 t32 > 0.6·t64（204.5 > 188.1）→ 64；按每小时有效题数 64 题约高 30%；与用户事先偏好一致。另征询 gpt-6-astra（建议限时 60 分钟排查 gen；按其建议先做零成本分项计时，结论如上，不再排查），采纳其提醒把每组硬上限从 18 小时提到 22 小时（`PER_GROUP_H=20`），应对后期轨迹变长。
- 预计（按 t64 实测，每组 2 卡时的估计，配置改为每组 1 卡后作废）：每组 150 × 313.5 秒 ≈ 13.1 小时 + 9 次 dev 评测和存档 ≈ 14 小时、约 ¥200；四组约 56 小时、约 ¥800。

## 单组 2 卡记录（2026-11-06，工程记录，不进入四组比较）
- run：`e0_c0_q3-4b-2507_lora_2gpu_pro6000_20261106`（开跑时 run_id 为 `e0_c0_q3-4b-2507_lora_pro6000_20261106`，停止后改名以免与单卡正式 E0 重名；`meta.txt` 记有停止时间和原因）。配置与正式 E0 相同，只是每组 2 卡（FSDP dp=2 + 2 个 vLLM 副本）。
- 03:43 开跑，06:50 按用户决定停止，完成第 1–32 步。完整 checkpoint 已删，保留 metrics、rollouts、val（step 0、20）、adapters/step_20。
- 实测（每 10 步平均，单位秒）：

| 步 | 训练 EM | 每步 | gen | update_actor | old_log_prob | 平均轮数 | 训推 logprob 差 |
|---|---|---|---|---|---|---|---|
| 1–10 | 0.459 | 320 | 150 | 130 | 39 | 1.30 | 0.0064 |
| 11–20 | 0.487 | 317 | 148 | 128 | 39 | 1.30 | 0.0064 |
| 21–32 | 0.493 | 325 | 156 | 129 | 39 | 1.30 | 0.0062 |

- dev EM（dev_select 500，greedy）：step 0 = 0.448，step 20 = 0.482。grad_norm 0.021–0.071，无 NaN / inf。
- 4B 完整 checkpoint 18GB / 个，LoRA adapter 253MB；峰值显存 80.3GB / 卡。
- 开跑到第 20 步存档完成用时 116 分钟（含启动、step 0 评测、step 20 评测和存档），平均每步约 5.8 分钟，150 步预计约 14.5 小时。
- 结论：
  - 与 D3 单卡（每步 350–394 秒）比，2 卡只快约 1.19 倍。update_actor 快约 1.45 倍（没到 2 倍：轨迹按整条分卡、行数少的卡要补填充行，加上 FSDP 跨卡通信走 PCIe，两者各占多少未单独测），old_log_prob 快约 1.3 倍。
  - gen 不变快：每步要等全部轨迹完成，最慢一条（4–5 轮串行）约 117–120 秒，另有约 30 秒 vLLM 唤醒 / 权重同步 / 休眠的固定开销。
  - gen 约占每步一半且不能并行，按 Amdahl 定律，即使训练部分完美提速 2 倍，整体上限也只有约 1.35–1.4 倍。
  - 缩短长尾需要异步或 partial rollout（会变成轻度 off-policy），E0–E3 不做。
  - 因此改为每组 1 卡、多组并行。

## 结果（每组跑完填写）

| 组 | run_id | 实际时长 / 费用 | 每步耗时 | 峰值显存 | checkpoint / adapter 大小 | 选中 step | 选中 dev EM | 异常 |
|---|---|---|---|---|---|---|---|---|
| E0 C0 | `e0_c0_q3-4b-2507_lora_pro6000_20261106`（11-06 07:03 开跑，卡 0，与 E1 并行） | 开跑到 step 20 存档 137 分钟（约 6.9 分钟 / 步，含启动和 2 次评测），预计约 17 小时 | 第 16–20 步平均 382 秒（单组独占 D3 为 372 秒） | | checkpoint 18GB，adapter 253MB（step 20） | | dev EM step 0 = 0.446，step 20 = 0.482 | |
| E1 C1 | `e1_c1_q3-4b-2507_lora_pro6000_20261106`（11-06 07:05 开跑，卡 1，与 E0 并行） | 开跑到 step 20 存档 141 分钟，预计约 17.5 小时 | 第 15–19 步平均 394 秒 | | checkpoint 18GB，adapter 253MB（step 20） | | dev EM step 0 = 0.436，step 20 = 0.462 | |
| E2 C2 | | | | | | | | |
| E3 C3 | | | | | | | | |
