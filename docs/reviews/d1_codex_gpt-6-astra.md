# D1 只读审查（codex，gpt-6-astra，2026-11-03）

> 原样归档。处理结果见 `docs/experiment_cards/d1_smoke_hotpotqa.md` 末尾「codex D1 只读审查的处理」。

审查结论：**现有证据支持“单卡训练栈跑通，并发生了 step 10→11/12 的续跑”，但不足以独立验收实验卡所称的“完整轨迹可回放、checkpoint 恢复可靠”。建议将无条件 Go 改为“有条件 Go，待补齐回放与恢复证据”。**

本次未修改文件、未运行训练、未联网下载。已核实 Agent-R1 为固定提交 `b124aa46534cbf2fb8bc8af11405774984c42ac7`，工作树无修改；本地仓库及已检查的 Python 环境未发现 veRL 0.7.0 源码，因此涉及其内部实现的结论明确保留核实边界。

1. **严重：`check_replay.py` 会漏检，并以成功状态结束，不能作为验收门禁。**

   依据：[scripts/check_replay.py:18](../../scripts/check_replay.py#L18)、[第 26–37 行](../../scripts/check_replay.py#L26)。

   只检查索引排序，没有检查从零开始、连续、唯一；没有检查预期文件范围、轨迹数量、UID 唯一性。`bad` 只打印，不影响退出码。

   本次通过纯内存输入验证，无文件写入：

   | 输入 | 实际结果 |
   |---|---|
   | 不存在／无 JSONL 的目录 | 无输出，退出成功 |
   | 重复索引 `[0, 0]` | `unreplayable=0` |
   | 单步缺少 `step_index` | `unreplayable=0` |
   | 零步轨迹 | `unreplayable=0` |
   | 空 input | `unreplayable=1`，仍退出成功 |

   **修复建议：**要求文件覆盖预期训练步、每文件轨迹数正确、`num_steps>0`、索引严格等于 `range(num_steps)`、UID 和字段类型合法；任何缺失或坏记录均返回非零退出码。至少展示并检查一条多步轨迹。

2. **严重：实验卡的回放验收证据不足，而且“回放”的能力边界写得过宽。**

   依据：[实验卡:30–32](../../docs/experiment_cards/d1_smoke_hotpotqa.md#L30)、`runs/d1_smoke_hotpotqa_q3-0p6b_20261103_0956/replay_check.txt:1`、[上游 ray_trainer.py:512](https://github.com/AgentR1/Agent-R1/blob/b124aa46534cbf2fb8bc8af11405774984c42ac7/agent_r1/trainer/ppo/ray_trainer.py#L512)。

   字段名称没有用错：上游确实写出 `trajectory_uid / num_steps / steps / step_index / input / output / score`。但有三个限制：

   - 本地没有 `rollouts/*.jsonl`，只有检查摘要及一条轨迹的 input 尾部、output 开头各 300 字符，无法独立确认“每一步完整”。
   - 上游使用 `skip_special_tokens=True` 解码。它支持逐步阅读模型输入输出，不能据此证明 token 精确重建或训推 logprob 重算。
   - 接口笔记建议把状态放入 `extra_fields`，但**进入 `non_tensor_batch` 不等于进入落盘文件**。[agent_flow.py:744](https://github.com/AgentR1/Agent-R1/blob/b124aa46534cbf2fb8bc8af11405774984c42ac7/agent_r1/agent_flow/agent_flow.py#L744) 收集任意 extra fields；[ray_trainer.py:517](https://github.com/AgentR1/Agent-R1/blob/b124aa46534cbf2fb8bc8af11405774984c42ac7/agent_r1/trainer/ppo/ray_trainer.py#L517) 落盘只传奖励附加信息，并额外处理 `request_id`。notes、doc_id、事件快照不会自动保存。

   **修复建议：**归档原始 rollout，先把 D1 验收明确为“逐步文本可审阅”。D2–D3 若要求状态重建或数值复核，需要显式保存状态快照、原始 token IDs、mask、采样参数及 logprobs。

3. **严重：`--resume` 不是严格恢复，可能静默重新训练或恢复到较旧状态。**

   依据：[smoke_hotpotqa.sh:27–34](../../scripts/smoke_hotpotqa.sh#L27)、`scripts/gcp/sync_from_gcs.sh:8`。

   `sync_from_gcs` 失败被 `|| true` 吞掉，随后无条件使用 `resume_mode=auto`，不验证 checkpoint 指针、分片完整性或恢复目标。依照接口笔记所描述的 auto 语义，缺 checkpoint 时可能从零开始。远端恢复也没有比较本地与远端 checkpoint 的完整性和步数，旧远端指针可能覆盖较新的本地指针。

   另有明确的边界错误：如果已经完成 10 步，再运行默认 `--resume`，包装脚本没有“目标已完成”检查。上游先将 global step 加一，再进入训练，并在循环内判断结束条件，可能额外训练第 11 步：[ray_trainer.py:916–960](https://github.com/AgentR1/Agent-R1/blob/b124aa46534cbf2fb8bc8af11405774984c42ac7/agent_r1/trainer/ppo/ray_trainer.py#L916)。

   **修复建议：**先选择并验证一个完整 checkpoint，再显式恢复；没有可用 checkpoint 时停止。远端先恢复到暂存目录，禁止盲目覆盖更完整的本地状态；若已达到目标步数，直接结束。

4. **严重：云归档 同步没有完整 checkpoint 的发布边界，退出同步失败也被隐藏。**

   依据：[smoke_hotpotqa.sh:47–49](../../scripts/smoke_hotpotqa.sh#L47)、`scripts/gcp/sync_to_gcs.sh:8`、[smoke_hotpotqa_inner.sh:73–74](../../scripts/smoke_hotpotqa_inner.sh#L73)。

   每 600 秒扫描整个仍在写入的 run，既不是“checkpoint 保存完成后同步”，也不能保证远端先收到全部分片、最后才收到 latest 指针。Spot 中断可能留下指向不完整目录的指针。

   后台同步和 EXIT 同步都忽略失败；`kill $SYNC_PID` 后没有等待正在运行的同步子进程结束，可能与最终同步重叠。仅有 EXIT trap，也不能保证 VM 抢占时获得完成大文件上传所需的时间。

   **修复建议：**只发布已完成 checkpoint；先上传分片和校验清单，验证完成后最后发布指针。串行同步并保留同步日志；最终同步失败必须明确报错，保留本地数据供补传。

5. **严重：删除脚本的实际删除范围大于头注释所述范围。**

   依据：`scripts/gcp/delete_vm.sh:2`。

   注释称“删除 VM 和启动盘”，实际执行 `--delete-disks=all`，可能同时删除附加的数据盘。目标由可覆盖的环境变量决定，没有核对实例标签、磁盘清单或备份状态。

   **修复建议：**默认保留磁盘；若确实需要删除启动盘，限定为启动盘。执行前展示并验证项目、zone、实例及磁盘清单，对数据盘删除单独处理。

6. **一般：attempt 编号仍可能覆盖历史指标或训练日志。**

   依据：[smoke_hotpotqa.sh:38–55](../../scripts/smoke_hotpotqa.sh#L38)。

   `文件数量 + 1` 不等于“最大编号 + 1”：若只有 attempt1、attempt3，会再次选择 attempt3。若启动在 file logger 创建前失败，只留下 `train.attempt1.log`，下一次仍会选择 attempt1，`tee` 会覆盖旧日志。并发启动同一 run 也没有锁。

   **修复建议：**使用独立 attempt 目录及原子创建／锁，或从所有 attempt 产物计算最大编号并排他分配。当前两份指标文件没有发现覆盖，但实现尚不可靠。

7. **一般：记录的代码 SHA 无法还原成功运行，实际模型 revision 也缺失。**

   依据：`runs/d1_smoke_hotpotqa_q3-0p6b_20261103_0956/meta.txt:2`、[smoke_hotpotqa.sh:43](../../scripts/smoke_hotpotqa.sh#L43)、[GPU_ENV.md:13](../../env/GPU_ENV.md#L13)。

   通过本地 Git 对比，记录的 `76183656`：

   - 仍使用会在空目录失败的 `ls glob` attempt 计数；
   - 没有 `max_position_embeddings=5120` 覆盖。

   因此成功运行依赖了未被该 SHA 表达的修改，`extra_overrides` 也没有记录这项长度覆盖。另外，环境文件记录的是 1.7B snapshot，实际成功模型是 0.6B。

   **修复建议：**保存实际运行脚本、dirty diff、完整解析后配置及文件哈希；补录 0.6B 和 tokenizer revision、数据与索引哈希。不能只用当前修正后的提交代替当时运行证据。

8. **一般：`max_position_embeddings=5120` 可以是冒烟 workaround，但不能直接认定对正式训练无副作用。**

   依据：[smoke_hotpotqa_inner.sh:31–51](../../scripts/smoke_hotpotqa_inner.sh#L31)、[实验卡:38–43](../../docs/experiment_cards/d1_smoke_hotpotqa.md#L38)、[PLAN.md:220](../../docs/PLAN.md#L220)。

   当前 `4096+1024=5120` 配套一致。但这是 **model 配置覆盖**，不能当成仅改变 vLLM 调度参数。FSDP 模型构造继承 veRL，具体传播路径本次因缺少其源码未复核。

   对默认静态 RoPE，缩小最大位置数**不必然改变范围内的旋转频率**，不能直接断言它导致训推不一致；反过来，没有实际模型配置、RoPE 类型及两端加载配置，也不能证明无影响。尤其不能把固定 5120 原样带到计划中的 8k 上下文实验。

   本次日志的最大 prompt 仅 2749 token，未覆盖 4096/5120 边界。L4 上遇到容量问题，也不能推出“4B/A100 同样必须如此修改”。

   **修复建议：**归档两端实际 HF/RoPE 配置；优先将长度限制放在 rollout server 配置或受记录的补丁中。若继续使用模型覆盖，长度应随实验预算计算，并在 D2 检验同 token 输入下的训推差异及边界长度。

9. **一般：优势广播结论正确，但建议的零和断言容易写错。**

   依据：[接口笔记:78–85](../../docs/notes/agent_r1_interfaces.md#L78)、[core_algos.py:300–342](https://github.com/AgentR1/Agent-R1/blob/b124aa46534cbf2fb8bc8af11405774984c42ac7/agent_r1/trainer/ppo/core_algos.py#L300)。

   上游确实先求轨迹总奖励，再按 prompt 分组归一化，最后广播到该轨迹每一步的有效 response token。但“同一 uid 组内优势和约为零”必须**每条轨迹只计一次**；对展开后的全部步骤求和，会因轨迹长度不同而不为零。只有一个样本的组还走 `mean=0,std=1` 的特殊分支。

   同一轨迹“优势相同”也应仅比较有效 token，padding 是零。

   **修复建议：**先按 `trajectory_uid` 去重再检查组均值；加入不同步数、不同 response 长度、padding 和单样本组案例。

   另外两点核对结果是正确的：

   - 非终局步显式给 `0.0` 符合本项目“仅终局 EM”的设定；`None` 会调用奖励函数。但这是奖励设计要求，并非框架禁止过程奖励。
   - `seq-mean-token-mean` 确实按有效步骤归一化；按整条轨迹分 mini-batch 不会消除长轨迹权重。依据：[core_algos.py:653](https://github.com/AgentR1/Agent-R1/blob/b124aa46534cbf2fb8bc8af11405774984c42ac7/agent_r1/trainer/ppo/core_algos.py#L653)、[trajectory_batching.py:182](https://github.com/AgentR1/Agent-R1/blob/b124aa46534cbf2fb8bc8af11405774984c42ac7/agent_r1/trainer/ppo/trajectory_batching.py#L182)。

10. **一般：把 `rollout_probs_diff_*` 直接称为“logprob 差”，缺少公式核实。**

    依据：[实验卡:21](../../docs/experiment_cards/d1_smoke_hotpotqa.md#L21)、[接口笔记:47](../../docs/notes/agent_r1_interfaces.md#L47)、[ray_trainer.py:1029–1051](https://github.com/AgentR1/Agent-R1/blob/b124aa46534cbf2fb8bc8af11405774984c42ac7/agent_r1/trainer/ppo/ray_trainer.py#L1029)。

    可确认的含义是：`calculate_log_probs=True` 请求 rollout 返回 logprobs；trainer 默认仍重新计算 `old_log_probs`，随后调用 veRL 的 `calculate_debug_metrics`。**它不等于启用 rollout logprobs 作为 PPO 分母，也不等于已通过一致性检查。**

    精确差异公式、是否先取指数、mask 和聚合口径位于缺失的 veRL 源码中，本次不能确认。现有报告也没有 PLAN 要求的逐 token 绝对 logprob 差及 p99。

    **修复建议：**暂以原始指标名报告，补存 veRL 0.7.0 对应实现后注明公式；D2 显式计算有效 token 上的 `abs(logp_rollout-logp_actor)`，并记录采样温度、权重版本和 token 对齐情况。

11. **一般：一步轨迹的统计成立，“随即直接作答”的原因判断尚未证实。**

    依据：[实验卡:35](../../docs/experiment_cards/d1_smoke_hotpotqa.md#L35)、`runs/d1_smoke_hotpotqa_q3-0p6b_20261103_0956/replay_check.txt:27`、[hotpotqa_agent_flow.py:332–362](https://github.com/AgentR1/Agent-R1/blob/b124aa46534cbf2fb8bc8af11405774984c42ac7/recipes/hotpotqa/hotpotqa_agent_flow.py#L332)。

    摘要显示 320 条中有 **312 条一步轨迹、8 条多步轨迹**。但 flow 在“没有解析到工具调用”时就终止并评分，不要求存在合法 `<answer>`；格式错误也可能表现为一步结束。检索异常还会被捕获后继续执行：[第 398–418 行](https://github.com/AgentR1/Agent-R1/blob/b124aa46534cbf2fb8bc8af11405774984c42ac7/recipes/hotpotqa/hotpotqa_agent_flow.py#L398)。

    **修复建议：**区分合法作答、工具解析失败、空／失败检索等终止原因；抽查真实多步样本。当前数据足以提醒上下文策略作用空间很小，但不足以把原因完全归于模型主动作答，更不能证明 C1–C3 已具备有效实验空间。

12. **一般：VM 创建、停止和任务结束之间缺少可靠绑定，存在空转计费风险。**

    依据：`scripts/gcp/retry_create.sh:25`、`scripts/gcp/config.sh:3`、`scripts/gcp/stop_vm.sh:5`、[smoke_hotpotqa.sh:49](../../scripts/smoke_hotpotqa.sh#L49)。

    - retry 在 `-a` 创建成功后只打印 zone，没有持久化目标；后续默认 stop/delete 仍指向 `-c`，可能停错实例或未停实际运行实例。
    - 训练成功或失败后只同步，不关机；bootstrap 的容器运行 `sleep infinity`。创建时的 10 小时上限提供兜底，但仍允许任务结束后长时间空转。
    - 默认 retry zones 都在 us-central1，**没有发现默认跨区域或自动切按需的行为**；但自定义 `--zones` 没有同区域校验，环境变量也可改变 provisioning。

    **修复建议：**保存创建结果中的 project/zone/name/instance ID，后续生命周期操作读取并核验；任务完成后同步、记录状态并停止指定实例。跨区域和 provisioning 变化应显式校验，不能仅依靠注释。

13. **建议：清理会误导 D2–D3 的旧接口说明。**

    依据：[接口笔记:40](../../docs/notes/agent_r1_interfaces.md#L40)、[第 60–68 行](../../docs/notes/agent_r1_interfaces.md#L60)、[PLAN.md:209](../../docs/PLAN.md#L209)。

    笔记仍称“C0 必须自己实现”，示例仍使用 6 步、top-k=3 和旧动作协议；这些与已确认的 recipe C0、5 步、top-k=5、Hermes 协议冲突。PLAN 的 B1 仍写 top-k=3“与训练一致”。

    **修复建议：**标记旧方案已废弃，统一当前有效配置，避免 B1 与 E0 的检索预算不一致。

14. **建议：实验卡数字大体有据，但应补充统计口径和缺失证据。**

    从两份指标日志重新计算：

    | 项目 | 核对结果 |
    |---|---|
    | attempt1 / attempt2 | 分别为 step 1–10 / 11–12 |
    | 10 步训练耗时 | step 时间合计 479.019 秒，加两次保存 32.834 秒，约 **8 分 32 秒**；不含初始化 |
    | 第 1 步、后续平均耗时 | 69.912 秒、step 2–12 平均 46.298 秒，支持卡中数字 |
    | allocated 显存指标峰值 | 17.896，支持 17.9；不能替代整卡峰值显存 |
    | diff_mean / diff_max 范围 | 0.006171–0.007112 / 0.107564–0.546787，支持卡中舍入值 |
    | 回放摘要得分范围 | 0–0.21875，限前 10 步；step 11 为 0.34375 |

    依据：`runs/d1_smoke_hotpotqa_q3-0p6b_20261103_0956/metrics.attempt1.jsonl:1`、`runs/d1_smoke_hotpotqa_q3-0p6b_20261103_0956/metrics.attempt1.jsonl:5`、`runs/d1_smoke_hotpotqa_q3-0p6b_20261103_0956/metrics.attempt1.jsonl:10`、`runs/d1_smoke_hotpotqa_q3-0p6b_20261103_0956/metrics.attempt2.jsonl:1`。

    VM 开关机时间、约一小时空转、1.7B OOM、checkpoint 加载原文及清理阶段报错，目前只有文档叙述，本地四个文件不足以独立核实。应补充对应原始日志和退出码。0.6B 与 L4 替代已明确披露，不属于隐瞒；结论需限定于该实际配置。

**对 D1 Go 的最终判断：**

- **判据 1：基本成立。** step 1–10 指标完整，且保存计时存在；但“整个进程无报错退出”和实际 checkpoint 文件还缺直接证据。
- **判据 2：暂不能验收。** 字段设计支持逐步文本审阅，但检查器有漏检，原始轨迹未提供。
- **判据 3：有较强支持，但验收不完整。** `resume=1` 后只出现 step 11、12，符合恢复行为；尚不能验证模型、optimizer、scheduler、RNG、数据进度全部恢复正确，更未证明 云归档 灾后恢复。

因此，**可以保留 Agent-R1 并有条件推进 D2，但当前实验卡“轨迹回放、checkpoint 续跑已全面可用”的无条件 Go 表述过强**。优先补齐现有原始产物、修正检查器并重新验收；现有发现尚不足以判定底座失败、必须切换 Search-R1。