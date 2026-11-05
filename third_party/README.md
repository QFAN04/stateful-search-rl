# third_party

D1 添加 submodule 并在此记录 SHA：

| 仓库 | 用途 | SHA | 添加日期 |
|---|---|---|---|
| AgentR1/Agent-R1 | 训练底座（要求 verl==0.7.0；上游 PR #100 迁移 0.8.0 未合并，不跟 main） | `b124aa46534cbf2fb8bc8af11405774984c42ac7`（2026-07-21） | 2026-11-03 |
| PeterGriffinJin/Search-R1 | 退路 / 检索服务参考 | 待定（仅在需要时添加） | |

对上游的任何修改放在 `third_party/patches/`，并说明原因。

## 补丁（`scripts/autodl/smoke_lora.sh` 每次启动训练前自动打上，幂等）

| 文件 | 原因 | 日期 |
|---|---|---|
| `agent-r1_skip_rollout_corr_on_padding_microbatch.patch` | 多卡（dp≥2）时 `trajectory_batching` 按整条轨迹轮流分到各卡，行数少的卡用 `response_mask` 全 0 的填充行补齐；若某个 micro batch 全是填充行，veRL 的 `compute_offpolicy_metrics` 断言 `response_mask.any()` 失败，训练崩溃（`esmoke_c0_q3-0p6b_lora_2gpu_pro6000_20261106` 第 3 步）。补丁只在该 micro batch 没有有效 token 时跳过这项**诊断指标**（训推 logprob 差）；loss 按全局 batch 归一化，全填充 micro batch 的 loss 本来就是 0，训练不受影响。单卡没有填充行，不触发。 | 2026-11-06 |
