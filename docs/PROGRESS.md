# 进度日志

> 按项目工作日记录（个人原因未推进的日子不计入）。规则见 CLAUDE.md「进度规则」。

| 日期 | 工作日 | 可见产出（run_id / 文件） | 卡点 | 下一步 |
|---|---|---|---|---|
| 2026-11-03 | 1（D1） | Go：Agent-R1 HotpotQA recipe 在单卡上跑通 10 步 + 续跑到 12 步，320 条轨迹全部可回放（`d1_smoke_hotpotqa_q3-0p6b_20261103_0956`，实验卡 `docs/experiment_cards/d1_smoke_hotpotqa.md`）；Agent-R1 接口笔记 `docs/notes/agent_r1_interfaces.md` | us-central1 Spot A100 两次在 2 分钟内被抢占，D1 经同意改用按需 L4；1.7B 在 L4 上 OOM，冒烟用 0.6B；启动脚本 bug 浪费约 1 小时 | D2：4B probe（重点看步数分布）+ B0/B1 dev 基线 + logprob 一致性；需要稳定的 A100 |
| 2026-11-03 | （D2 尝试，不计工作日） | 无：D2 准备完成（固定划分 + manifest `data/manifests/`、C0 probe 脚本、分析脚本、实验卡 `docs/experiment_cards/d2_probe_c0_4b.md`），但 Spot A100 三次开机分别 21 / 4 / 6 分钟即被抢占，probe 未产出结果 | us-central1 Spot A100 不稳定；按规则停手 | 等扩容（11-05 15:22 后申请，含按需 A100）；或经同意改用按需 L4 跑 4B probe |
| 2026-11-03 | 2（D2，部分） | B0/B1 dev 基线：EM 0.128 / 0.312（`d2_b0b1_dev_q3-4b-2507_20261103`，实验卡 `docs/experiment_cards/d2_baselines_b0b1.md`）；4B C0 probe 完成：greedy EM 0.4367，sample pass@1/8 = 0.430/0.533，mixed 21.3%，约 74% 轨迹一步作答（`d2_probe_c0_q3-4b-2507_20261103_1100`，实验卡 `docs/experiment_cards/d2_probe_c0_4b.md`）；自动抢占流程（`ssrl-poller` + A100 开机任务）跑通；B0/B1 与 logprob 一致性脚本 + 单测 | Spot A100 窗口短，靠轮询机自动重试完成；开启桶统一访问控制时漏补对象权限（已修） | Colab 跑 B0/B1 与一致性检查；决定是否四组统一关闭 force_first_search |
