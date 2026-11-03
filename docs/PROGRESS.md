# 进度日志

> 按项目工作日记录（个人原因未推进的日子不计入）。规则见 CLAUDE.md「进度规则」。

| 日期 | 工作日 | 可见产出（run_id / 文件） | 卡点 | 下一步 |
|---|---|---|---|---|
| 2026-11-03 | 1（D1） | Go：Agent-R1 HotpotQA recipe 在单卡上跑通 10 步 + 续跑到 12 步，320 条轨迹全部可回放（`d1_smoke_hotpotqa_q3-0p6b_20261103_0956`，实验卡 `docs/experiment_cards/d1_smoke_hotpotqa.md`）；Agent-R1 接口笔记 `docs/notes/agent_r1_interfaces.md` | us-central1 Spot A100 两次在 2 分钟内被抢占，D1 经同意改用按需 L4；1.7B 在 L4 上 OOM，冒烟用 0.6B；启动脚本 bug 浪费约 1 小时 | D2：4B probe（重点看步数分布）+ B0/B1 dev 基线 + logprob 一致性；需要稳定的 A100 |
| 2026-11-03 | （D2 尝试，不计工作日） | 无：D2 准备完成（固定划分 + manifest `data/manifests/`、C0 probe 脚本、分析脚本、实验卡 `docs/experiment_cards/d2_probe_c0_4b.md`），但 Spot A100 三次开机分别 21 / 4 / 6 分钟即被抢占，probe 未产出结果 | us-central1 Spot A100 不稳定；按规则停手 | 等扩容（11-05 15:22 后申请，含按需 A100）；或经同意改用按需 L4 跑 4B probe |
