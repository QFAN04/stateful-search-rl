# 进度日志

> 按项目工作日记录（个人原因未推进的日子不计入）。规则见 CLAUDE.md「进度规则」。

| 日期 | 工作日 | 可见产出（run_id / 文件） | 卡点 | 下一步 |
|---|---|---|---|---|
| 2026-11-03 | 1（D1） | Go：Agent-R1 HotpotQA recipe 在单卡上跑通 10 步 + 续跑到 12 步，320 条轨迹全部可回放（`d1_smoke_hotpotqa_q3-0p6b_20261103_0956`，实验卡 `docs/experiment_cards/d1_smoke_hotpotqa.md`）；Agent-R1 接口笔记 `docs/notes/agent_r1_interfaces.md` | us-central1 Spot A100 两次在 2 分钟内被抢占，D1 经同意改用按需 L4；1.7B 在 L4 上 OOM，冒烟用 0.6B；启动脚本 bug 浪费约 1 小时 | D2：4B probe（重点看步数分布）+ B0/B1 dev 基线 + logprob 一致性；需要稳定的 A100 |
