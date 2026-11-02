# StatefulSearch-RL

多轮搜索 Agent 的上下文管理强化学习：比较 append / 滑窗 / 模型自写证据笔记 / 规则压缩四种上下文策略在多跳问答上的准确率–token 帕累托关系，并分析 RL 是否真正学会了提炼证据。

> 状态：开发中，暂无实验结果。

- 底座：[Agent-R1](https://github.com/AgentR1/Agent-R1)（veRL）
- 模型：Qwen3-4B-Instruct-2507
- 计划：`docs/PLAN.md`

## 目录
```
third_party/      上游代码（submodule，固定 SHA）
src/ssrl/         本项目代码
configs/          实验配置
scripts/          启动 / 评测 / 同步脚本
tests/            单元测试
docs/             计划、实验卡
reports/          分析报告与图
data/manifests/   数据划分 ID 与 hash
env/              本地与 GPU 环境说明
```
