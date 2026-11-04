# D3 只读审查（codex，gpt-6-astra，2026-11-05）

> 审查范围：提交 bc2cf7f。下面是原样归档的审查结果，末尾附处理记录。

发现 **2 个中等问题**，均已用内存样例复现。

1. **中等：C2 接受实际未展示的文档来源**  
   位置：[src/ssrl/env_search.py:141](/D:/AI-Projects/stateful-search-rl/src/ssrl/env_search.py:141)  
   `allowed_sources` 取自截断前的文档列表，没有考虑字符预算和后续 token 预算收缩。触发场景：一轮包含 4 个 query、各返回 5 篇长文档，`MAX_PROMPT=4096` 时文档 `[20]` 已被截掉，但 `note(source=20)` 仍被接受。  
   **后果**：违反“只能引用本步展示文档”的约束，无效来源笔记进入后续上下文，且不触发拒绝反馈，影响 C2 行为及来源有效性判断。

2. **中等：成功执行的 note 调用可能被统计为零**  
   位置：[scripts/analyze_probe.py:16](/D:/AI-Projects/stateful-search-rl/scripts/analyze_probe.py:16)、[第 60 行](/D:/AI-Projects/stateful-search-rl/scripts/analyze_probe.py:60)  
   `NOTE_RE` 只识别双引号形式。触发场景：模型输出 `{'name': 'note', 'arguments': {'fact': 'A fact', 'source': 1}}`；上游回退解析器可以恢复该调用，C2 也成功保存笔记，但 `frac_traj_with_note` 返回 `0.0`。  
   **后果**：低估笔记工具使用率，可能错误触发实验卡中“C2 未探索到 note”的判断。

其他已核实项：

- HEAD 为 `bc2cf7f`，上游固定为 `b124aa4`，工作区干净。
- 指定两份测试：**42 项通过，1 项上游对拍未运行**；C0 prompt 逐字比较通过。
- C0 源码提取后的 20 种依赖桩流程对拍通过；工具步奖励 `0.0`、无工具调用终局奖励 `None` 与上游一致。
- GRPO 独立重算的轨迹汇总、无偏标准差、单轨迹组特例和 mask 广播与上游源码一致。
- 启动脚本参数化默认值保持一致；三个 Bash 脚本语法检查通过。

**未核实**：本地缺少 veRL，未验证真实 LoRA checkpoint 读取与恢复、GPU 吞吐、显存或端到端训练运行。未联网、未运行训练、未修改文件。


## 处理（Claude，2026-11-05）
1. **C2 接受被截掉的文档作为来源：属实，已修。** 改为按最终使用的字符预算计算本步真正出现的文档（`context_policies.visible_passage_ids`，flow 用 `_prompt_ids` 返回的预算），新增单测 `test_visible_ids_follow_truncation`。实际影响有限：只有在文档段超出预算被截断时才会触发（单步 prompt 上限 8192 token 对应约 24k 字符），但会破坏"笔记必有可见来源"这一约束，所以修。
2. **单引号形式的 note 调用统计为 0：属实，已修。** `SEARCH_RE` / `NOTE_RE` 同时接受单、双引号。核对 D2 三个 probe 的落盘：单引号形式的 search 调用为 0 次，D2 的检索轮数结论不受影响。
- "未核实"项（LoRA checkpoint 读取与续训、吞吐、显存、端到端训练）正是 `scripts/autodl/d3_run.sh` 要在实例上验证的内容，不另作处理。
