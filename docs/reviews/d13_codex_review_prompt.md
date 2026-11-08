你是只读审查者（D13 报告定稿审查节点）。仓库是当前目录 stateful-search-rl。请审查三份新文档：docs/REPORT.md、README.md、docs/resume_bullets.md。

审查重点（按重要性）：
1. 数字是否都能在仓库里找到出处并且一致：对照 docs/experiment_cards/e0_e3_main.md、d9_sealed_eval.md（含末尾「补充分析」）、d10_rq2_note_quality.md、d2_logprob_consistency.md、d2_probe_c0_4b.md、docs/analysis/d10_analysis.md。列出任何对不上的数字（写明文档、原文、应为多少、出处）。
2. 过度表述：结论是否超出证据（单 seed、HotpotQA 差异不显著、dev 复测偏低 1.8pp、judge 局限、分层用的是 C0 步数近似等）。简历 bullet 是否有面试会被追问倒的说法。
3. 与红线冲突：任何数字不是来自 runs/ 日志或实验卡、预填或「预期结果」写成结果。
4. 明显遗漏：读者理解结论必需、但文档里没写的限制或口径。

不要改任何文件。输出中文，按严重程度排序，每条给：位置、问题、建议改法。没有问题的方面不用写。控制在 800 字以内。
