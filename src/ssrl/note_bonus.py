"""★ E4 笔记过程奖励（实验卡 docs/experiment_cards/e4_e5_candidates.md）。纯 Python，可在本地单测。

奖励 = EM + λ · bonus。bonus 只在以下条件同时满足时 > 0：
  1. 最终答对（EM = 1）；
  2. 检索 ≥ 2 轮（含自动首搜，即模型自己至少又检索了一轮）；
  3. 至少写过一条笔记（笔记只能在非终局步写，所以一定在作答之前）。
bonus = 笔记中「引用的文档属于该题 supporting fact 文档」的比例，取值 [0, 1]，因此每条轨迹的额外奖励最多 λ。
判断「属于」：引用文档的文本以某个 supporting fact 标题开头（检索语料每段是「标题 + 正文」，与 RQ2 的自动指标同一判定）。
"""
from __future__ import annotations

import json
from collections.abc import Sequence


def note_bonus(acc: float, n_rounds: int, note_sources: Sequence[int], passages: Sequence[str],
               sf_titles: Sequence[str]) -> float:
    """note_sources：每条笔记引用的文档编号（1 起，对应 passages 的位置）；passages：去重累积的文档文本。"""
    if acc < 1.0 or n_rounds < 2 or not note_sources or not sf_titles:
        return 0.0
    hits = 0
    for src in note_sources:
        if 1 <= src <= len(passages):
            text = passages[src - 1]
            if any(t and text.startswith(t) for t in sf_titles):
                hits += 1
    return hits / len(note_sources)


def load_sf_titles(path: str) -> dict[str, list[str]]:
    """scripts/build_supporting_facts.py 生成的 {问题文本: [supporting fact 标题]}。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
