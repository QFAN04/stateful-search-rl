"""★ C0–C3 上下文策略：每一步 user prompt 怎么渲染（纯 Python，不依赖 torch / veRL，可在本地单测）。

四组共用同一个外置状态 `TrajState`，区别只在 `render()` 把状态的哪些部分放进 prompt：

| 策略 | History Actions | 额外段落 | Retrieved Passages |
|---|---|---|---|
| c0 append | 全部 query | 无 | 全部历史文档（去重累积）——与 recipe 原版逐字相同 |
| c1 window | 最近 k 轮的 query | 无 | 最近 k 轮的文档 |
| c2 curate | 全部 query | `### Notes`：模型用 `note` 工具写的笔记（累积） | 仅最近一轮的文档 |
| c3 rule | 全部 query | `### Key Sentences`：从更早的文档里按词重叠规则抽出的句子 | 仅最近一轮的文档 |

约定：
- 一"轮"(round) = 一次检索动作（自动首搜算第 0 轮；同一步里的多个并行 query 算同一轮）。
- 文档编号 = 在去重累积列表里的位置（从 1 开始），四组一致；C2 笔记引用的就是这个编号。
- 模型写的 `<analysis>` 不带到下一步（与 recipe 一致）。
- 额外段落放在 History Actions 之后，保证 `### User Query\n...\n\n### History` 结构不变（analyze_probe 依赖它）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from recipes.hotpotqa.prompts import HOTPOTQA_TOOL_SCHEMAS, HOTPOTQA_USER_PROMPT

POLICIES = ("c0", "c1", "c2", "c3")

NOTE_MAX_CHARS = 300  # 单条笔记最长字符数，超出截断
NOTES_PER_STEP = 3  # 每步最多接受几条 note 调用
SENTENCE_MAX_CHARS = 300  # C3 单句最长字符数

# ---------------------------------------------------------------------------
# 外置状态
# ---------------------------------------------------------------------------


@dataclass
class Note:
    text: str
    source: int  # 引用的文档编号（1 起）


@dataclass
class TrajState:
    question: str
    # 与 recipe 相同：(首次检索到它的 query, 文本)，按文本去重累积
    passages: list[tuple[str, str]] = field(default_factory=list)
    # 与 recipe 相同：所有发出的 query（含失败的），按顺序
    history_actions: list[str] = field(default_factory=list)
    # 每轮检索：该轮的 query 列表、该轮检索到的文档编号（0 起，轮内去重，保持顺序；含之前已见过的文档）
    round_queries: list[list[str]] = field(default_factory=list)
    round_passages: list[list[int]] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)

    def add_round(self, queries: list[str], texts_per_query: list[list[str]]) -> None:
        """记录一轮检索。texts_per_query[i] 是第 i 个 query 解析出的文档文本（失败为空列表）。"""
        assert len(queries) == len(texts_per_query)
        ids: list[int] = []
        for q, texts in zip(queries, texts_per_query):
            for t in texts:
                idx = self._index_of(t)
                if idx is None:
                    self.passages.append((q, t))
                    idx = len(self.passages) - 1
                if idx not in ids:
                    ids.append(idx)
            self.history_actions.append(q)
        self.round_queries.append(list(queries))
        self.round_passages.append(ids)

    def _index_of(self, text: str) -> int | None:
        for i, (_, t) in enumerate(self.passages):
            if t == text:
                return i
        return None

    def snapshot(self) -> dict:
        return {
            "n_passages": len(self.passages),
            "n_rounds": len(self.round_passages),
            "round_passages": [[i + 1 for i in r] for r in self.round_passages],
            "notes": [{"text": n.text, "source": n.source} for n in self.notes],
        }


# ---------------------------------------------------------------------------
# 格式化（c0 部分与 recipe hotpotqa_agent_flow.py 的 _format_passage_list / _format_history_actions 逐字一致）
# ---------------------------------------------------------------------------


def _passage_lines(entries: list[tuple[int, str, str]], max_chars: int) -> tuple[list[str], int]:
    """返回 (输出行, 实际展示的条数)。max_chars>0 时超出部分截断。"""
    lines: list[str] = []
    total = 0
    for i, (label, query, text) in enumerate(entries, start=1):
        snippet = text[:1200].replace("\n", " ")
        line = f"[{label}] (query: {query}) {snippet}"
        if max_chars > 0 and total + len(line) > max_chars:
            lines.append(f"... ({len(entries) - i + 1} more passages truncated)")
            return lines, i - 1
        lines.append(line)
        total += len(line)
    return lines, len(entries)


def format_passages(entries: list[tuple[int, str, str]], max_chars: int = 0) -> str:
    """entries: (编号, query, 文本)。"""
    if not entries:
        return "None"
    return "\n".join(_passage_lines(entries, max_chars)[0])


def format_actions(actions: list[str]) -> str:
    if not actions:
        return "None"
    return "\n".join(f"[Search] {q}" for q in actions)


def _entries(state: TrajState, ids: list[int]) -> list[tuple[int, str, str]]:
    return [(i + 1, state.passages[i][0], state.passages[i][1]) for i in ids]


# ---------------------------------------------------------------------------
# C3：规则抽取
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or but is are was were be been being as that this these "
    "those which who whom whose what when where why how it its he she they his her their them there than then "
    "do does did not no so if into about also has have had can could would should will may might one two".split()
)
_WORD = re.compile(r"[a-z0-9]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS}


def extract_key_sentences(
    state: TrajState, exclude: set[int], max_sentences: int, min_overlap: int = 2
) -> list[tuple[int, str]]:
    """从未被 exclude 的文档中，按与"问题 + 全部历史 query"的内容词重叠数选句子。

    返回 [(文档编号 1 起, 句子)]，按出现顺序排列。确定性：同分时取更早出现的。
    """
    target = _content_words(state.question)
    for q in state.history_actions:
        target |= _content_words(q)
    cands: list[tuple[int, int, int, str]] = []  # (-score, 出现顺序, 文档编号, 句子)
    order = 0
    for idx, (_, text) in enumerate(state.passages):
        if idx in exclude:
            continue
        for sent in _SENT_SPLIT.split(text.replace("\n", " ")):
            sent = sent.strip()
            if not sent:
                continue
            score = len(_content_words(sent) & target)
            if score >= min_overlap:
                cands.append((-score, order, idx + 1, sent[:SENTENCE_MAX_CHARS]))
            order += 1
    top = sorted(cands)[:max_sentences]
    top.sort(key=lambda c: c[1])
    return [(label, sent) for _, _, label, sent in top]


# ---------------------------------------------------------------------------
# 模板与工具
# ---------------------------------------------------------------------------

_HIST = "### History Actions\n{history_actions}\n\n"
assert _HIST in HOTPOTQA_USER_PROMPT

C2_INSTRUCTION = (
    "**Only passages from the latest search are shown; passages from earlier searches are not shown again.** "
    "To keep evidence for later steps, call the `note` tool with a short fact and the id of the passage it "
    "comes from (at most 3 notes per step). Saved notes appear under **Notes**.\n"
)
C2_USER_PROMPT = HOTPOTQA_USER_PROMPT.replace(_HIST, _HIST + "### Notes\n{extra}\n\n").replace(
    "### Output Format", C2_INSTRUCTION + "\n### Output Format"
)
C3_USER_PROMPT = HOTPOTQA_USER_PROMPT.replace(
    _HIST, _HIST + "### Key Sentences (extracted from earlier passages)\n{extra}\n\n"
)

NOTE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "note",
        "description": (
            "Save one short evidence note for later steps. Earlier passages will not be shown again, "
            "so record the facts you need to answer the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "One short fact copied or paraphrased from a passage."},
                "source": {"type": "integer", "description": "Id of the passage the fact comes from, e.g. 3 for [3]."},
            },
            "required": ["fact", "source"],
        },
    },
}


def tool_schemas(policy: str) -> list[dict]:
    return HOTPOTQA_TOOL_SCHEMAS + [NOTE_TOOL_SCHEMA] if policy == "c2" else list(HOTPOTQA_TOOL_SCHEMAS)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConfig:
    policy: str = "c0"
    window_k: int = 1  # C1 保留最近几轮
    c3_max_sentences: int = 6  # C3 最多抽几句

    def __post_init__(self):
        assert self.policy in POLICIES, self.policy
        assert self.window_k >= 1 and self.c3_max_sentences >= 1


def shown_passage_ids(cfg: PolicyConfig, state: TrajState) -> list[int]:
    """本步 prompt 中 Retrieved Passages 段展示的文档（0 起编号）。"""
    if cfg.policy == "c0":
        return list(range(len(state.passages)))
    if cfg.policy == "c1":
        ids: list[int] = []
        for r in state.round_passages[-cfg.window_k :]:
            ids += [i for i in r if i not in ids]
        return ids
    return list(state.round_passages[-1]) if state.round_passages else []


def visible_passage_ids(cfg: PolicyConfig, state: TrajState, max_chars: int) -> list[int]:
    """按与 render 相同的 max_chars 截断后，Retrieved Passages 段里真正出现的文档（0 起编号）。
    C2 的 note 只能引用这些文档。"""
    shown = shown_passage_ids(cfg, state)
    return shown[: _passage_lines(_entries(state, shown), max_chars)[1]]


def render(cfg: PolicyConfig, state: TrajState, tool_feedback: str, max_chars: int) -> str:
    """返回本步的 user 消息内容。max_chars 只截断 Retrieved Passages 段（与 recipe 的预算收缩一致）。"""
    fb = tool_feedback.strip() if tool_feedback.strip() else "None"
    shown = shown_passage_ids(cfg, state)
    passages = format_passages(_entries(state, shown), max_chars=max_chars)
    if cfg.policy == "c1":
        actions = [q for qs in state.round_queries[-cfg.window_k :] for q in qs]
    else:
        actions = state.history_actions
    common = dict(
        user_query=state.question,
        history_actions=format_actions(actions),
        passage_list=passages,
        tool_feedback=fb,
    )
    if cfg.policy in ("c0", "c1"):
        return HOTPOTQA_USER_PROMPT.format(**common)
    if cfg.policy == "c2":
        extra = "\n".join(f"- {n.text} (source: [{n.source}])" for n in state.notes) or "None"
        return C2_USER_PROMPT.format(extra=extra, **common)
    sents = extract_key_sentences(state, exclude=set(shown), max_sentences=cfg.c3_max_sentences)
    extra = "\n".join(f"- {s} (source: [{label}])" for label, s in sents) or "None"
    return C3_USER_PROMPT.format(extra=extra, **common)


# ---------------------------------------------------------------------------
# C2 笔记动作
# ---------------------------------------------------------------------------

_INT = re.compile(r"\d+")


def parse_note(arguments: str | dict, allowed_sources: set[int]) -> tuple[Note | None, str]:
    """解析一次 note 调用。返回 (Note 或 None, 错误原因)。allowed_sources 为本步可引用的编号（1 起）。"""
    args = arguments
    if isinstance(args, str):
        try:
            args = json.loads(args)
            if isinstance(args, str):
                args = json.loads(args)
        except Exception:
            return None, "arguments are not valid JSON"
    if not isinstance(args, dict):
        return None, "arguments must be a JSON object"
    fact = str(args.get("fact", "")).strip()
    if not fact:
        return None, "fact is empty"
    src = args.get("source")
    if isinstance(src, bool) or src is None:
        return None, "source is missing"
    if isinstance(src, int):
        source = src
    else:
        m = _INT.search(str(src))
        if not m:
            return None, "source must be a passage id"
        source = int(m.group())
    if source not in allowed_sources:
        return None, f"source [{source}] is not one of the passages shown in this step"
    return Note(text=fact[:NOTE_MAX_CHARS].replace("\n", " "), source=source), ""
