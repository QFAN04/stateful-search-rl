"""★ C0–C3 上下文构造的单测。

最重要的一条：C0 的渲染与 recipe 原版逐字一致（E0 = recipe 基线）。上游 flow 模块会 import transformers / verl，
本地装不了，所以用 ast 只取出上游的两个格式化函数源码来执行，与我们的实现逐字比较。
"""
import ast
import pathlib
import re

import pytest

from recipes.hotpotqa.prompts import HOTPOTQA_TOOL_SCHEMAS, HOTPOTQA_USER_PROMPT
from ssrl.context_policies import (
    NOTE_MAX_CHARS,
    PolicyConfig,
    TrajState,
    extract_key_sentences,
    parse_note,
    render,
    shown_passage_ids,
    tool_schemas,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
FLOW_SRC = ROOT / "third_party/Agent-R1/recipes/hotpotqa/hotpotqa_agent_flow.py"
QUERY_RE = re.compile(r"### User Query\n(.*?)\n\n### History", re.DOTALL)  # 与 scripts/analyze_probe.py 相同


def _upstream_funcs():
    tree = ast.parse(FLOW_SRC.read_text(encoding="utf-8"))
    wanted = {"_format_passage_list", "_format_history_actions"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    ns: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(FLOW_SRC), "exec"), ns)
    return ns["_format_passage_list"], ns["_format_history_actions"]


def _upstream_ingest(passages, query, texts):
    """recipe HotpotQAAgentFlow._ingest_results 的去重规则：文本相同就不再加入。"""
    for t in texts:
        if not any(existing == t for _, existing in passages):
            passages.append((query, t))


def _state() -> TrajState:
    """三轮检索：第 0 轮自动首搜；第 1 轮两个并行 query（含一篇重复文档）；第 2 轮一个 query。"""
    st = TrajState(question="Which band did the singer of Song X join in 1990?")
    st.add_round(["Which band did the singer of Song X join in 1990?"],
                 [["Song X is a song by Alice Smith. It was released in 1988.", "Unrelated passage about cats."]])
    st.add_round(["Alice Smith singer", "Song X 1988"],
                 [["Alice Smith is a singer. She joined the band Blue Moon in 1990.", "Unrelated passage about cats."],
                  ["Song X charted in 1988.\nIt reached number 3."]])
    st.add_round(["Blue Moon band"], [["Blue Moon is a rock band formed in 1985.", "Song X charted in 1988.\nIt reached number 3."]])
    return st


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------

def test_state_matches_recipe_dedup():
    st = _state()
    passages, actions = [], []
    for qs, texts_per_q in [
        (["Which band did the singer of Song X join in 1990?"],
         [["Song X is a song by Alice Smith. It was released in 1988.", "Unrelated passage about cats."]]),
        (["Alice Smith singer", "Song X 1988"],
         [["Alice Smith is a singer. She joined the band Blue Moon in 1990.", "Unrelated passage about cats."],
          ["Song X charted in 1988.\nIt reached number 3."]]),
        (["Blue Moon band"], [["Blue Moon is a rock band formed in 1985.", "Song X charted in 1988.\nIt reached number 3."]]),
    ]:
        for q, texts in zip(qs, texts_per_q):
            _upstream_ingest(passages, q, texts)
            actions.append(q)
    assert st.passages == passages
    assert st.history_actions == actions
    # 每轮记录检索到的文档（含已见过的），编号 0 起
    assert st.round_passages == [[0, 1], [2, 1, 3], [4, 3]]


def test_failed_search_still_records_query():
    st = TrajState(question="q")
    st.add_round(["q"], [[]])
    assert st.history_actions == ["q"] and st.passages == [] and st.round_passages == [[]]


# ---------------------------------------------------------------------------
# C0 = recipe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_chars", [0, 120, 24576])
@pytest.mark.parametrize("fb", ["", "  ", "some format issue"])
def test_c0_identical_to_recipe(max_chars, fb):
    fmt_passages, fmt_actions = _upstream_funcs()
    st = _state()
    expected = HOTPOTQA_USER_PROMPT.format(
        user_query=st.question,
        passage_list=fmt_passages(st.passages, max_chars=max_chars),
        history_actions=fmt_actions(st.history_actions),
        tool_feedback=fb.strip() if fb.strip() else "None",
    )
    assert render(PolicyConfig("c0"), st, fb, max_chars) == expected


def test_c0_empty_state_identical_to_recipe():
    fmt_passages, fmt_actions = _upstream_funcs()
    st = TrajState(question="q?")
    expected = HOTPOTQA_USER_PROMPT.format(
        user_query="q?", passage_list=fmt_passages([], 100), history_actions=fmt_actions([]), tool_feedback="None"
    )
    assert render(PolicyConfig("c0"), st, "", 100) == expected


# ---------------------------------------------------------------------------
# C1 / C2 / C3
# ---------------------------------------------------------------------------

def test_c1_window_shows_only_last_k_rounds():
    st = _state()
    out = render(PolicyConfig("c1", window_k=1), st, "", 0)
    assert "[Search] Blue Moon band" in out
    assert "[Search] Alice Smith singer" not in out
    assert "[5] (query: Blue Moon band) Blue Moon is a rock band" in out
    assert "[4] (query: Song X 1988) Song X charted in 1988. It reached number 3." in out  # 重复文档用原编号
    assert "Alice Smith is a singer" not in out and "released in 1988" not in out

    out2 = render(PolicyConfig("c1", window_k=2), st, "", 0)
    assert "[Search] Alice Smith singer" in out2 and "[Search] Song X 1988" in out2
    assert "[Search] Which band" not in out2
    assert shown_passage_ids(PolicyConfig("c1", window_k=2), st) == [2, 1, 3, 4]
    # k 大于轮数时等价于看到全部文档
    assert shown_passage_ids(PolicyConfig("c1", window_k=9), st) == shown_passage_ids(PolicyConfig("c0"), st)


def test_c2_latest_round_plus_notes():
    st = _state()
    cfg = PolicyConfig("c2")
    out = render(cfg, st, "", 0)
    assert "### Notes\nNone\n" in out
    for q in st.history_actions:  # 全部历史 query
        assert f"[Search] {q}" in out
    assert "Blue Moon is a rock band" in out and "Alice Smith is a singer" not in out
    assert "`note` tool" in out

    note, err = parse_note('{"fact": "Alice Smith joined Blue Moon in 1990", "source": 3}', {3, 5})
    assert err == "" and note.source == 3
    st.notes.append(note)
    out = render(cfg, st, "", 0)
    assert "### Notes\n- Alice Smith joined Blue Moon in 1990 (source: [3])\n" in out


def test_c3_key_sentences_from_earlier_passages_only():
    st = _state()
    cfg = PolicyConfig("c3", c3_max_sentences=2)
    current = set(shown_passage_ids(cfg, st))
    assert current == {4, 3}
    sents = extract_key_sentences(st, exclude=current, max_sentences=2)
    assert all(label - 1 not in current for label, _ in sents)
    assert (3, "She joined the band Blue Moon in 1990.") in sents  # 与问题 / query 重叠最多
    assert all("cats" not in s for _, s in sents)
    assert extract_key_sentences(st, current, 2) == sents  # 确定性
    out = render(cfg, st, "", 0)
    assert "### Key Sentences (extracted from earlier passages)\n- " in out
    assert "(source: [3])" in out
    assert "Alice Smith is a singer. She joined" not in out  # 早先文档本身不展示


@pytest.mark.parametrize("policy", ["c0", "c1", "c2", "c3"])
def test_common_structure(policy):
    st = _state()
    out = render(PolicyConfig(policy), st, "", 0)
    m = QUERY_RE.search(out)
    assert m and m.group(1) == st.question  # analyze_probe 能取出题目
    assert out.count("### Retrieved Passages\n") == 1
    assert "### Instructions" in out and "### Output Format" in out
    assert "<analysis>" in out  # 指令部分沿用 recipe


@pytest.mark.parametrize("policy", ["c1", "c2", "c3"])
def test_truncation_only_cuts_passages(policy):
    st = _state()
    full = render(PolicyConfig(policy), st, "", 0)
    short = render(PolicyConfig(policy), st, "", 60)
    assert "more passages truncated" in short and len(short) < len(full)
    assert short.split("### Retrieved Passages")[0] == full.split("### Retrieved Passages")[0]


def test_tool_schemas():
    assert tool_schemas("c0") == HOTPOTQA_TOOL_SCHEMAS
    assert [s["function"]["name"] for s in tool_schemas("c2")] == ["search", "note"]
    assert tool_schemas("c1") == tool_schemas("c3") == HOTPOTQA_TOOL_SCHEMAS


# ---------------------------------------------------------------------------
# note 解析
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args,ok", [
    ('{"fact": "x", "source": 2}', True),
    ('{"fact": "x", "source": "[2]"}', True),
    ({"fact": "x", "source": "2"}, True),
    ('"{\\"fact\\": \\"x\\", \\"source\\": 2}"', True),  # 双重编码
    ('{"fact": "x", "source": 9}', False),  # 本步没展示
    ('{"fact": "", "source": 2}', False),
    ('{"fact": "x"}', False),
    ('{"fact": "x", "source": true}', False),
    ('{"fact": "x", "source": "abc"}', False),
    ("not json", False),
    ("[1, 2]", False),
])
def test_parse_note(args, ok):
    note, err = parse_note(args, {1, 2})
    assert (note is not None) == ok and (err == "") == ok


def test_note_truncated():
    note, _ = parse_note({"fact": "a" * 1000 + "\nb", "source": 1}, {1})
    assert len(note.text) == NOTE_MAX_CHARS and "\n" not in note.text


def test_bad_policy():
    with pytest.raises(AssertionError):
        PolicyConfig("c9")
