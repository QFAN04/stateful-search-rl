"""E4 笔记过程奖励的条件与边界（src/ssrl/note_bonus.py）。"""
import json

import pytest

from ssrl.note_bonus import load_sf_titles, note_bonus

PASSAGES = [
    "Cecil Day-Lewis Cecil Day-Lewis was an Anglo-Irish poet and Poet Laureate.",
    "Tamasin Day-Lewis Tamasin Day-Lewis is an English television chef.",
    "Daniel Day-Lewis Daniel Day-Lewis is an English actor.",
]
SF = ["Cecil Day-Lewis", "Tamasin Day-Lewis"]


def test_full_bonus_when_all_notes_cite_supporting_docs():
    assert note_bonus(1.0, 2, [1, 2], PASSAGES, SF) == 1.0


def test_partial_bonus_is_fraction_of_notes():
    assert note_bonus(1.0, 3, [1, 3], PASSAGES, SF) == pytest.approx(0.5)
    assert note_bonus(1.0, 3, [3, 3, 1], PASSAGES, SF) == pytest.approx(1 / 3)


def test_no_bonus_when_wrong():
    assert note_bonus(0.0, 3, [1, 2], PASSAGES, SF) == 0.0


def test_no_bonus_with_single_round():
    # 只有自动首搜一轮：模型没有自己检索，不需要跨轮记忆
    assert note_bonus(1.0, 1, [1, 2], PASSAGES, SF) == 0.0


def test_no_bonus_without_notes():
    assert note_bonus(1.0, 3, [], PASSAGES, SF) == 0.0


def test_no_bonus_without_supporting_facts():
    assert note_bonus(1.0, 3, [1], PASSAGES, []) == 0.0


def test_out_of_range_source_counts_as_miss_not_error():
    assert note_bonus(1.0, 2, [1, 9, 0], PASSAGES, SF) == pytest.approx(1 / 3)


def test_match_is_title_prefix_not_substring():
    # 正文里提到 supporting fact 标题，但文档本身不是那篇：不算
    passages = ["Daniel Day-Lewis Daniel is the son of Cecil Day-Lewis."]
    assert note_bonus(1.0, 2, [1], passages, ["Cecil Day-Lewis"]) == 0.0


def test_bonus_bounded_in_unit_interval():
    for srcs in ([1], [3], [1, 2, 3], [2, 2, 2]):
        b = note_bonus(1.0, 2, srcs, PASSAGES, SF)
        assert 0.0 <= b <= 1.0


def test_load_sf_titles(tmp_path):
    p = tmp_path / "sf.json"
    p.write_text(json.dumps({"Q?": ["A", "B"]}), encoding="utf-8")
    assert load_sf_titles(str(p)) == {"Q?": ["A", "B"]}
