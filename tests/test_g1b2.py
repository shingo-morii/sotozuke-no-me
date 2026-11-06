"""G1b その2 の関門。2本目と3本目の初見の取り込みで分かったことの直し（G1b_設計.md の「その2」）。

合成の画面だけを使う。2本目と3本目の値は使わない（知らない）。
"""
from __future__ import annotations

import json

import synth
from harness import Sim
from sotozuke.game_config import load_config

CHOICE_HINT = "上下 選択  決定  履歴  設定"
S = ["最初の文です。", "次の文が出ました。", "三つ目の文です。"]


def edit_game(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return load_config(path)


def choice_view(items, selected=0, dark=False):
    return synth.View(scene=1, choices=items, hint=CHOICE_HINT, selected=selected, choice_dark=dark)


# H1 選択肢


def only_two_rows(data):
    """行の位置を2項目の分だけにする。ほかの数は伸ばす元が無いので行が取れない。"""
    data["choice"]["rows_by_count"] = {"2": data["choice"]["rows_by_count"]["2"]}


def test_gate_choice_without_rows_does_not_stop(game_path, style):
    """項目に合う行の位置が無い選択肢で止まらず、何も言わず、失敗の音も鳴らさない。次の文は読む。"""
    sim = Sim(style, edit_game(game_path, only_two_rows))
    sim.view = synth.View(scene=1, text=S[0])
    sim.run(2.0)
    sim.view = choice_view(["一つ目", "二つ目", "三つ目"])
    sim.run(3.0)
    assert sim.events("choice_no_rows")
    assert sim.said("choice") == []
    assert [r for r in sim.events("tone") if r["name"] == "fail"] == []
    sim.press("cursor", "down")
    sim.run(1.0)
    sim.press("advance", "enter")
    sim.view = synth.View(scene=1, text=S[1])
    sim.run(3.0)
    assert sim.said("sentence")[-1] == S[1]


def test_gate_choice_with_no_items_is_quiet(game_path, style):
    sim = Sim(style, load_config(game_path))
    sim.reader.truth_items = lambda: []
    sim.view = choice_view(["一つ目", "二つ目"])
    sim.run(3.0)
    assert sim.events("choice_no_rows")
    assert sim.said("choice") == []
    assert [r for r in sim.events("tone") if r["name"] == "fail"] == []


def bright_margin(data):
    data["choice"]["bright_margin"] = 20


def dark_rows(data):
    data["choice"]["selected"] = "dark"
    data["choice"]["dark_margin"] = 40


def test_gate_unselected_bright_choice_says_no_row(game_path, style):
    """bright_margin を書いた明るい行の選択肢で、どの行も同じ明るさなら「選んでいるのは」を言わない。"""
    sim = Sim(style, edit_game(game_path, bright_margin))
    sim.view = choice_view(["一つ目", "二つ目"], selected=-1)
    sim.run(3.0)
    intro = sim.said("choice")
    assert intro == ["選択肢。一つ目。二つ目。"]
    assert sim.events("choice_ready")[-1]["selected"] is None


def test_gate_bright_margin_still_finds_a_lit_row(game_path, style):
    sim = Sim(style, edit_game(game_path, bright_margin))
    sim.view = choice_view(["一つ目", "二つ目"], selected=1)
    sim.run(3.0)
    assert sim.said("choice") == ["選択肢。一つ目。二つ目。選んでいるのは、二つ目。"]


def test_gate_unselected_dark_choice_says_no_row(game_path, style):
    sim = Sim(style, edit_game(game_path, dark_rows))
    sim.view = choice_view(["一つ目", "二つ目"], selected=-1, dark=True)
    sim.run(3.0)
    assert sim.said("choice") == ["選択肢。一つ目。二つ目。"]


def test_unselected_choice_then_key_reads_first_row(game_path, style):
    def change(data):
        bright_margin(data)
        data["input"] = {"choice": {"keys": False, "rows": "choice"}}

    sim = Sim(style, edit_game(game_path, change))
    sim.view = choice_view(["一つ目", "二つ目"], selected=-1)
    sim.run(3.0)
    sim.press("cursor", "down")
    sim.run(0.5)
    assert sim.said("choice")[-1] == "一つ目"


def test_gate_without_bright_margin_reads_top_row_as_before(game_path, style):
    """bright_margin を書かなければ今のまま。どの行も同じ明るさでも1位の行を読む。"""
    sim = Sim(style, load_config(game_path))
    sim.view = choice_view(["一つ目", "二つ目"], selected=-1)
    sim.run(3.0)
    assert sim.said("choice")[0].startswith("選択肢。一つ目。二つ目。選んでいるのは、")
