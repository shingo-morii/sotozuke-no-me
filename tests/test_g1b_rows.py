"""G1b フェイズ F の関門。本体がマウスを乗せた行を、毎回読む。

- rows が箱の一覧の画面（セーブやロードの枠、確認の窓のボタン）は、乗せた行の箱を読み手で読む。同じ字の形の箱は覚えて読み直さない
- 選択肢は、最初の前後のキーで1行目に乗せ、すでに選ばれている行でも読む（test_g1b_input.py）
"""
from __future__ import annotations

import json
from pathlib import Path

import synth
from harness import FakeResult, Sim
from sotozuke.game_config import load_config

ROOT = Path(__file__).resolve().parent.parent
SAVE_HINT = "上下左右 選択  決定  戻る"
CHOICE_HINT = "上下 選択  決定  履歴  設定"
ITEMS = ["枠01 三月三日", "枠02 空き", "枠03 三月四日"]


def menu_rows(style):
    w = style.size[0]
    return [[w // 4, 200 + i * 56 - 22, w * 3 // 4, 200 + i * 56 + 22] for i in range(len(ITEMS))]


def box_game(path, style):
    data = json.loads(path.read_text(encoding="utf-8"))
    data["input"] = {"セーブ": {"keys": False, "rows": menu_rows(style), "pick": "f9"}}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return load_config(path)


def save_view(selected):
    return synth.View(scene=1, hint=SAVE_HINT, menu_title="セーブ", menu_items=ITEMS, selected=selected)


def counting_reader(sim):
    """箱を読むたびに番号の違う字を返す。覚えた箱を読み直すと番号が増えるので分かる。"""
    calls = []
    plain = sim.reader.read_plain

    def read_plain(frame, prompt, purpose):
        if purpose != "row":
            return plain(frame, prompt, purpose)
        calls.append(frame)
        return FakeResult(f"箱{len(calls)}")

    sim.reader.read_plain = read_plain
    return calls


def test_gate_box_rows_are_read_each_move_and_remembered(game_path, style):
    sim = Sim(style, box_game(game_path, style))
    calls = counting_reader(sim)
    sim.view = save_view(-1)
    sim.run(3.0)
    entered = len(sim.said("screen"))
    assert entered == 1  # 入った直後に画面を読む（前からの動き）
    for key, row in (("down", 0), ("down", 1), ("up", 0), ("down", 1), ("down", 2)):
        sim.press("cursor", key)
        sim.view = save_view(row)  # ゲームは乗せた枠を光らせる
        sim.run(2.0)
    assert sim.said("screen")[entered:] == ["箱1", "箱2", "箱1", "箱2", "箱3"]
    assert len(calls) == 3  # 同じ字の形の箱は読み直さない
    assert len(sim.events("row_cached")) == 2


def test_box_row_reading_is_dropped_when_moved_again(game_path, style):
    """続けて動かすと、読みかけの箱は読まない。最後に乗せた箱だけ読む。"""
    sim = Sim(style, box_game(game_path, style), read_delay=1.0)
    counting_reader(sim)
    sim.view = save_view(-1)
    sim.run(3.0)
    entered = len(sim.said("screen"))
    sim.press("cursor", "down")
    sim.view = save_view(0)
    sim.run(0.6)
    sim.press("cursor", "down")
    sim.view = save_view(1)
    sim.run(3.0)
    assert sim.said("screen")[entered:] == ["箱2"]


def test_empty_box_says_no_text(game_path, style):
    sim = Sim(style, box_game(game_path, style))
    sim.reader.read_plain = lambda frame, prompt, purpose: FakeResult("" if purpose == "row" else "画面")
    sim.view = save_view(-1)
    sim.run(3.0)
    sim.press("cursor", "down")
    sim.view = save_view(0)
    sim.run(2.0)
    assert sim.said("screen")[-1] == "字がありません。"


def test_pick_clicks_the_row_that_was_read(game_path, style):
    sim = Sim(style, box_game(game_path, style))
    counting_reader(sim)
    sim.view = save_view(-1)
    sim.run(3.0)
    for key in ("down", "down"):
        sim.press("cursor", key)
        sim.run(1.0)
    sim.press("key", "f9")
    rows = menu_rows(style)
    assert sim.sent[-1].do == "click"
    assert sim.sent[-1].at == ((rows[1][0] + rows[1][2]) // 2, (rows[1][1] + rows[1][3]) // 2)


def test_first_prev_key_on_choice_goes_to_the_first_row(game_path, style):
    data = json.loads(game_path.read_text(encoding="utf-8"))
    data["input"] = {"choice": {"keys": False, "rows": "choice"}}
    game_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    sim = Sim(style, load_config(game_path))
    items = ["一つ目の項目", "二つ目の項目", "三つ目の項目"]
    sim.view = synth.View(scene=1, choices=items, selected=0, hint=CHOICE_HINT)
    sim.run(3.0)
    sim.press("cursor", "up")
    assert sim.said("choice")[-1] == items[0]
    rows = load_config(game_path).choice.rows_for(3)
    assert sim.sent[-1].at == ((rows[0][0] + rows[0][2]) // 2, (rows[0][1] + rows[0][3]) // 2)


def test_gate_readme_and_skill_have_the_new_notes():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "通知を切る" in readme
    skill = (ROOT / ".claude" / "skills" / "add-game" / "SKILL.md").read_text(encoding="utf-8")
    assert "本体の記録に `key` が出るか" in skill
