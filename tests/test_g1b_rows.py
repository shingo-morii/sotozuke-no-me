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


# F2 の関門


def dark_choice_game(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    data["input"] = {"choice": {"keys": False, "rows": "choice"}}
    data["choice"]["selected"] = "dark"
    data["choice"]["dark_margin"] = 40
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return load_config(path)


def dark_view(items, selected):
    return synth.View(scene=1, choices=items, selected=selected, hint=CHOICE_HINT, choice_dark=True)


def test_gate_keys_on_choice_are_not_overwritten_by_a_late_picture(game_path, style):
    """ゲームの光がキーから1秒遅れて移る。下、下、下、上、上、上で、乗せた行は1、2、3、2、1、1行目。"""
    config = dark_choice_game(game_path)
    sim = Sim(style, config)
    items = ["一つ目の項目", "二つ目の項目", "三つ目の項目"]
    rows = config.choice.rows_for(3)
    centers = [((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in rows]
    sim.view = dark_view(items, -1)
    sim.run(3.0)
    for key, row in (("down", 0), ("down", 1), ("down", 2), ("up", 1), ("up", 0), ("up", 0)):
        before = len(sim.said("choice"))
        sim.press("cursor", key)
        assert sim.sent[-1].at == centers[row], (key, row)
        sim.run(1.0)  # ゲームはまだ前の行を暗くしている
        sim.view = dark_view(items, row)
        sim.run(1.0)
        assert sim.said("choice")[before:] == [items[row]], (key, row)
    assert all(e.get("by") == "key" for e in sim.events("choice_move"))
    # 合ったあとは、利用者がマウスで動かした行を明るさで追う
    sim.view = dark_view(items, 2)
    sim.run(1.0)
    assert sim.said("choice")[-1] == items[2]


def test_key_row_stays_when_the_picture_never_follows(game_path, style):
    sim = Sim(style, dark_choice_game(game_path))
    items = ["一つ目の項目", "二つ目の項目", "三つ目の項目"]
    sim.view = dark_view(items, 2)  # ゲームは3行目を暗くしたまま動かない
    sim.run(3.0)
    for key in ("down", "down"):
        sim.press("cursor", key)
        sim.run(2.0)
    assert sim.said("choice")[-1] == items[1]
    assert sim.director.row_index == 1


def test_gate_buttons_without_glyph_points_get_different_keys():
    """字と地の明るさが近い同じ大きさの2つのボタン。字の点は取れないが、別のものとして覚える。"""
    import hashlib

    import numpy as np
    from PIL import Image, ImageDraw

    from sotozuke.director import row_key
    from sotozuke.game_config import GlyphRule
    from sotozuke.image_ops import glyph_mask, to_gray

    def button(text):
        image = Image.new("RGB", (120, 40), (150, 150, 160))
        ImageDraw.Draw(image).text((60, 20), text, font=synth.font(22), fill=(175, 175, 185), anchor="mm")
        return np.asarray(image)

    ok, cancel = button("OK"), button("Cancel")
    box = (10, 10, 130, 50)
    old = [hashlib.md5(np.packbits(glyph_mask(to_gray(b), GlyphRule())).tobytes()).hexdigest() for b in (ok, cancel)]
    assert old[0] == old[1]  # 前の鍵では同じになった
    assert row_key(ok, box) != row_key(cancel, box)
    assert row_key(ok, box) == row_key(ok.copy(), box)  # まったく同じ絵は覚えを使う
    assert row_key(ok, box) != row_key(ok, (20, 10, 140, 50))  # 箱が違えば別


def test_gate_ok_and_cancel_are_read_separately(game_path, style):
    data = json.loads(game_path.read_text(encoding="utf-8"))
    w = style.size[0]
    boxes = [[w // 4, 200 + i * 56 - 22, w * 3 // 4, 200 + i * 56 + 22] for i in range(2)]
    data["input"] = {"セーブ": {"keys": False, "rows": boxes, "pick": "f9"}}
    game_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    sim = Sim(style, load_config(game_path))
    calls = counting_reader(sim)
    view = lambda selected: synth.View(scene=1, hint=SAVE_HINT, menu_title="確認",  # noqa: E731
                                       menu_items=["Cancel", "OK"], selected=selected)
    sim.view = view(-1)
    sim.run(3.0)
    entered = len(sim.said("screen"))
    for key, row in (("down", 0), ("down", 1), ("up", 0), ("down", 1)):
        sim.press("cursor", key)
        sim.view = view(row)
        sim.run(2.0)
    assert sim.said("screen")[entered:] == ["箱1", "箱2", "箱1", "箱2"]
    assert len(calls) == 2
