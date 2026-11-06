"""G1b フェイズ A の関門。読む側の直し。乗せると暗くなる選択肢、場面の見分けが無いゲームの説明、
文の終わりの印、暗転を通らない場面の切り替わり。どれも合成の画面で確かめる。"""
from __future__ import annotations

import json

import pytest

import synth
from harness import Sim
from sotozuke.director import selected_row, strip_end_mark
from sotozuke.game_config import ConfigError, load_config

HINT = "上下 選択  決定  履歴  設定"
MARGIN = 40.0   # 合成の画面では、乗せた行は約120暗い
TEXT_DIFF = 20.0  # 合成の画面では、同じ場面の差は8未満、別の場面は30より大きい


def edit_game(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return load_config(path)


def dark_choice(data):
    data["choice"]["selected"] = "dark"
    data["choice"]["dark_margin"] = MARGIN


def choice_view(scene, items, selected):
    return synth.View(scene=scene, choices=list(items), selected=selected, hint=HINT, choice_dark=True)


# 2 選ばれている行の暗い行


def test_gate_dark_row_on_all_positions(game_path, style):
    config = edit_game(game_path, dark_choice)
    rule = config.choice
    for count in (2, 3, 4, 5):
        rows = rule.rows_for(count)
        items = [f"項目{n}" for n in range(count)]
        for chosen in range(count):
            frame = synth.render(style, choice_view(3, items, chosen))
            assert selected_row(frame, rows, rule.selected, rule.dark_margin, previous=0) == chosen
        # マウスが行から外れると、どの行も同じ明るさ。前の行のまま
        frame = synth.render(style, choice_view(3, items, -1))
        assert selected_row(frame, rows, rule.selected, rule.dark_margin, previous=count - 1) == count - 1


def test_gate_dark_choice_keeps_row_and_starts_from_first(game_path, style):
    sim = Sim(style, edit_game(game_path, dark_choice))
    items = ["一つ目の項目", "二つ目の項目", "三つ目の項目"]
    sim.view = choice_view(1, items, -1)
    sim.run(3.0)
    assert sim.events("choice_ready")[0]["selected"] is None  # どの行も暗くなければ選ばれていない（11-07 に改めた）
    sim.view = choice_view(1, items, 2)
    sim.run(0.5)
    assert sim.said("choice")[-1] == "三つ目の項目"
    sim.view = choice_view(1, items, -1)  # 外れても読み直さない
    sim.run(0.5)
    assert sim.said("choice")[-1] == "三つ目の項目"
    assert len(sim.events("choice_move")) == 1
    # 次の選択肢も選び直す。前の選択肢の行を持ち越さない
    sim.view = synth.View(scene=1, text="間の文です。")
    sim.run(2.0)
    sim.view = choice_view(1, ["別の一つ目", "別の二つ目"], -1)
    sim.run(3.0)
    assert sim.events("choice_ready")[-1]["selected"] is None


def test_dark_needs_margin(game_path):
    def no_margin(data):
        data["choice"]["selected"] = "dark"
    with pytest.raises(ConfigError):
        edit_game(game_path, no_margin)
    with pytest.raises(ConfigError):
        edit_game(game_path, lambda d: d["choice"].update(selected="dim"))


# 3 場面の見分けが無いゲームの F7


def test_gate_describe_reads_the_screen_now_without_scene_rule(game_path, style):
    sim = Sim(style, edit_game(game_path, lambda d: d.pop("scene")))
    sim.view = synth.View(scene=1, text="一つ目の場面の文です。")
    sim.run(2.0)
    sim.press("describe", "f7")
    sim.run(2.0)
    sim.view = synth.View(scene=2, text="二つ目の場面の文です。")
    sim.run(2.0)
    sim.press("describe", "f7")
    sim.run(2.0)
    assert sim.reader.describe_calls == 2
    assert sim.said("description") == ["説明1", "説明2"]


def test_describe_with_scene_rule_repeats_as_before(game_path, style):
    sim = Sim(style, load_config(game_path))
    sim.view = synth.View(scene=1, dark=1.0)
    sim.run(1.0)
    sim.view = synth.View(scene=1, text="文です。")
    sim.run(4.0)
    sim.press("describe", "f7")
    sim.run(2.0)
    assert sim.reader.describe_calls == 1
    assert sim.said("description") == ["説明1", "説明1"]


# 4 文の終わりの印


def test_gate_end_mark_is_removed_only_when_single():
    assert strip_end_mark("文です。▼", "▼・") == "文です。"
    assert strip_end_mark("文です。・ ", "▼・") == "文です。"
    assert strip_end_mark("え・・・", "▼・") == "え・・・"
    assert strip_end_mark("文です。▼", "") == "文です。▼"
    assert strip_end_mark("▼", "▼") == "▼"


def test_end_mark_is_not_spoken(game_path, style):
    sim = Sim(style, edit_game(game_path, lambda d: d["text"].update(end_marks="▼")))
    sim.reader.truth_text = lambda: sim.view.text + "▼"
    sim.view = synth.View(scene=1, text="印のある文です。")
    sim.run(2.0)
    assert sim.said("sentence") == ["印のある文です。"]


# 5 暗転を通らない場面の切り替わり


def no_dark_scene(data):
    data["scene"]["text_diff"] = TEXT_DIFF


def test_gate_scene_change_without_darkness(game_path, style):
    """場面3つを暗転なしで切り替える。同じ絵の文と、選択肢を挟んだあとでは説明しない。"""
    sim = Sim(style, edit_game(game_path, no_dark_scene))
    sim.view = synth.View(scene=1, text="一つ目の場面の文です。")
    sim.run(4.0)
    assert sim.said("description") == ["説明1"]
    assert sim.said("sentence") == ["一つ目の場面の文です。"]  # 説明、文の順
    sim.press()
    sim.view = synth.View(scene=1, text="同じ場面の次の文です。")
    sim.run(3.0)
    sim.press()
    sim.view = synth.View(scene=2, text="二つ目の場面の文です。")
    sim.run(4.0)
    sim.view = choice_view(2, ["選ぶ一つ目", "選ぶ二つ目"], 0)
    sim.view.choice_dark = False
    sim.run(3.0)
    sim.press()
    sim.view = synth.View(scene=2, text="選択肢のあとの文です。")
    sim.run(3.0)
    sim.press()
    sim.view = synth.View(scene=3, text="三つ目の場面の文です。")
    sim.run(4.0)
    assert sim.reader.describe_calls == 3
    assert sim.said("description") == ["説明1", "説明2", "説明3"]
    assert "選択肢のあとの文です。" in sim.said("sentence")


def test_scene_by_text_is_off_when_not_written(game_path, style):
    sim = Sim(style, load_config(game_path))
    sim.view = synth.View(scene=1, text="一つ目の文です。")
    sim.run(3.0)
    sim.press()
    sim.view = synth.View(scene=2, text="二つ目の文です。")
    sim.run(3.0)
    assert sim.reader.describe_calls == 0
    assert sim.events("scene_by_text") == []
