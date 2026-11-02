"""P8 の関門。項目と選択中の項目が読める。確認の窓が重なった画面で窓の中を答える。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import synth
from harness import Sim
from sotozuke.costs import CostLog
from sotozuke.game_config import load_config
from sotozuke.reader import ClaudeReader

ROOT = Path(__file__).resolve().parent.parent
MENU_HINT = "上下左右 選択  決定  戻る"
MAP_HINT = "上下左右 選択  決定  戻る  設定"
WALK_HINT = "前後左右 移動  話す  地図  設定"
ITEMS = ["セーブ", "ロード", "設定", "タイトルへ戻る"]


def menu_view(selected: int) -> synth.View:
    return synth.View(scene=1, hint=MENU_HINT, menu_items=ITEMS, menu_title="メニュー", selected=selected)


def sim_for(game_path, style, **kwargs) -> Sim:
    return Sim(style, load_config(game_path), **kwargs)


def test_menu_first_entry_reads_title_and_selection(game_path, style):
    sim = sim_for(game_path, style)
    sim.view = menu_view(0)
    sim.run(2.0)
    assert sim.said("screen") == ["メニュー。項目1"]


def test_menu_move_reads_only_selection(game_path, style):
    sim = sim_for(game_path, style)
    sim.view = menu_view(0)
    sim.run(2.0)
    sim.press("cursor", "down")
    sim.view = menu_view(1)
    sim.run(2.0)
    assert sim.said("screen")[-1] == "項目2"


def test_menu_screen_change_without_key_is_not_read(game_path, style):
    """利用者が動かしていない変化は読まない。"""
    sim = sim_for(game_path, style)
    sim.view = menu_view(0)
    sim.run(2.0)
    calls = sim.reader.other_calls
    sim.view = menu_view(1)
    sim.run(2.0)
    assert sim.reader.other_calls == calls


def test_two_quick_moves_read_the_final_item(game_path, style):
    """続けて動かしたら、最後の項目を読む。古い項目で止まらない。"""
    sim = sim_for(game_path, style)
    sim.view = menu_view(0)
    sim.run(2.0)
    sim.press("cursor", "down")
    sim.view = menu_view(1)
    sim.run(0.4)
    sim.press("cursor", "down")
    sim.view = menu_view(2)
    sim.run(2.5)
    assert sim.reader.other_calls >= 3
    assert sim.said("screen")[-1] == f"項目{sim.reader.other_calls}"


def test_menu_after_dawn_discards_the_description(game_path, style):
    """暗転のあとに出たのがメニューなら、説明を読まず合図も鳴らさない。"""
    sim = sim_for(game_path, style, read_delay=1.5)
    sim.view = synth.View(scene=1, dark=1.0)
    sim.run(1.0)
    sim.view = synth.View(scene=2)
    sim.run(0.6)
    assert sim.reader.describe_calls == 1
    sim.view = menu_view(0)
    sim.run(3.0)
    assert sim.said("description") == []
    assert sim.events("tone") == []


def test_ignored_text_is_not_read(style, tmp_path):
    import json
    path = synth.write_game(style, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["text"]["ignore"] = ["読まない字。"]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    sim = Sim(style, load_config(path))
    sim.view = synth.View(scene=1, text="読まない字。")
    sim.run(2.0)
    assert sim.said() == []


def test_revisited_menu_item_uses_memory(game_path, style, monkeypatch):
    monkeypatch.setattr(synth, "MOTION", 0.0)
    sim = sim_for(game_path, style)
    sim.view = menu_view(0)
    sim.run(2.0)
    sim.press("cursor", "down")
    sim.view = menu_view(1)
    sim.run(2.0)
    sim.press("cursor", "up")
    sim.view = menu_view(0)
    sim.run(2.0)
    sim.press("cursor", "down")
    sim.view = menu_view(1)
    sim.run(2.0)
    assert sim.events("screen_cached")


def test_confirm_window_is_seen_over_menu(game_path, style):
    sim = sim_for(game_path, style)
    sim.view = menu_view(0)
    sim.run(2.0)
    view = menu_view(0)
    view.confirm = "この場所にデータをセーブしますか？"
    sim.view = view
    sim.run(2.0)
    assert sim.watcher.kind == "confirm"
    assert sim.said("screen")[-1] == "確認の文。はい"


def test_map_is_described(game_path, style):
    sim = sim_for(game_path, style)
    sim.view = synth.View(scene=1, hint=MAP_HINT)
    sim.run(2.0)
    assert sim.said("screen") == ["地図の説明"]


def test_walk_screen_says_where_and_what(game_path, style):
    sim = sim_for(game_path, style)
    sim.view = synth.View(scene=1, hint=WALK_HINT)
    sim.run(2.0)
    assert sim.said("screen") == ["walkの字。歩く画面です。"]


def test_walk_screen_not_treated_as_scene_change(game_path, style):
    """歩く画面も暗転の判定に入る。種類の判定を先にする。"""
    sim = sim_for(game_path, style)
    sim.view = synth.View(scene=1, dark=1.0)
    sim.run(1.0)
    sim.view = synth.View(scene=2, hint=WALK_HINT)
    sim.run(2.0)
    assert sim.reader.describe_calls == 0


def test_backlog_screen_tag_is_ignored(game_path, style):
    """札の線は履歴の画面でも出る。文の画面でだけ札を見る。"""
    sim = sim_for(game_path, style)
    sim.view = synth.View(scene=1, text="履歴の中の文。", name="名前", hint="上下 送り  再生  閉じる")
    sim.run(2.0)
    assert sim.watcher.kind == "backlog"
    assert not [r for r in sim.events("text_ready")]


def test_repeat_on_menu_rereads_selection(game_path, style):
    sim = sim_for(game_path, style)
    sim.view = menu_view(0)
    sim.run(2.0)
    before = len(sim.said("screen"))
    sim.press("repeat", "f6")
    assert len(sim.said("screen")) == before + 1
    assert sim.said("screen")[-1] == "メニュー。項目1"


def test_sentence_after_menu_is_not_repeated(game_path, style):
    """メニューから戻っても、同じ文を読み直さない。"""
    sim = sim_for(game_path, style)
    sim.view = synth.View(scene=1, text="メニューの前の文です。")
    sim.run(3.0)
    assert len(sim.said("sentence")) == 1
    sim.view = menu_view(0)
    sim.run(2.0)
    sim.view = synth.View(scene=1, text="メニューの前の文です。")
    sim.run(3.0)
    assert len(sim.said("sentence")) == 1


def read_real_menu(style, selected):
    reader = ClaudeReader(CostLog(ROOT / "costs.jsonl"))
    return reader.read_menu(synth.render(style, menu_view(selected)))[0]


@pytest.mark.api
@pytest.mark.parametrize("selected", [0, 2])
def test_gate_menu_items_and_selection(style, selected):
    assert os.environ.get("ANTHROPIC_API_KEY_FILE")
    data = read_real_menu(style, selected)
    print(data)
    assert data["items"] == ITEMS
    assert data["selected"] == ITEMS[selected]
    assert data["title"] == "メニュー"


@pytest.mark.api
@pytest.mark.parametrize("selected,button", [(0, "はい"), (1, "いいえ")])
def test_gate_confirm_window_contents(style, selected, button):
    reader = ClaudeReader(CostLog(ROOT / "costs.jsonl"))
    view = menu_view(0)
    view.confirm = "この場所にデータをセーブしますか？"
    view.selected = selected
    data = reader.read_confirm(synth.render(style, view))[0]
    print(data)
    assert data["message"] == "この場所にデータをセーブしますか？"
    assert data["selected"] == button


@pytest.mark.api
def test_gate_map_answer_is_short():
    reader = ClaudeReader(CostLog(ROOT / "costs.jsonl"))
    frame = synth.render(synth.STYLE_A, synth.View(scene=4))
    result = reader.read_map(frame, moved=False)
    print(len(result.text), result.text)
    assert 0 < len(result.text) <= 140
    assert isinstance(np.zeros(1), np.ndarray)
