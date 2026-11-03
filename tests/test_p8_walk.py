"""歩く画面の道案内の関門。地図で決めた人の向きと近さ、回した角度、進めないことを伝える。"""
from __future__ import annotations

import json

import numpy as np

import synth
from harness import Sim
from sotozuke.game_config import load_config
from sotozuke.inputs import EdgeDetector, KeyMapper
from sotozuke.walk import bearing_text, heading_text, target_text

MAP_HINT = "上下左右 選択  決定  戻る  設定"
WALK_HINT = "前後左右 移動  話す  地図  設定"
HINT = "Wで前へ。"
WALK = {"turn_deg_per_s": 90, "view_deg": 60, "tick_deg": 30, "bump_diff": 1.0, "hint": HINT}


def walk_sim(game_path, style, **walk) -> Sim:
    data = json.loads(game_path.read_text(encoding="utf-8"))
    data["walk"] = {**WALK, **walk}
    game_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return Sim(style, load_config(game_path))


def enter_walk_from_map(sim: Sim) -> None:
    sim.view = synth.View(scene=1, hint=MAP_HINT)
    sim.run(2.0)
    sim.press("advance")
    sim.view = synth.View(scene=2, hint=WALK_HINT)
    sim.run(2.0)


def tones(sim: Sim, name: str) -> int:
    return sum(1 for r in sim.events("tone") if r["name"] == name)


def test_words_for_direction_and_distance():
    assert bearing_text(50, 60) == "正面"
    assert bearing_text(80, 60) == "右へ20度"
    assert bearing_text(0, 60) == "左へ30度"
    assert bearing_text(70, 0) == "やや右"
    assert bearing_text(98, 0) == "右の端"
    assert target_text({"found": False}, 60) == "見えません。"
    assert target_text({"found": True, "x": 52, "distance": "すぐ前"}, 60) == "正面、すぐ前。話しかけられそうです。"
    assert target_text({"found": True, "x": "?", "distance": "変な答え"}, 60) == "正面。"
    assert heading_text(0) == "初めの向き"
    assert heading_text(-95) == "初めの向きから左へ90度"
    assert heading_text(350) == "初めの向き"
    assert heading_text(180) == "初めの向きの逆"


def test_walk_section_is_read_from_the_config(game_path, style):
    sim = walk_sim(game_path, style, move_keys=["w", "s"])
    assert sim.config.walk.turn_deg_per_s == 90
    assert sim.config.walk.move_keys == ("w", "s")


def test_release_is_reported_for_held_keys():
    mapper = KeyMapper(["enter"], ["up", "down"], move=["w"], hold=["left", "right"])
    assert mapper.on_press("w", 0).kind == "move"
    assert mapper.on_press("left", 0).kind == "cursor"
    assert mapper.on_release("w", 1).kind == "release"
    assert mapper.on_release("enter", 1) is None
    edges = EdgeDetector()
    edges.update({"w"})
    assert edges.update(set()) == [] and edges.released == ["w"]


def test_entry_says_the_hint_once_and_where_the_chosen_person_is(game_path, style):
    sim = walk_sim(game_path, style)
    enter_walk_from_map(sim)
    assert sim.said("screen")[-1] == f"walkの字。歩く画面です。{HINT}目当ての人は右へ20度、近い。"
    assert sim.reader.last_target == "金髪の女子"
    sim.view = synth.View(scene=3, text="文です。")
    sim.run(2.0)
    enter_walk_from_map(sim)
    assert sim.said("screen")[-1] == "walkの字。歩く画面です。目当ての人は右へ20度、近い。"


def test_without_a_map_choice_nothing_is_searched(game_path, style):
    sim = walk_sim(game_path, style)
    sim.view = synth.View(scene=2, hint=WALK_HINT)
    sim.run(2.0)
    assert sim.said("screen") == [f"walkの字。歩く画面です。{HINT}"]
    sim.press("describe")
    sim.run(2.0)
    assert sim.said("description")[-1].startswith("walk_viewの字")
    assert sim.reader.find_calls == 0


def test_after_the_keys_are_released_the_person_is_searched(game_path, style):
    sim = walk_sim(game_path, style)
    enter_walk_from_map(sim)
    before = sim.reader.find_calls
    sim.press("move", "w")
    sim.run(1.0)
    sim.press("release", "w")
    sim.run(0.3)
    assert sim.reader.find_calls == before  # 離してすぐは探さない
    sim.run(2.0)
    assert sim.reader.find_calls == before + 1
    assert sim.said("walk") == ["右へ20度、近い。"]


def test_a_new_key_drops_the_answer_for_the_old_view(game_path, style):
    sim = walk_sim(game_path, style)
    enter_walk_from_map(sim)
    sim.press("move", "w")
    sim.press("release", "w")
    sim.run(0.9)  # 探し始めた。答えは1秒後に届く
    sim.press("cursor", "right")
    sim.run(1.0)
    assert sim.said("walk") == []
    assert sim.events("drop_stale")[-1]["what"] == "walk_check"


def test_turning_ticks_and_the_heading_is_told(game_path, style):
    sim = walk_sim(game_path, style)
    enter_walk_from_map(sim)
    sim.press("cursor", "right")
    sim.run(1.1)
    sim.press("release", "right")
    assert tones(sim, "walk_tick") == 3
    turned = sim.events("walk_turn")[-1]
    assert 95 <= turned["degrees"] <= 104
    sim.run(0.3)
    sim.press("describe")
    sim.run(2.0)
    said = sim.said("description")[-1]
    assert said.startswith("目当ての人は右へ20度、近い。右に入口。")
    assert said.endswith("いまは初めの向きから右へ105度。")


def test_turning_left_counts_down(game_path, style):
    sim = walk_sim(game_path, style)
    enter_walk_from_map(sim)
    sim.press("cursor", "left")
    sim.run(0.55)
    sim.press("release", "left")
    assert sim.events("walk_turn")[-1]["heading"] < -40


def test_bump_when_walking_into_a_wall(game_path, style):
    sim = walk_sim(game_path, style, bump_diff=20)  # 合成の背景のゆっくりした動きは、止まっていると見る
    enter_walk_from_map(sim)
    sim.press("move", "w")
    sim.run(1.5)
    assert tones(sim, "walk_bump") == 1  # 止まっている間は1回だけ
    sim.press("release", "w")


def test_no_bump_while_the_view_moves(game_path, style):
    sim = walk_sim(game_path, style, bump_diff=20)
    enter_walk_from_map(sim)
    sim.press("move", "w")
    for i in range(15):
        sim.view = synth.View(scene=10 + i, hint=WALK_HINT)
        sim.run(0.1)
    sim.press("release", "w")
    assert tones(sim, "walk_bump") == 0


def test_not_found_is_said_briefly(game_path, style):
    sim = walk_sim(game_path, style)
    sim.reader.found = {"found": False}
    enter_walk_from_map(sim)
    assert sim.said("screen")[-1].endswith("目当ての人は見えません。")
    sim.press("cursor", "right")
    sim.run(0.3)
    sim.press("release", "right")
    sim.run(2.0)
    assert sim.said("walk") == ["見えません。"]


def test_keys_on_other_screens_do_not_start_a_search(game_path, style):
    sim = walk_sim(game_path, style)
    sim.view = synth.View(scene=1, text="文です。")
    sim.run(2.0)
    sim.press("move", "w")
    sim.press("release", "w")
    sim.run(2.0)
    assert sim.reader.find_calls == 0
    assert sim.said("walk") == []
