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


# H2 画面の種類


WINDOW_PATCH = (170, 690, 420, 706)  # 合成の型 b の文の枠の、字の下の帯


def absent_walk(tmp_path, style, frames=None):
    """文の枠が無いことで歩く画面と見る設定。枠の一部を見本にし、閾値は当たりと外れの間に置く。"""
    from PIL import Image
    from sotozuke.image_ops import crop, to_gray

    x0, y0, x1, y1 = WINDOW_PATCH
    with_window = [synth.render(style, synth.View(scene=s, text=S[0]), t) for s in (1, 2) for t in (0.0, 3.0)]
    without = [synth.render(style, synth.View(scene=s), t) for s in (1, 2) for t in (0.0, 3.0)]
    template = to_gray(crop(with_window[0], WINDOW_PATCH))
    Image.fromarray(template).save(tmp_path / "tpl_window.png")

    def score(frame):
        return 1.0 - float(abs(to_gray(crop(frame, WINDOW_PATCH)).astype(int) - template.astype(int)).mean()) / 255.0

    low, high = max(score(f) for f in without), min(score(f) for f in with_window)
    assert low < high
    path = synth.write_game(style, tmp_path)

    def change(data):
        data["screens"].append({"name": "外を歩く", "kind": "walk", "box": [x0, y0, x1, y1],
                                "template": "tpl_window.png", "threshold": (low + high) / 2, "match": "absent"})
        if frames:
            data["kind_frames"] = frames

    return edit_game(path, change)


def kinds(sim):
    return [r["new"] for r in sim.events("kind")]


def test_gate_absent_match_sees_walk_without_window(tmp_path):
    style = synth.STYLES["b"]
    sim = Sim(style, absent_walk(tmp_path, style))
    sim.view = synth.View(scene=1, text=S[0])
    sim.run(2.0)
    assert kinds(sim) == []
    assert sim.said("sentence") == [S[0]]
    sim.view = synth.View(scene=2)  # 文の枠の無い絵
    sim.run(3.0)
    assert kinds(sim) == ["walk"]
    assert any("歩く画面です" in t for t in sim.said())
    sim.view = synth.View(scene=2, text=S[1])  # 歩く画面で話しかけると文の枠が出る
    sim.run(3.0)
    assert kinds(sim) == ["walk", "text"]
    assert sim.said("sentence")[-1] == S[1]


def test_gate_absent_match_yields_to_other_templates(tmp_path):
    """ほかの見本が当たる画面は、文の枠が無くてもそちら。"""
    style = synth.STYLES["b"]
    sim = Sim(style, absent_walk(tmp_path, style))
    sim.view = synth.View(scene=1, hint="上下左右 選択  決定  戻る", menu_title="セーブ", menu_items=["枠01", "枠02"])
    sim.run(3.0)
    assert kinds(sim) == ["menu"]


def flicker(sim):
    """歩く画面の見分けが、ときどき3枚だけ外れる（その間は文の枠があるように見える）。"""
    sim.view = synth.View(scene=2)
    sim.run(3.0)
    for _ in range(6):
        sim.view = synth.View(scene=2, text=S[2])
        sim.run(0.3)
        sim.view = synth.View(scene=2)
        sim.run(2.5)


def test_gate_kind_frames_stops_repeated_walk_announcements(tmp_path):
    style = synth.STYLES["b"]
    sim = Sim(style, absent_walk(tmp_path, style, frames=5))
    flicker(sim)
    assert kinds(sim) == ["walk"]
    assert sum("歩く画面です" in t for t in sim.said()) == 1


def test_without_kind_frames_flicker_repeats(tmp_path):
    """書かないときは今のまま（2枚）。同じ揺れで何回も言う。直す前の動きを確かめる。"""
    style = synth.STYLES["b"]
    sim = Sim(style, absent_walk(tmp_path, style))
    flicker(sim)
    assert kinds(sim).count("walk") > 1
    assert sum("歩く画面です" in t for t in sim.said()) > 1


def test_kind_frames_must_be_positive(game_path):
    import pytest
    from sotozuke.game_config import ConfigError

    with pytest.raises(ConfigError):
        edit_game(game_path, lambda d: d.update(kind_frames=0))


# H3 歩く画面


WALK_HINT = "前後左右 移動  話す  地図  設定"
MAP_HINT = "上下左右 選択  決定  戻る  設定"


def record_reader(sim):
    """読み手に渡した頼みと、探す頼みの視点を残す。"""
    from harness import FakeResult

    asked = []
    plain = sim.reader.read_plain
    find = sim.reader.find_target

    def read_plain(frame, prompt, purpose):
        asked.append((purpose, prompt))
        return plain(frame, prompt, purpose)

    def find_target(frame, target, view="behind"):
        asked.append(("find", view))
        return find(frame, target)

    sim.reader.read_plain = read_plain
    sim.reader.find_target = find_target
    return asked


def walk_game(path, **walk):
    return edit_game(path, lambda d: d.update(walk={"turn_deg_per_s": 90, "view_deg": 60, **walk}))


def test_gate_first_person_describe_does_not_say_protagonist(game_path, style):
    sim = Sim(style, walk_game(game_path, view="first_person"))
    asked = record_reader(sim)
    sim.view = synth.View(scene=2, hint=WALK_HINT)
    sim.run(3.0)
    sim.press("describe", "f7")
    sim.run(2.0)
    prompts = [p for purpose, p in asked if purpose == "walk_view"]
    assert prompts and "主人公" not in prompts[-1] and "一人称" in prompts[-1]


def test_gate_first_person_search_passes_the_view(game_path, style):
    sim = Sim(style, walk_game(game_path, view="first_person"))
    asked = record_reader(sim)
    sim.view = synth.View(scene=1, hint=MAP_HINT)
    sim.run(2.0)
    sim.press("advance")
    sim.view = synth.View(scene=2, hint=WALK_HINT)
    sim.run(3.0)
    assert ("find", "first_person") in asked


def test_walk_target_prompt_by_view():
    from sotozuke import reader as prompts

    behind = prompts.walk_target_prompt("金髪の人", "behind")
    first = prompts.walk_target_prompt("金髪の人", "first_person")
    assert "主人公なので数えないで" in behind and "金髪の人" in behind
    assert "主人公" not in first and "一人称" in first and "金髪の人" in first
    assert "主人公" in prompts.walk_view_prompt("behind")
    assert prompts.walk_view_prompt("behind") == prompts.PROMPT_WALK_VIEW  # 書かないときは今の頼みのまま


def test_gate_default_view_keeps_the_old_prompt(game_path, style):
    sim = Sim(style, walk_game(game_path))
    asked = record_reader(sim)
    sim.view = synth.View(scene=2, hint=WALK_HINT)
    sim.run(3.0)
    sim.press("describe", "f7")
    sim.run(2.0)
    from sotozuke import reader as prompts
    assert [p for purpose, p in asked if purpose == "walk_view"][-1] == prompts.PROMPT_WALK_VIEW


def test_walk_view_must_be_known(game_path):
    import pytest
    from sotozuke.game_config import ConfigError

    with pytest.raises(ConfigError):
        walk_game(game_path, view="above")


def hint_game(path, style):
    """案内の字の範囲と点の数。取り込みのときと同じく、字の無い画面と字の出た画面で数えて間に置く。"""
    from sotozuke.image_ops import crop, glyph_mask, to_gray

    glyph = load_config(path).glyph

    def count(text, t):
        frame = synth.render(style, walk_with(text), t)
        return int(glyph_mask(to_gray(crop(frame, style.band)), glyph).sum())

    empty = max(count("", t) for t in (0.0, 2.0, 5.0))
    shown = min(count("決めるキーで話しかける", t) for t in (0.0, 2.0, 5.0))
    assert empty < shown

    def change(data):
        walk = next(s for s in data["screens"] if s["kind"] == "walk")
        walk["read_box"] = list(style.band)
        walk["read_min_glyphs"] = (empty + shown) // 2
    return edit_game(path, change)


def walk_with(text=""):
    return synth.View(scene=2, hint=WALK_HINT, text=text)


def test_gate_walk_hint_is_read_once_and_remembered(game_path, style):
    sim = Sim(style, hint_game(game_path, style))
    asked = record_reader(sim)
    sim.view = walk_with()
    sim.run(3.0)
    entered = len(sim.said())
    sim.view = walk_with("決めるキーで話しかける")
    sim.run(4.0)
    assert sim.said()[entered:] == ["walk_hintの字"]
    sim.view = walk_with()  # 消えて
    sim.run(2.0)
    sim.view = walk_with("決めるキーで話しかける")  # また出たら、覚えた字を言う
    sim.run(3.0)
    assert sim.said()[entered:] == ["walk_hintの字", "walk_hintの字"]
    assert sum(1 for purpose, _ in asked if purpose == "walk_hint") == 1
    sim.view = walk_with("右の扉は鍵がかかっている")  # 違う字は読む
    sim.run(3.0)
    assert sum(1 for purpose, _ in asked if purpose == "walk_hint") == 2


def test_walk_hint_flicker_is_not_said_again(game_path, style):
    """字が1コマ消えただけなら、もう一度言わない。"""
    sim = Sim(style, hint_game(game_path, style))
    sim.view = walk_with()
    sim.run(3.0)
    sim.view = walk_with("決めるキーで話しかける")
    sim.run(3.0)
    count = len(sim.said())
    sim.view = walk_with()
    sim.run(0.2)
    sim.view = walk_with("決めるキーで話しかける")
    sim.run(3.0)
    assert len(sim.said()) == count


def test_without_read_box_walk_hint_is_not_read(game_path, style):
    sim = Sim(style, load_config(game_path))
    asked = record_reader(sim)
    sim.view = walk_with()
    sim.run(3.0)
    sim.view = walk_with("決めるキーで話しかける")
    sim.run(4.0)
    assert not any(purpose == "walk_hint" for purpose, _ in asked)


def test_read_box_needs_min_glyphs_and_walk(game_path, style):
    import pytest
    from sotozuke.game_config import ConfigError

    def no_min(data):
        next(s for s in data["screens"] if s["kind"] == "walk")["read_box"] = list(style.band)

    def not_walk(data):
        menu = next(s for s in data["screens"] if s["kind"] == "menu")
        menu["read_box"], menu["read_min_glyphs"] = list(style.band), 40

    for change in (no_min, not_walk):
        with pytest.raises(ConfigError):
            edit_game(game_path, change)


def describe_twice(sim):
    sim.view = synth.View(scene=1, text=S[0])
    sim.run(2.0)
    sim.press("describe", "f7")
    sim.run(2.0)
    sim.press("advance")
    sim.view = synth.View(scene=2, text=S[1])  # 暗転を通らずに絵が変わった
    sim.run(2.0)
    sim.press("describe", "f7")
    sim.run(2.0)
    sim.press("describe", "f7")
    sim.run(2.0)
    return sim.said("description")


def test_gate_describe_current_tells_the_new_picture(game_path, style):
    sim = Sim(style, edit_game(game_path, lambda d: d["scene"].update(describe="current")))
    assert describe_twice(sim) == ["説明1", "説明2", "説明2"]


def test_gate_describe_last_is_kept_by_default(game_path, style):
    sim = Sim(style, load_config(game_path))
    assert describe_twice(sim) == ["説明1", "説明1", "説明1"]


def test_scene_describe_must_be_known(game_path):
    import pytest
    from sotozuke.game_config import ConfigError

    with pytest.raises(ConfigError):
        edit_game(game_path, lambda d: d["scene"].update(describe="always"))
