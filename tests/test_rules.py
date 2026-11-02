"""読み上げのきまりを、音を聞かずに記録で確かめる。"""
from __future__ import annotations

import pytest

import synth
import timeline
from harness import Sim
from sotozuke.game_config import load_config

S = timeline.SAMPLE_SENTENCES


def sim_for(game_path, style, **kwargs) -> Sim:
    return Sim(style, load_config(game_path), **kwargs)


def show(sim: Sim, text: str, name: str = "") -> None:
    sim.view = synth.View(scene=sim.view.scene, text=text, name=name)


def dawn(sim: Sim, scene: int) -> None:
    """暗転して、次の場面の絵が明ける。"""
    old = sim.view.scene
    sim.view = synth.View(scene=old, dark=1.0)
    sim.run(1.0)
    sim.view = synth.View(scene=scene)
    sim.run(0.5)


def test_sentence_is_read_after_it_stops(game_path, style):
    sim = sim_for(game_path, style)
    show(sim, S[0])
    sim.run(0.2)
    assert sim.reader.text_calls == 0  # 0.3秒止まるまで呼ばない
    sim.run(2.0)
    assert sim.reader.text_calls == 1
    assert sim.said("sentence") == [S[0]]


def test_only_the_latest_sentence_is_read_no_queue(game_path, style):
    """読み手を待っている間に次の文が出たら、古い文は鳴らさない。"""
    sim = sim_for(game_path, style, read_delay=1.0)
    show(sim, S[0])
    sim.run(0.6)
    assert sim.reader.text_calls == 1
    show(sim, S[1])
    sim.run(3.0)
    assert sim.said("sentence") == [S[1]]
    assert sim.events("drop_stale")


def test_press_stops_speech_and_next_sentence_is_read(game_path, style):
    sim = sim_for(game_path, style)
    show(sim, S[0])
    sim.run(1.6)
    assert sim.sound.playing
    sim.press()
    assert not sim.sound.playing
    assert sim.sound.stops == 1
    show(sim, S[1])
    sim.run(2.0)
    assert sim.said("sentence") == [S[0], S[1]]


def test_press_while_reading_drops_the_result(game_path, style):
    sim = sim_for(game_path, style, read_delay=1.0)
    show(sim, S[0])
    sim.run(0.8)
    assert sim.reader.text_calls == 1
    sim.press()
    show(sim, S[1])
    sim.run(3.0)
    assert sim.said("sentence") == [S[1]]


@pytest.mark.parametrize("interval", [0.2])
def test_rapid_presses_read_only_the_last(game_path, style, interval):
    """0.3秒より速く押し続ける間は、途中の文に読み手を呼ばない。"""
    sim = sim_for(game_path, style)
    for i in range(10):
        show(sim, S[i])
        sim.run(interval)
        sim.press()
    assert sim.reader.text_calls == 0
    show(sim, S[10])
    sim.run(3.0)
    assert sim.reader.text_calls == 1
    assert sim.said("sentence") == [S[10]]


@pytest.mark.parametrize("interval", [0.5, 1.0])
def test_slower_presses_never_play_old_sentences(game_path, style, interval):
    sim = sim_for(game_path, style, read_delay=0.8)
    order = []
    for i in range(6):
        show(sim, S[i])
        sim.run(interval)
        order.append(S[i])
        sim.press()
    show(sim, S[6])
    sim.run(3.0)
    said = sim.said("sentence")
    assert said[-1] == S[6]
    # 鳴らした文は順番どおりで、押したあとに古い文は鳴らない
    positions = [S.index(text) for text in said]
    assert positions == sorted(positions)
    for key in sim.events("key"):
        later = [r for r in sim.events("speak") if r["t"] > key["t"]]
        if later:
            assert S.index(later[0]["text"]) > 0


def test_voiced_line_is_not_read(game_path, style):
    sim = sim_for(game_path, style)
    show(sim, S[0], name="名前")
    sim.run(2.0)
    assert sim.reader.text_calls == 0
    assert sim.said() == []
    assert sim.events("text_ready")[0]["voiced"] is True


def test_voice_rule_all_reads_everything(style, tmp_path):
    path = synth.write_game(style, tmp_path, voice_rule="all")
    sim = Sim(style, load_config(path))
    show(sim, S[0], name="名前")
    sim.run(2.0)
    assert sim.said("sentence") == [S[0]]


def test_dots_only_are_told_as_silence(game_path, style):
    sim = sim_for(game_path, style)
    show(sim, "………")
    sim.run(2.0)
    assert sim.said("sentence") == ["無言。"]


def test_misread_dots_are_told_as_silence():
    from sotozuke.director import speakable
    assert speakable("ぉぉぉぉぉぉぉぉぉ") == "無言。"
    assert speakable("○○○○○○○○") == "無言。"
    assert speakable("………") == "無言。"
    assert speakable("あああ、そうか。") == "あああ、そうか。"


def test_repeat_reads_the_last_sentence_again(game_path, style):
    sim = sim_for(game_path, style)
    show(sim, S[0])
    sim.run(3.5)
    sim.press("repeat", "f6")
    assert sim.said("sentence") == [S[0], S[0]]
    assert sim.reader.text_calls == 1


def test_repeat_reads_a_voiced_line_on_request(game_path, style):
    sim = sim_for(game_path, style)
    show(sim, S[0], name="名前")
    sim.run(1.0)
    sim.press("repeat", "f6")
    sim.run(2.0)
    assert sim.said("sentence") == [S[0]]


def test_repeat_on_an_empty_screen_reads_the_whole_screen(game_path, style):
    sim = sim_for(game_path, style)
    sim.run(0.5)
    sim.press("repeat", "f6")
    sim.run(2.0)
    assert sim.said("screen") == ["screenの字"]


def test_failure_gives_a_short_tone_after_three_seconds(game_path, style):
    """読み手が遅いときは3秒で短い音が鳴る。黙り込まない。"""
    sim = sim_for(game_path, style, read_delay=5.0)
    show(sim, S[0])
    sim.run(3.2)
    assert sim.events("tone") == []
    sim.run(0.5)
    tones = sim.events("tone")
    assert [t["name"] for t in tones] == ["fail"]
    assert 3.2 <= tones[0]["t"] <= 3.6  # 文が止まって0.3秒、そこから3秒


def test_error_gives_a_tone_and_repeat_retries(game_path, style):
    sim = sim_for(game_path, style)
    sim.reader.fail_next = 1
    show(sim, S[0])
    sim.run(2.0)
    assert [t["name"] for t in sim.events("tone")] == ["fail"]
    sim.press("repeat", "f6")
    sim.run(2.0)
    assert sim.said("sentence") == [S[0]]


def test_description_then_first_sentence(game_path, style):
    sim = sim_for(game_path, style)
    dawn(sim, 2)
    show(sim, S[0])
    sim.run(6.0)
    spoken = [(r["what"], r["text"]) for r in sim.events("speak")]
    assert spoken[0][0] == "description"
    assert spoken[1] == ("sentence", S[0])


def test_key_stops_description_and_moves_on(game_path, style):
    sim = sim_for(game_path, style)
    dawn(sim, 2)
    show(sim, S[0])
    sim.run(1.2)
    assert sim.said("description")
    # 説明は短いので長くして途中で止める
    sim.sound.current = (sim.clock.now + 5.0, sim.sound.current[1])
    sim.press()
    assert not sim.sound.playing
    show(sim, S[1])
    sim.run(3.0)
    assert sim.said("sentence")[-1] == S[1]


def test_stop_key_stops_description_and_reads_the_held_sentence(game_path, style):
    sim = sim_for(game_path, style)
    dawn(sim, 2)
    show(sim, S[0])
    sim.run(1.2)
    sim.sound.current = (sim.clock.now + 5.0, sim.sound.current[1])
    sim.press("stop", "f8")
    sim.run(0.2)
    assert sim.said("sentence") == [S[0]]


def test_voiced_first_sentence_gets_a_chime_not_description(game_path, style):
    sim = sim_for(game_path, style, read_delay=1.0)
    sim.view = synth.View(scene=1, dark=1.0)
    sim.run(1.0)
    sim.view = synth.View(scene=2, text=S[0], name="名前")
    sim.run(3.0)
    assert sim.said("description") == []
    assert [t["name"] for t in sim.events("tone")] == ["chime"]
    sim.press("describe", "f7")
    assert sim.said("description")


def test_voiced_line_cuts_a_speaking_description(game_path, style):
    sim = sim_for(game_path, style, read_delay=0.2)
    dawn(sim, 2)
    sim.run(0.3)
    assert sim.said("description")
    sim.sound.current = (sim.clock.now + 5.0, sim.sound.current[1])
    show(sim, S[0], name="名前")
    sim.run(0.6)
    assert sim.sound.stops == 1  # 説明は止まり、合図の音だけが鳴る
    assert [t["name"] for t in sim.events("tone")] == ["chime"]


def test_same_picture_is_not_described_twice(game_path, style, monkeypatch):
    monkeypatch.setattr(synth, "MOTION", 0.0)
    sim = sim_for(game_path, style)
    dawn(sim, 2)
    show(sim, S[0])
    sim.run(3.0)
    dawn(sim, 3)
    show(sim, S[1])
    sim.run(3.0)
    dawn(sim, 2)
    show(sim, S[2])
    sim.run(3.0)
    assert sim.reader.describe_calls == 2
    assert len(sim.said("description")) == 3


def test_movie_without_text_is_not_described_again(game_path, style):
    """文が出ないまま暗転が続くのは映像の区間。説明は最初の1回だけ。"""
    sim = sim_for(game_path, style)
    for scene in (2, 3, 4, 5):
        dawn(sim, scene)
        sim.run(2.0)
    assert sim.reader.describe_calls == 1
    assert len(sim.events("dawn_skipped")) == 3


def test_description_is_kept_for_the_key(game_path, style):
    sim = sim_for(game_path, style)
    dawn(sim, 2)
    sim.run(3.0)
    count = len(sim.said("description"))
    sim.press("describe", "f7")
    assert len(sim.said("description")) == count + 1
    assert sim.reader.describe_calls == 1


def test_slow_description_does_not_block_the_sentence(game_path, style):
    sim = sim_for(game_path, style, read_delay=4.0)
    dawn(sim, 2)
    show(sim, S[0])
    sim.run(4.0)
    assert sim.said("sentence") == [S[0]] or sim.reader.text_calls == 1
    sim.run(3.0)
    assert sim.said("sentence") == [S[0]]
    assert sim.said("description") == []
    assert "chime" in [t["name"] for t in sim.events("tone")]


def test_description_failure_still_reads_the_sentence(game_path, style):
    sim = sim_for(game_path, style)
    sim.reader.fail_next = 1
    dawn(sim, 2)
    show(sim, S[0])
    sim.run(3.0)
    assert sim.said("sentence") == [S[0]]
