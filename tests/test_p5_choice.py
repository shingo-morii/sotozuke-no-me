"""P5 の関門。全部の選択肢を読める。項目の移動から0.3秒以内に読み始める。

読み上げのきまりも、音を聞かずに記録で確かめる。
"""
from __future__ import annotations

import queue
import random
import threading
import time

import numpy as np
import pytest

import synth
from harness import Sim
from sotozuke.director import Director, Services, selected_row
from sotozuke.game_config import load_config
from sotozuke.watcher import Watcher

HINT = "上下 選択  決定  履歴  設定"


def sim_for(game_path, style, **kwargs) -> Sim:
    return Sim(style, load_config(game_path), **kwargs)


def choice_view(view: synth.View, items, selected=0) -> synth.View:
    return synth.View(scene=view.scene, text=view.text, name=view.name, choices=list(items),
                      selected=selected, hint=HINT)


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_all_choice_items_are_read(game_path, style, count):
    sim = sim_for(game_path, style)
    items = [f"項目の{n}番目の字" for n in range(1, count + 1)]
    sim.view = synth.View(scene=1, text="前の文です。")
    sim.run(2.0)
    sim.view = choice_view(sim.view, items)
    sim.run(3.0)
    ready = sim.events("choice_ready")
    assert len(ready) == 1
    assert ready[0]["items"] == items
    assert ready[0]["selected"] == 0
    intro = sim.said("choice")[0]
    for item in items:
        assert item in intro


def test_selected_row_on_all_positions(game_path, style):
    config = load_config(game_path)
    for count in (2, 3, 4, 5):
        rows = config.choice.rows_for(count)
        items = [f"項目{n}" for n in range(count)]
        for selected in range(count):
            frame = synth.render(style, synth.View(scene=3, choices=items, selected=selected, hint=HINT))
            assert selected_row(frame, rows) == selected


def test_rows_for_unknown_count_extends_known_rows(game_path):
    config = load_config(game_path)
    rule = config.choice
    five = rule.rows_by_count[5]
    del rule.rows_by_count[5]
    for got, truth in zip(rule.rows_for(5), five):
        assert all(abs(a - b) <= 2 for a, b in zip(got, truth))


def test_move_reads_only_the_new_item_at_once(game_path, style):
    sim = sim_for(game_path, style)
    items = ["一つ目の項目", "二つ目の項目", "三つ目の項目"]
    sim.view = choice_view(synth.View(scene=1), items)
    sim.run(3.0)
    assert sim.said("choice")
    sim.view = choice_view(synth.View(scene=1), items, selected=1)
    sim.run(0.3)
    assert sim.said("choice")[-1] == "二つ目の項目"
    sim.view = choice_view(synth.View(scene=1), items, selected=2)
    sim.run(0.3)
    assert sim.said("choice")[-1] == "三つ目の項目"
    # 動かすたびに読み手を呼ばない
    assert sim.reader.items_calls == 1


def test_move_while_intro_stops_and_reads_item(game_path, style):
    sim = sim_for(game_path, style)
    items = ["一つ目の長い項目です", "二つ目の長い項目です"]
    sim.view = choice_view(synth.View(scene=1), items)
    sim.run(2.2)
    assert sim.sound.playing  # 紹介の途中
    sim.view = choice_view(synth.View(scene=1), items, selected=1)
    sim.run(0.3)
    assert sim.said("choice")[-1] == "二つ目の長い項目です"


def test_move_before_items_are_ready_skips_intro(game_path, style):
    sim = sim_for(game_path, style, read_delay=2.0)
    items = ["一つ目の項目", "二つ目の項目"]
    sim.view = choice_view(synth.View(scene=1), items)
    sim.run(0.8)
    sim.view = choice_view(synth.View(scene=1), items, selected=1)
    sim.run(2.5)
    # 読めたときに、今の項目が intro の選択中として伝わる
    assert "選んでいるのは、二つ目の項目" in sim.said("choice")[-1]


def test_one_frame_flicker_is_not_read(game_path, style):
    """帯が動く途中の1コマの順位の揺れでは読まない。"""
    sim = sim_for(game_path, style)
    items = ["一つ目の項目", "二つ目の項目", "三つ目の項目"]
    sim.view = choice_view(synth.View(scene=1), items, selected=0)
    sim.run(5.0)
    sim.view = choice_view(synth.View(scene=1), items, selected=2)
    sim.run(0.1)
    sim.view = choice_view(synth.View(scene=1), items, selected=1)
    sim.run(0.5)
    assert sim.said("choice")[1:] == ["二つ目の項目"]


def test_nothing_is_read_after_deciding(game_path, style):
    sim = sim_for(game_path, style)
    items = ["一つ目の項目", "二つ目の項目"]
    sim.view = choice_view(synth.View(scene=1), items, selected=0)
    sim.run(5.0)
    before = len(sim.said("choice"))
    sim.press()
    sim.view = choice_view(synth.View(scene=1), items, selected=1)
    sim.run(0.5)
    assert len(sim.said("choice")) == before


def test_last_sentence_before_choice_is_read(game_path, style):
    """文が止まりきる前に選択肢が重なっても、文を読み落とさない。"""
    sim = sim_for(game_path, style)
    sim.run(0.5)
    sim.view = synth.View(scene=1, text="すぐに選択肢が出る文。")
    sim.run(0.2)
    sim.view = choice_view(sim.view, ["一つ目", "二つ目"])
    sim.run(6.0)
    assert sim.said("sentence") == ["すぐに選択肢が出る文。"]
    assert sim.said("choice")


def test_old_sentence_is_not_read_again_in_choice(game_path, style):
    sim = sim_for(game_path, style)
    sim.view = synth.View(scene=1, text="前の文です。", name="")
    sim.run(3.0)
    assert sim.said("sentence") == ["前の文です。"]
    sim.view = choice_view(sim.view, ["一つ目", "二つ目"])
    sim.run(4.0)
    assert sim.said("sentence") == ["前の文です。"]


def test_failure_is_told_by_a_tone(game_path, style):
    sim = sim_for(game_path, style)
    sim.reader.fail_next = 1
    sim.view = choice_view(synth.View(scene=1), ["一つ目", "二つ目"])
    sim.run(2.0)
    assert [r["name"] for r in sim.events("tone")] == ["fail"]


class RealTimeTasks:
    def __init__(self):
        self.q: queue.Queue = queue.Queue()

    def submit(self, work, done):
        def run():
            try:
                value, error = work(), None
            except Exception as exc:  # noqa: BLE001
                value, error = None, exc
            self.q.put(lambda: done(value, error))
        threading.Thread(target=run, daemon=True).start()

    def post(self, call):
        self.q.put(call)

    def drain(self):
        while True:
            try:
                self.q.get_nowait()()
            except queue.Empty:
                return


@pytest.mark.audio
def test_gate_move_starts_within_limit(game_path, style, tmp_path):
    """移動した項目が画面に出てから、再生の処理を始めるまで。実時間、実際の声と再生で測る。"""
    from sotozuke.player import Player, load_audio, tone
    from sotozuke.voice import SapiVoice, VoiceCache

    config = load_config(game_path)
    cache = VoiceCache(SapiVoice(), tmp_path / "voice")
    items = ["一つ目の項目です", "二つ目の項目です", "三つ目の項目です"]
    frames = [synth.render(style, synth.View(scene=2, choices=items, selected=i, hint=HINT))
              for i in range(3)]

    class Reader:
        def read_items(self, area):
            return items, None

    real_player = Player()
    started: list[float] = []

    class Spy:
        playing = False

        def play(self, audio, on_end=None):
            started.append(time.perf_counter())
            real_player.play(audio * 0.02, None)

        def stop(self):
            real_player.stop()

    tasks = RealTimeTasks()
    services = Services(
        reader=Reader(), speech=lambda text: load_audio(cache.get(text)), sound=Spy(), tasks=tasks,
        clock=time.monotonic, log=lambda *a, **k: None,
        chime=tone(880, 0.1), fail_tone=tone(330, 0.2),
    )
    director = Director(config, Watcher(config), services)

    period = 0.1
    state = {"selected": 0}
    latencies = []
    rng = random.Random(1)

    schedule = {"next": time.perf_counter()}

    def loop_for(seconds):
        """10コマ毎秒の見張りと同じ間隔で回す。移動はコマの間のどこでも起きる。"""
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            tasks.drain()
            now = time.perf_counter()
            if now >= schedule["next"]:
                director.step(frames[state["selected"]], time.monotonic())
                schedule["next"] += period
            time.sleep(0.002)

    try:
        loop_for(1.5)  # 選択肢が出て、項目を読み終わるまで
        assert director.choice is not None and director.choice.ready
        for _ in range(30):
            before = len(started)
            loop_for(rng.uniform(0.0, 0.1) + 0.12)
            state["selected"] = (state["selected"] + 1) % 3
            moved_at = time.perf_counter()
            deadline = moved_at + 1.0
            while len(started) == before and time.perf_counter() < deadline:
                loop_for(0.01)
            assert len(started) > before
            latencies.append(started[-1] - moved_at)
    finally:
        real_player.close()
    print(f"移動から読み始めまで 中央値 {np.median(latencies)*1000:.0f}ms 最大 {max(latencies)*1000:.0f}ms")
    assert max(latencies) <= 0.3
