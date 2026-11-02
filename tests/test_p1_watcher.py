"""P1 の関門。文の見落とし0、余分は1割以内、暗転の明けの見落とし0、1コマ5ミリ秒以内。"""
from __future__ import annotations

import json
import statistics
import time

import numpy as np
import pytest

import synth
import timeline
from sotozuke.game_config import load_config
from sotozuke.image_ops import crop, glyph_mask, to_gray
from sotozuke.watcher import (KindChanged, SceneDawn, ScreenSettled, TextReady, Watcher,
                              has_name_tag)

FPS = 10


def run(line: timeline.Timeline, config) -> list:
    watcher = Watcher(config)
    events = []
    t = 0.0
    while t < line.end:
        events += watcher.feed(line.frame(t), t)
        t += 1 / FPS
    return events


def score_sentences(line, events):
    ready = [e for e in events if isinstance(e, TextReady)]
    matched = set()
    extra = 0
    for event in ready:
        hit = None
        for i, seg in enumerate(line.sentences):
            if seg.start <= event.t < seg.end + 0.01:
                hit = i
                break
        if hit is None or hit in matched:
            extra += 1
        else:
            matched.add(hit)
    return len(line.sentences) - len(matched), extra


@pytest.mark.parametrize("cps", [None, 40, 13])
def test_gate_sentences_not_missed(style, game_path, cps):
    config = load_config(game_path)
    line = timeline.build(style, count=24, seed=3, typing_cps=cps)
    events = run(line, config)
    missed, extra = score_sentences(line, events)
    assert missed == 0
    assert extra <= len(line.sentences) * 0.1


def test_gate_sentences_without_gap(style, game_path):
    """前の文のすぐ次に別の文が出ても見落とさない。"""
    config = load_config(game_path)
    line = timeline.build(style, count=30, seed=5, gap_s=0.0, blackouts=0)
    # 同じ文が続く所は見分けられないので、続かないように直す
    for a, b in zip(line.sentences, line.sentences[1:]):
        if a.view.text == b.view.text:
            b.view.text = b.view.text + "そうだ。"
    events = run(line, config)
    missed, extra = score_sentences(line, events)
    assert missed == 0
    assert extra <= len(line.sentences) * 0.1


def test_same_sentence_after_gap_counts_as_new(style, game_path):
    config = load_config(game_path)
    line = timeline.Timeline(style)
    text = timeline.SAMPLE_SENTENCES[0]
    for i in range(3):
        start = i * 2.0
        seg = timeline.Segment(start, start + 1.7, synth.View(scene=1, text=text), label="sentence")
        line.segments.append(seg)
        line.sentences.append(seg)
        line.segments.append(timeline.Segment(start + 1.7, start + 2.0, synth.View(scene=1)))
    events = run(line, config)
    assert len([e for e in events if isinstance(e, TextReady)]) == 3


def test_key_restarts_the_wait(style, game_path):
    """連打の間は止まるまで知らせない。"""
    config = load_config(game_path)
    watcher = Watcher(config)
    ready = 0
    t = 0.0
    sentences = timeline.SAMPLE_SENTENCES
    for i in range(10):
        view = synth.View(scene=1, text=sentences[i])
        frame = synth.render(style, view, t)
        for _ in range(2):  # 0.2秒ごとに押す
            ready += sum(isinstance(e, TextReady) for e in watcher.feed(frame, t))
            t += 1 / FPS
        watcher.on_key()
    assert ready == 0
    frame = synth.render(style, synth.View(scene=1, text=sentences[10]), t)
    for _ in range(6):
        ready += sum(isinstance(e, TextReady) for e in watcher.feed(frame, t))
        t += 1 / FPS
    assert ready == 1


def noisy(frame: np.ndarray, band, rng, ratio: float) -> np.ndarray:
    """動く背景の字の形のノイズの代わりに、本文の帯へ明るい点と暗い点を散らす。"""
    out = frame.copy()
    x0, y0, x1, y1 = band
    count = int((x1 - x0) * (y1 - y0) * ratio)
    xs = rng.integers(x0 + 2, x1 - 2, count)
    ys = rng.integers(y0 + 2, y1 - 2, count)
    out[ys, xs] = 255
    out[ys + 1, xs] = 0
    return out


def test_moving_background_does_not_hide_sentences(style, tmp_path):
    """文字が一度に出るゲームでは、背景が動いて字の形が揺れても文を見落とさない。"""
    path = synth.write_game(style, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["text"]["typed"] = False
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    config = load_config(path)
    watcher = Watcher(config)
    rng = np.random.default_rng(5)
    texts = [timeline.SAMPLE_SENTENCES[0], "うん", timeline.SAMPLE_SENTENCES[1] * 2, "はい。", "そう。"]
    t = 0.0
    ready = 0
    for text in texts:
        frame = synth.render(style, synth.View(scene=1, text=text))
        for _ in range(12):
            ready += sum(isinstance(e, TextReady)
                         for e in watcher.feed(noisy(frame, config.text.band, rng, 0.002), t))
            t += 1 / FPS
    assert ready == len(texts)


def test_gate_dawn_not_missed(style, game_path):
    config = load_config(game_path)
    line = timeline.build(style, count=24, seed=7, blackouts=5)
    assert len(line.dawns) >= 5
    events = run(line, config)
    dawns = [e.t for e in events if isinstance(e, SceneDawn)]
    for truth in line.dawns:
        assert any(truth - 1.0 <= d <= truth + 1.0 for d in dawns), truth
    assert len(dawns) <= len(line.dawns)


def test_text_in_dark_is_still_read(style, game_path):
    """黒い背景のまま文が出ても読む。絵が無いので明けの知らせは出さない。"""
    config = load_config(game_path)
    watcher = Watcher(config)
    text = timeline.SAMPLE_SENTENCES[2]
    black = np.zeros((style.size[1], style.size[0], 3), np.uint8)
    t = 0.0
    events = []
    for _ in range(10):
        events += watcher.feed(black, t)
        t += 0.1
    view = synth.View(scene=1, text=text, dark=1.0)
    frame = synth.render(style, synth.View(scene=1, text=text), t)
    mask = glyph_mask(to_gray(crop(frame, config.text.band)), config.glyph)
    assert mask.any()
    composed = black.copy()
    x0, y0, x1, y1 = config.text.band
    composed[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
    for _ in range(6):
        events += watcher.feed(composed, t)
        t += 0.1
    assert view.dark == 1.0
    assert any(isinstance(e, TextReady) for e in events)
    assert not any(isinstance(e, SceneDawn) for e in events)


def test_tag_seen_on_text_screen(style, game_path):
    config = load_config(game_path)
    with_name = synth.render(style, synth.View(scene=2, text="文です。", name="名前"))
    without = synth.render(style, synth.View(scene=2, text="文です。"))
    assert has_name_tag(with_name, config)
    assert not has_name_tag(without, config)


def test_kind_decided_first_and_text_stops(style, game_path):
    """選択肢の画面には前の文と札が残っている。文の見張りを止める。"""
    config = load_config(game_path)
    watcher = Watcher(config)
    text = timeline.SAMPLE_SENTENCES[3]
    t = 0.0
    frame = synth.render(style, synth.View(scene=1, text=text, name="名前"))
    events = []
    for _ in range(8):
        events += watcher.feed(frame, t)
        t += 0.1
    assert sum(isinstance(e, TextReady) for e in events) == 1

    hint = "上下 選択  決定  履歴  設定"
    frame = synth.render(style, synth.View(scene=1, text=text, name="名前",
                                           choices=["一つ目", "二つ目"], hint=hint))
    events = []
    for _ in range(8):
        events += watcher.feed(frame, t)
        t += 0.1
    assert any(isinstance(e, KindChanged) and e.new == "choice" for e in events)
    assert not any(isinstance(e, TextReady) for e in events)
    assert watcher.kind == "choice"


def test_old_sentence_not_repeated_after_menu(style, game_path):
    config = load_config(game_path)
    watcher = Watcher(config)
    text = timeline.SAMPLE_SENTENCES[4]
    t = 0.0
    events = []
    frame = synth.render(style, synth.View(scene=1, text=text))
    for _ in range(8):
        events += watcher.feed(frame, t)
        t += 0.1
    menu = synth.render(style, synth.View(scene=1, hint="上下左右 選択  決定  戻る",
                                          menu_items=["セーブ", "ロード"], menu_title="メニュー"))
    for _ in range(8):
        events += watcher.feed(menu, t)
        t += 0.1
    for _ in range(8):
        events += watcher.feed(frame, t)
        t += 0.1
    assert sum(isinstance(e, TextReady) for e in events) == 1


def test_screen_settled_once_per_change(style, game_path):
    config = load_config(game_path)
    watcher = Watcher(config)
    t = 0.0
    events = []
    for selected in (0, 0, 1, 1):
        menu = synth.render(style, synth.View(scene=1, hint="上下左右 選択  決定  戻る", selected=selected,
                                              menu_items=["セーブ", "ロード"], menu_title="メニュー"))
        for _ in range(6):
            events += watcher.feed(menu, t)
            t += 0.1
    assert sum(isinstance(e, ScreenSettled) for e in events) == 2


def test_gate_frame_time(style, game_path):
    config = load_config(game_path)
    line = timeline.build(style, count=20, seed=11)
    frames = [line.frame(t) for t in np.arange(0, line.end, 1.0)]
    watcher = Watcher(config)
    times = []
    for i, frame in enumerate(frames * 3):
        start = time.perf_counter()
        watcher.feed(frame, i / FPS)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    median = statistics.median(times)
    p95 = times[int(len(times) * 0.95)]
    print(f"1コマ 中央値 {median:.2f}ms 95% {p95:.2f}ms")
    assert median <= 5.0
    assert p95 <= 5.0
