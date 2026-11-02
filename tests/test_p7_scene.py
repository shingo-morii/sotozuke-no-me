"""P7 の関門。説明の回数が見込みの範囲に入る。2文以内、60字以内。"""
from __future__ import annotations

import os
import re
import statistics
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import synth
import timeline
from harness import Sim
from sotozuke.costs import CostLog
from sotozuke.game_config import load_config
from sotozuke.reader import ClaudeReader

ROOT = Path(__file__).resolve().parent.parent
S = timeline.SAMPLE_SENTENCES


def go_dark_then(sim: Sim, view: synth.View) -> None:
    sim.view = synth.View(scene=sim.view.scene, dark=1.0)
    sim.run(1.0)
    sim.view = view
    sim.run(0.6)


def test_gate_description_count(game_path, style, monkeypatch):
    """場面5つ、日付の札1枚、同じ絵へ戻る1回の通しで、説明の生成は5回から6回。"""
    monkeypatch.setattr(synth, "MOTION", 0.0)
    sim = Sim(style, load_config(game_path))
    for scene in (2, 3, 4):
        go_dark_then(sim, synth.View(scene=scene))
        sim.view = synth.View(scene=scene, text=S[scene])
        sim.run(3.0)
    go_dark_then(sim, synth.View(scene=4, card="三月四日 午後五時"))
    sim.run(2.0)
    for scene in (5, 6):
        go_dark_then(sim, synth.View(scene=scene))
        sim.view = synth.View(scene=scene, text=S[scene])
        sim.run(3.0)
    go_dark_then(sim, synth.View(scene=2))  # 同じ絵へ戻る
    sim.view = synth.View(scene=2, text=S[7])
    sim.run(3.0)
    assert 5 <= sim.reader.describe_calls <= 6
    assert "cardの字" in sim.said("screen")
    assert len(sim.said("description")) == 6  # 戻った場面は作り直さず読む


def test_card_is_read_not_described(game_path, style):
    sim = Sim(style, load_config(game_path))
    go_dark_then(sim, synth.View(scene=1, card="三月四日"))
    sim.run(2.0)
    assert sim.reader.describe_calls == 0
    assert sim.said("screen") == ["cardの字"]


def test_small_changes_are_not_scene_changes(game_path, style):
    sim = Sim(style, load_config(game_path))
    sim.view = synth.View(scene=3, text=S[0])
    sim.run(3.0)
    sim.view = synth.View(scene=3, text=S[1])
    sim.run(3.0)
    assert sim.reader.describe_calls == 0


@pytest.mark.api
def test_gate_description_length():
    assert os.environ.get("ANTHROPIC_API_KEY_FILE")
    reader = ClaudeReader(CostLog(ROOT / "costs.jsonl"))
    times = []
    for scene in (1, 2, 3):
        frame = synth.render(synth.STYLE_A, synth.View(scene=scene))
        result = reader.describe(frame)
        times.append(result.seconds)
        print(f"説明 {len(result.text)}字 {result.seconds:.2f}秒 {result.text}")
        assert 0 < len(result.text) <= 60
        assert len(re.findall(r"[。！？]", result.text)) <= 2
    print(f"中央値 {statistics.median(times):.2f}秒")


@pytest.mark.local
@pytest.mark.api
def test_local_scenes_are_described():
    folder = os.environ.get("SOTOZUKE_LOCAL_SCENES")
    if not folder:
        pytest.skip("手元の場面の画面がありません")
    reader = ClaudeReader(CostLog(ROOT / "costs.jsonl"))
    for path in sorted(Path(folder).glob("場面*.png")):
        frame = np.asarray(Image.open(path).convert("RGB"))
        result = reader.describe(frame)
        print(path.name, f"{result.seconds:.2f}秒", result.text)
        assert 0 < len(result.text) <= 60
