"""G1b フェイズ B の関門。札のある文で、札の字も読み「名前。本文」と読む。同じ札は読み直さない。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import synth
import timeline
from harness import Sim
from sotozuke.costs import CostLog
from sotozuke.game_config import ConfigError, load_config
from sotozuke.image_ops import crop
from sotozuke.reader import ClaudeReader

ROOT = Path(__file__).resolve().parent.parent
S = timeline.SAMPLE_SENTENCES


def name_box(style):
    """札の字の位置。style a は線の上の名前、style b は窓の上の札。"""
    if style.tag_kind == "hline":
        w = style.size[0]
        return [w // 2 - 150, style.tag_y - 30, w // 2 + 150, style.tag_y - 3]
    return [170, 492, 420, 526]


def named_game(style, tmp_path, read_name=True):
    path = synth.write_game(style, tmp_path, voice_rule="all")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tag"]["box"] = name_box(style)
    if read_name:
        data["tag"]["read_name"] = True
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return load_config(path)


def show(sim, text, name=""):
    sim.press()
    sim.view = synth.View(scene=1, text=text, name=name)
    sim.run(2.5)


def test_gate_name_is_read_with_the_sentence_and_remembered(style, tmp_path, monkeypatch):
    monkeypatch.setattr(synth, "MOTION", 0.0)
    sim = Sim(style, named_game(style, tmp_path))
    show(sim, S[0], "山田")
    show(sim, S[1])
    show(sim, S[2], "山田")
    show(sim, S[3], "先生")
    show(sim, S[4], "山田")
    assert sim.said("sentence") == [
        f"山田。{S[0]}", S[1], f"山田。{S[2]}", f"先生。{S[3]}", f"山田。{S[4]}",
    ]
    assert sim.reader.name_calls == 2  # 同じ形の札は覚えている


def test_name_is_not_read_when_not_written(style, tmp_path):
    sim = Sim(style, named_game(style, tmp_path, read_name=False))
    show(sim, "台詞です。", "山田")
    assert sim.said("sentence") == ["台詞です。"]
    assert sim.reader.name_calls == 0


def test_empty_name_is_remembered_and_not_spoken(style, tmp_path, monkeypatch):
    monkeypatch.setattr(synth, "MOTION", 0.0)
    sim = Sim(style, named_game(style, tmp_path))
    sim.reader.truth_name = lambda: ""
    show(sim, S[0], "山田")
    show(sim, S[2], "山田")
    assert sim.said("sentence") == [S[0], S[2]]
    assert sim.reader.name_calls == 1


def test_read_name_needs_box(tmp_path):
    path = synth.write_game(synth.STYLE_A, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tag"]["read_name"] = True
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.api
def test_gate_name_tag_is_read_by_claude():
    assert os.environ.get("ANTHROPIC_API_KEY_FILE")
    reader = ClaudeReader(CostLog(ROOT / "costs.jsonl"))
    for style, name in ((synth.STYLE_A, "山田"), (synth.STYLE_B, "先生")):
        frame = synth.render(style, synth.View(scene=2, text="台詞の文です。", name=name))
        image = crop(frame, tuple(name_box(style)))
        result = reader.read_text(crop(frame, style.band), name_image=image)
        print(f"名前 {result.name} 本文 {result.text} {result.seconds:.2f}秒")
        assert result.name == name
        assert result.text == "台詞の文です。"
    empty = synth.render(synth.STYLE_B, synth.View(scene=2, text="台詞の文です。"))
    assert reader.read_name(crop(empty, tuple(name_box(synth.STYLE_B)))).text == ""
