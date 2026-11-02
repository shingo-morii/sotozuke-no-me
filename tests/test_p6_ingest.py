"""P6 の関門。作りの違う合成の画面2種類で、本文の帯の位置のずれが許容の範囲に入る。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import synth
from ingest.action_time import summarize
from ingest.check_map import check
from ingest.locate import locate, refine_band
from sotozuke.costs import CostLog
from sotozuke.game_config import GlyphRule, load_config
from sotozuke.image_ops import glyph_mask, to_gray
from sotozuke.reader import ClaudeReader

ROOT = Path(__file__).resolve().parent.parent
LIMIT = 16
LONG = "長い文の見本です。" * 8


def sample_frames(style) -> list[np.ndarray]:
    texts = [LONG[:52], "短い文です。", LONG[:40], "もう少しだけ長い文を書いています。", LONG[:58], "朝になった。"]
    return [synth.render(style, synth.View(scene=i + 1, text=t, name="名前" if i % 2 else ""), i * 3.0)
            for i, t in enumerate(texts)]


def true_extent(style) -> tuple[int, int, int, int]:
    """字が実際に出た範囲。絵の無い背景で描いて数える。"""
    view = synth.View(scene=1, text=LONG[:58])
    with_text = synth.render(style, view).astype(int)
    without = synth.render(style, synth.View(scene=1)).astype(int)
    mask = np.abs(with_text - without).sum(axis=2) > 200
    if style.window is not None:
        x0, y0, x1, y1 = style.window
        inside = np.zeros(mask.shape, bool)
        inside[y0 + 3:y1 - 3, x0 + 3:x1 - 3] = True
        window_only = synth.render(style, synth.View(scene=1, text=" ")).astype(int)
        mask = (np.abs(with_text - window_only).sum(axis=2) > 200) & inside
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def assert_within(band, truth):
    assert all(abs(a - b) <= LIMIT for a, b in zip(band, truth)), (band, truth)
    assert band[0] <= truth[0] and band[1] <= truth[1]
    assert band[2] >= truth[2] and band[3] >= truth[3]


@pytest.mark.parametrize("error", [0, 4, -4])
def test_refine_fixes_a_rough_answer(style, error):
    frames = sample_frames(style)
    truth = true_extent(style)
    rough = (truth[0] + error, truth[1] + error, truth[2] - error, truth[3] - error)
    band = refine_band(frames, rough, GlyphRule())
    assert_within(band, truth)


@pytest.mark.api
def test_gate_position_from_claude(style):
    frames = sample_frames(style)
    truth = true_extent(style)
    reader = ClaudeReader(CostLog(ROOT / "costs.jsonl"))
    result = locate(reader, frames, GlyphRule())
    print("帯", style.name, result, "正解", truth)
    assert_within(tuple(result["band"]), truth)


def test_check_map_counts_each_kind(style, game_path, tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "選択肢").mkdir(parents=True)
    (corpus / "文").mkdir()
    hint = "上下 選択  決定  履歴  設定"
    frame = synth.render(style, synth.View(scene=1, choices=["一", "二"], hint=hint))
    Image.fromarray(frame).save(corpus / "選択肢" / "0.png")
    frame = synth.render(style, synth.View(scene=1, text="文です。"))
    Image.fromarray(frame).save(corpus / "文" / "0.png")
    report = check(str(game_path), str(corpus))
    assert report["選択肢"]["counts"] == {"選択肢": 1}
    assert report["文"]["counts"] == {"文": 1}


def test_check_map_flags_other_sizes(game_path, tmp_path):
    corpus = tmp_path / "corpus" / "文"
    corpus.mkdir(parents=True)
    Image.new("RGB", (640, 480)).save(corpus / "0.png")
    report = check(str(game_path), str(tmp_path / "corpus"))
    assert report["文"]["counts"] == {"大きさが違う": 1}


def test_action_log_is_counted(tmp_path):
    path = tmp_path / "actions.jsonl"
    rows = [{"time": 100.0, "action": "key", "name": "enter"},
            {"time": 110.0, "action": "mark", "name": "タイトル"},
            {"time": 120.0, "action": "mark", "name": "進行開始"},
            {"time": 150.0, "action": "mark", "name": "進行終了"},
            {"time": 160.0, "action": "shot", "name": "a"}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    result = summarize(path)
    assert result["total_s"] == 60.0
    assert result["keys"] == 1
    assert result["progress_s"] == 30.0
    assert ("タイトル", 10.0) in [tuple(m) for m in result["marks"]]


def test_key_table_covers_what_a_game_needs():
    from sotozuke.inputs import VK
    for name in ("enter", "space", "escape", "up", "down", "left", "right", "shift", "backspace", "delete"):
        assert name in VK


def test_skill_exists_and_has_the_goals():
    text = (ROOT / ".claude" / "skills" / "add-game" / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\nname: add-game")
    for goal in ("タイトル", "はじめから", "名前の札", "選択肢", "セーブ", "ロード"):
        assert goal in text
    assert "## 経験" in text


def test_skill_has_no_engine_knowledge():
    """エンジンごとの知識を書くと初見の探索でなくなる。"""
    text = (ROOT / ".claude" / "skills" / "add-game" / "SKILL.md").read_text(encoding="utf-8")
    assert not re.search(r"Ren'?Py|ティラノ|Tyrano|KiriKiri|吉里吉里", text, re.IGNORECASE)


def test_generated_config_loads(game_path):
    assert load_config(game_path).name.startswith("synth_")
    assert glyph_mask(to_gray(np.zeros((8, 8, 3), np.uint8)), GlyphRule()).sum() == 0
