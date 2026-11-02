"""P2 の関門。合成の画面20枚で字まで全部正しい。時間と費用を記録する。"""
from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import synth
from sotozuke.costs import BudgetExceeded, CostLog, cost_of
from sotozuke.image_ops import crop
from sotozuke.reader import (ClaudeReader, ReaderError, clean_text, limit_description,
                             parse_items, response_text)

ROOT = Path(__file__).resolve().parent.parent

GATE_SENTENCES = [
    "三月の風はまだ少し冷たかった。",
    "「ねえ、聞いてる？」と彼女は言った。",
    "電車は十七時五分に駅を出た。",
    "カーテンの隙間から朝日が差しこむ。",
    "……まあ、そういうこともあるよな。",
    "コーヒーを一口飲んで、ため息をついた。",
    "黒板には大きな字で「自習」と書いてある。",
    "え？　今なんて言ったの？",
    "ページをめくる音だけが聞こえていた。",
    "放課後の教室に、夕日が長い影を落とす。",
    "テストの点数は、なんと百点満点だった！",
    "鞄の中から古い鍵が出てきた。",
    "そうか、君もあの町の生まれなんだね。",
    "ピアノの音が廊下の奥から流れてくる。",
    "雪が降り出したのは夜の九時ごろだった。",
    "ごめん、待った？　走ってきたんだけど。",
    "地図を広げて、次の行き先を確かめる。",
    "階段の踊り場で、誰かが泣いていた。",
    "明日はきっと晴れるよ、と母が笑った。",
    "遠くで踏切の鐘がカンカンと鳴っている。",
]


@dataclass
class Block:
    type: str
    text: str = ""


class Usage:
    input_tokens = 100
    output_tokens = 20


class FakeResponse:
    usage = Usage()

    def __init__(self, blocks):
        self.content = blocks


class FakeClient:
    def __init__(self, blocks):
        self.blocks = blocks
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        return FakeResponse(self.blocks)


def fake_reader(tmp_path, blocks, limit=4.5):
    return ClaudeReader(CostLog(tmp_path / "costs.jsonl", limit_usd=limit), client=FakeClient(blocks))


def test_text_part_is_found_after_thinking():
    assert response_text(FakeResponse([Block("thinking"), Block("text", "本文")])) == "本文"


def test_decorations_are_removed():
    assert clean_text("**朝の\n光**") == "朝の光"
    assert clean_text("  `文`\n") == "文"


def test_items_are_parsed_with_extra_words():
    assert parse_items('こちらです。\n["一つ目", "二つ目"]') == ["一つ目", "二つ目"]
    with pytest.raises(ReaderError):
        parse_items("項目はありません")


def test_description_limit():
    long = "放課後の教室で、長い黒髪の女子が窓の外を見ている。夕日が差している。机の上には本がある。"
    out = limit_description(long)
    assert len(out) <= 60
    assert out.count("。") <= 2
    assert out.endswith("。")


def test_cost_is_logged_and_totaled(tmp_path):
    reader = fake_reader(tmp_path, [Block("text", "文です。")])
    frame = np.zeros((10, 10, 3), np.uint8)
    result = reader.read_text(frame)
    assert result.text == "文です。"
    assert reader.costs.total() == pytest.approx(cost_of("claude-haiku-5-5", 100, 20), abs=1e-6)


def test_budget_stops_calls(tmp_path):
    reader = fake_reader(tmp_path, [Block("text", "文")], limit=0.0)
    with pytest.raises(BudgetExceeded):
        reader.read_text(np.zeros((10, 10, 3), np.uint8))
    assert reader.client.calls == 0


@pytest.mark.api
def test_gate_twenty_screens_exact():
    assert os.environ.get("ANTHROPIC_API_KEY_FILE")
    reader = ClaudeReader(CostLog(ROOT / "costs.jsonl"))
    times = []
    usd = 0.0
    wrong = []
    for i, text in enumerate(GATE_SENTENCES):
        style = synth.STYLES["a" if i % 2 == 0 else "b"]
        frame = synth.render(style, synth.View(scene=i + 1, text=text, name="名前" if i % 3 == 0 else ""))
        result = reader.read_text(crop(frame, style.band))
        times.append(result.seconds)
        usd += result.usd
        if result.text != text:
            wrong.append((text, result.text))
    print(f"20枚 中央値 {statistics.median(times):.2f}秒 最大 {max(times):.2f}秒 費用 {usd:.5f}ドル")
    for truth, got in wrong:
        print(f"違い 正解={truth} 読み={got}")
    assert not wrong
