"""本文の帯の位置を見つける。

Claude に文の画面を見せて、おおよその位置を答えてもらい、その近くの字の形を手元で数えて位置を詰める。
答えが揺れても、字の形で決めるので位置はぶれない。結果は設定に書き、以後は API を呼ばない。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from sotozuke.costs import CostLog
from sotozuke.game_config import GlyphRule
from sotozuke.image_ops import glyph_mask, to_gray
from sotozuke.reader import PICTURE_MODEL, ClaudeReader, parse_object

PROMPT_LOCATE = (
    "画像はノベルゲームの文の画面です。幅{width}、高さ{height}です。"
    "物語の本文が表示される範囲を、幅{width}、高さ{height}の画素の座標で答えてください。"
    "名前の札、操作の案内、飾りは含めず、本文だけの範囲にします。"
    'JSON だけを返してください。{{"band": [左, 上, 右, 下]}}'
)
ROUGH_MARGIN = 12  # Claude の答えは合成の画面で3画素以内だった。名前の札を拾わない幅にする
PAD = 8


def ask_rough_band(reader: ClaudeReader, frame: np.ndarray) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    prompt = PROMPT_LOCATE.format(width=width, height=height)
    result = reader.ask(frame, prompt, PICTURE_MODEL, "locate", max_tokens=2000)
    box = parse_object(result.text)["band"]
    return tuple(int(v) for v in box)


def refine_band(frames: list[np.ndarray], rough: tuple[int, int, int, int], rule: GlyphRule,
                margin: int = ROUGH_MARGIN, pad: int = PAD) -> tuple[int, int, int, int]:
    """おおよその範囲の近くで、字の形が出た範囲を数えて、帯を決める。"""
    height, width = frames[0].shape[:2]
    x0 = max(rough[0] - margin, 0)
    y0 = max(rough[1] - margin, 0)
    x1 = min(rough[2] + margin, width)
    y1 = min(rough[3] + margin, height)
    # 動かない背景の縁は全部の画面に出る。半分より多くの画面に出る点は字ではないと見なす
    masks = [glyph_mask(to_gray(frame[y0:y1, x0:x1]), rule) for frame in frames]
    votes = np.sum(masks, axis=0)
    limit = max(1, len(frames) // 2)
    text = (votes >= 1) & (votes <= limit) if len(frames) >= 3 else votes >= 1
    top, bottom = strong_rows(text.sum(axis=1), max(12, height // 25))
    if top is None:
        return (x0, y0, x1, y1)
    cols = np.where(text[top:bottom].sum(axis=0) >= 2)[0]
    if len(cols) == 0:
        return (x0, y0, x1, y1)
    return (max(int(cols[0]) + x0 - pad, 0), max(top + y0 - pad, 0),
            min(int(cols[-1]) + x0 + pad + 1, width), min(bottom + y0 + pad, height))


def strong_rows(strength: np.ndarray, gap_limit: int) -> tuple[int | None, int | None]:
    """いちばん字の多い行から上下へ広げる。名前の札の字は本文より少ないので外れる。"""
    if strength.max() == 0:
        return None, None
    strong = np.where(strength >= strength.max() * 0.05)[0]
    peak = int(np.argmax(strength))
    top = bottom = peak
    for row in strong[strong > peak]:
        if row - bottom > gap_limit:
            break
        bottom = int(row)
    for row in strong[strong < peak][::-1]:
        if top - row > gap_limit:
            break
        top = int(row)
    return top, bottom + 1


def locate(reader: ClaudeReader, frames: list[np.ndarray], rule: GlyphRule) -> dict:
    # 字のいちばん多い画面を見せる。短い文の画面だと、下の行が範囲から外れる
    longest = max(frames, key=lambda f: int(glyph_mask(to_gray(f), rule).sum()))
    rough = ask_rough_band(reader, longest)
    band = refine_band(frames, rough, rule)
    return {"rough": list(rough), "band": list(band)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="locate")
    parser.add_argument("images", nargs="+", help="文の画面。長い文の画面を混ぜる")
    parser.add_argument("--costs", default="costs.jsonl")
    options = parser.parse_args(argv)
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in options.images]
    reader = ClaudeReader(CostLog(Path(options.costs)))
    print(json.dumps(locate(reader, frames, GlyphRule()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
