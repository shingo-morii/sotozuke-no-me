"""画面の計算。API を呼ばず、手元で速く回す。"""
from __future__ import annotations

import hashlib

import numpy as np

from .game_config import Box, GlyphRule


def crop(frame: np.ndarray, box: Box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return frame[y0:y1, x0:x1]


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB を明るさにする。整数の計算で速くする。"""
    r = rgb[..., 0].astype(np.uint16)
    g = rgb[..., 1].astype(np.uint16)
    b = rgb[..., 2].astype(np.uint16)
    return ((r * 77 + g * 150 + b * 29) >> 8).astype(np.uint8)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """真の画素を radius だけ広げる。"""
    wide = mask.copy()
    for s in range(1, radius + 1):
        wide[s:, :] |= mask[:-s, :]
        wide[:-s, :] |= mask[s:, :]
    tall = wide.copy()
    for s in range(1, radius + 1):
        wide[:, s:] |= tall[:, :-s]
        wide[:, :-s] |= tall[:, s:]
    return wide


def glyph_mask(gray: np.ndarray, rule: GlyphRule) -> np.ndarray:
    """明るい画素のうち、近くに暗い画素があるものを字の形と見なす。

    白の数を数えると明るい背景で崩れるので、字の縁を見る。
    """
    bright = gray >= rule.bright
    dark = gray <= rule.dark
    return bright & dilate(dark, rule.radius)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """2つの字の形の重なり。背景の揺れを許すため片方を1画素広げて比べる。"""
    na = int(a.sum())
    nb = int(b.sum())
    if na == 0 or nb == 0:
        return 0.0
    shared = int((a & dilate(b, 1)).sum())
    return min(1.0, shared / max(na, nb))


def new_pixels(current: np.ndarray, reference: np.ndarray) -> int:
    """reference に無い字の点の数。"""
    return int((current & ~dilate(reference, 1)).sum())


def small_gray(frame: np.ndarray, box: Box | None = None, step: int = 8) -> np.ndarray:
    """粗く間引いた明るさ。全体の変化を安く見るときに使う。"""
    area = frame if box is None else crop(frame, box)
    return to_gray(area[::step, ::step])


def fingerprint(frame: np.ndarray, box: Box | None = None) -> str:
    """同じ絵かどうかを見る指紋。多少の揺れは同じになるように粗くしてから取る。"""
    tiny = small_gray(frame, box, step=16)
    quantized = (tiny >> 4).tobytes()
    return hashlib.sha1(quantized).hexdigest()[:16]
