"""テスト用の合成の画面を作る。画像のファイルはコミットせず、実行のときに描く。

作りの違う2種類の見た目を用意する。
- style "a": 絵の上に、白い字を中央に置くゲーム。窓の枠は無い。名前の札は横の線
- style "b": 下に半透明の窓があり、字を左にそろえるゲーム。名前は窓の上の小さな札
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


MOTION = 0.15  # 背景の動きの速さ


def font_path() -> str | None:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


@lru_cache(maxsize=8)
def font(size: int) -> ImageFont.FreeTypeFont:
    path = font_path()
    if path is None:
        raise RuntimeError("日本語のフォントがありません")
    return ImageFont.truetype(path, size)


@dataclass(frozen=True)
class Style:
    name: str
    size: tuple[int, int]
    band: tuple[int, int, int, int]      # 本文の帯
    text_anchor: str                     # "center" か "left"
    text_origin: tuple[int, int]
    font_px: int
    line_px: int
    window: tuple[int, int, int, int] | None
    tag_kind: str                        # "hline" か "box"
    tag_y: int
    picture: tuple[int, int, int, int]
    choice_x: tuple[int, int]
    choice_row_h: int
    choice_gap: int
    choice_center_y: int
    hint_box: tuple[int, int, int, int]
    scene_box: tuple[int, int, int, int]


STYLE_A = Style(
    name="a", size=(1024, 768), band=(100, 540, 924, 660), text_anchor="center",
    text_origin=(512, 572), font_px=22, line_px=34, window=None,
    tag_kind="hline", tag_y=553, picture=(0, 96, 1024, 672),
    choice_x=(30, 620), choice_row_h=34, choice_gap=7, choice_center_y=460,
    hint_box=(640, 648, 1020, 670), scene_box=(0, 110, 1024, 420),
)

STYLE_B = Style(
    name="b", size=(1280, 720), band=(180, 540, 1100, 700), text_anchor="left",
    text_origin=(200, 556), font_px=28, line_px=42, window=(160, 520, 1120, 710),
    tag_kind="box", tag_y=488, picture=(0, 0, 1280, 720),
    choice_x=(340, 940), choice_row_h=58, choice_gap=14, choice_center_y=320,
    hint_box=(20, 12, 420, 44), scene_box=(0, 0, 1280, 420),
)

STYLES = {"a": STYLE_A, "b": STYLE_B}


def background(style: Style, scene: int, t: float = 0.0, moving: bool = True) -> np.ndarray:
    """場面ごとに違う、なめらかに動く背景を作る。"""
    # 描くのが重いので、動きは0.5秒きざみにして覚えておく
    return _background(style, scene, round(t * 2) / 2 * MOTION if moving else 0.0)


@lru_cache(maxsize=256)
def _background(style: Style, scene: int, phase: float) -> np.ndarray:
    w, h = style.size
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    seed = scene * 37 % 97 / 97.0
    r = 110 + 70 * np.sin(x / (160 + 40 * seed) + 6 * seed + phase)
    g = 120 + 60 * np.sin(y / (120 + 50 * seed) + 11 * seed)
    b = 130 + 80 * np.sin((x + y) / (240 + 60 * seed) + 3 * seed - phase)
    img = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
    x0, y0, x1, y1 = style.picture
    out = np.zeros((h, w, 3), np.uint8)
    out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    out.setflags(write=False)
    return out


def _draw_outlined(draw: ImageDraw.ImageDraw, xy, text, fnt, fill=(255, 255, 255),
                   stroke=(20, 20, 30), anchor="la", width=2):
    draw.text(xy, text, font=fnt, fill=fill, stroke_width=width,
              stroke_fill=stroke, anchor=anchor)


@dataclass
class View:
    """ある時刻の画面の中身。"""
    scene: int = 0
    text: str = ""
    visible_chars: int | None = None
    name: str = ""
    choices: list[str] = field(default_factory=list)
    selected: int = 0
    hint: str = ""
    dark: float = 0.0        # 0 で絵、1 で真っ黒
    white: float = 0.0
    menu_items: list[str] = field(default_factory=list)
    menu_title: str = ""
    confirm: str = ""
    card: str = ""


def wrap_lines(text: str, per_line: int) -> list[str]:
    return [text[i:i + per_line] for i in range(0, len(text), per_line)] or [""]


def render(style: Style, view: View, t: float = 0.0) -> np.ndarray:
    """View を RGB の配列に描く。"""
    if view.card:
        base = np.full((style.size[1], style.size[0], 3), (214, 150, 170), np.uint8)
    else:
        base = background(style, view.scene, t)
    image = Image.fromarray(base)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if view.menu_items:
        _draw_menu(style, draw, view)
    elif style.window is not None and (view.text or view.name):
        draw.rectangle(style.window, fill=(10, 16, 40, 170))

    if view.text:
        _draw_text(style, draw, view)
    if view.name:
        _draw_name(style, draw, view)
    if view.choices:
        _draw_choices(style, draw, view)
    if view.hint:
        _draw_hint(style, draw, view.hint)
    if view.confirm:
        _draw_confirm(style, draw, view)
    if view.card:
        w, h = style.size
        draw.rectangle((w // 5, h // 3, w * 4 // 5, h // 2), fill=(240, 235, 235, 255))
        draw.text((w // 2, h * 5 // 12), view.card, font=font(36), fill=(50, 40, 40), anchor="mm")
        _draw_hint(style, draw, "日付と時刻")

    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    out = np.asarray(image, dtype=np.uint8).copy()
    if view.dark > 0:
        out = (out * (1 - view.dark)).astype(np.uint8)
    if view.white > 0:
        out = (out + (255 - out) * view.white).astype(np.uint8)
    return out


def _draw_text(style: Style, draw, view: View) -> None:
    text = view.text if view.visible_chars is None else view.text[:view.visible_chars]
    per_line = 30 if style.name == "a" else 26
    full_lines = wrap_lines(view.text, per_line)
    lines = wrap_lines(text, per_line)
    fnt = font(style.font_px)
    x, y = style.text_origin
    for i, line in enumerate(lines):
        if style.text_anchor == "center":
            # 字の位置が途中で動かないよう、出きったときの行で位置を決める
            width = draw.textlength(full_lines[i], font=fnt)
            xy = (x - width / 2, y + i * style.line_px)
            _draw_outlined(draw, xy, line, fnt)
        else:
            _draw_outlined(draw, (x, y + i * style.line_px), line, fnt)


def _draw_name(style: Style, draw, view: View) -> None:
    w = style.size[0]
    if style.tag_kind == "hline":
        y = style.tag_y
        draw.line((36, y, w - 36, y), fill=(255, 255, 255, 240), width=2)
        _draw_outlined(draw, (w // 2, y - 16), view.name, font(18), anchor="mm")
    else:
        draw.rectangle((170, 492, 420, 526), fill=(200, 60, 90, 255))
        draw.text((295, 509), view.name, font=font(22), fill=(255, 255, 255), anchor="mm")


def choice_rows(style: Style, count: int) -> list[tuple[int, int, int, int]]:
    pitch = style.choice_row_h + style.choice_gap
    top = style.choice_center_y - (count * pitch - style.choice_gap) // 2
    x0, x1 = style.choice_x
    return [(x0, top + i * pitch, x1, top + i * pitch + style.choice_row_h)
            for i in range(count)]


def _draw_choices(style: Style, draw, view: View) -> None:
    rows = choice_rows(style, len(view.choices))
    for i, (box, label) in enumerate(zip(rows, view.choices)):
        selected = i == view.selected
        fill = (240, 170, 210, 235) if selected else (30, 30, 40, 200)
        draw.rectangle(box, fill=fill)
        color = (60, 20, 40) if selected else (255, 255, 255)
        draw.text((box[0] + 24, (box[1] + box[3]) // 2), label,
                  font=font(style.font_px), fill=color, anchor="lm")


def _draw_hint(style: Style, draw, hint: str) -> None:
    x0, y0, x1, y1 = style.hint_box
    draw.rectangle((x0, y0, x1, y1), fill=(40, 34, 38, 230))
    draw.text((x1 - 6, (y0 + y1) // 2), hint, font=font(18), fill=(255, 255, 255), anchor="rm")


def _draw_menu(style: Style, draw, view: View) -> None:
    w, h = style.size
    draw.rectangle((0, 0, w, h), fill=(250, 240, 245, 245))
    draw.text((w // 2, 130), view.menu_title, font=font(30), fill=(70, 50, 80), anchor="mm")
    for i, item in enumerate(view.menu_items):
        y = 200 + i * 56
        selected = i == view.selected
        if selected:
            draw.rectangle((w // 4, y - 22, w * 3 // 4, y + 22), fill=(80, 70, 120, 255))
        draw.text((w // 2, y), item, font=font(26),
                  fill=(255, 255, 255) if selected else (70, 50, 80), anchor="mm")


def _draw_confirm(style: Style, draw, view: View) -> None:
    w, h = style.size
    box = (w // 4, h * 3 // 8, w * 3 // 4, h * 5 // 8)
    draw.rectangle(box, fill=(255, 255, 255, 255), outline=(230, 40, 90, 255), width=3)
    draw.text((w // 2, h * 7 // 16), view.confirm, font=font(24), fill=(60, 40, 50), anchor="mm")
    labels = ("はい", "いいえ")
    for i, label in enumerate(labels):
        cx = w // 2 + (i * 2 - 1) * 90
        selected = i == view.selected
        if selected:
            draw.rectangle((cx - 60, h * 9 // 16 - 20, cx + 60, h * 9 // 16 + 20), fill=(230, 40, 90, 255))
        draw.text((cx, h * 9 // 16), label, font=font(24),
                  fill=(255, 255, 255) if selected else (60, 40, 50), anchor="mm")


def write_game(style: Style, folder: Path, voice_rule: str = "tag") -> Path:
    """合成のゲームの設定と見本を folder に書き、設定のパスを返す。"""
    folder.mkdir(parents=True, exist_ok=True)
    hints = {
        "choice": ("選択肢", "choice", "上下 選択  決定  履歴  設定"),
        "map": ("行き先", "map", "上下左右 選択  決定  戻る  設定"),
        "walk": ("歩く", "walk", "前後左右 移動  話す  地図  設定"),
        "save": ("セーブ", "menu", "上下左右 選択  決定  戻る"),
        "log": ("履歴", "backlog", "上下 送り  再生  閉じる"),
        "card": ("札", "card", "日付と時刻"),
    }
    screens = []
    x0, y0, x1, y1 = style.hint_box
    for key, (name, kind, hint) in hints.items():
        frame = render(style, View(scene=0, hint=hint))
        patch = frame[y0:y1, x0:x1]
        file = f"tpl_{key}.png"
        Image.fromarray(patch).save(folder / file)
        screens.append({"name": name, "kind": kind, "box": list(style.hint_box),
                        "template": file, "threshold": 0.7})

    # 確認の窓は字でなく枠の絵で見分ける。下の画面より先に見るので前に置く
    w, h = style.size
    box = (w // 4, h * 3 // 8, w // 4 + 60, h * 3 // 8 + 20)
    frame = render(style, View(scene=0, confirm="確認の文"))
    Image.fromarray(frame[box[1]:box[3], box[0]:box[2]]).save(folder / "tpl_confirm.png")
    screens.insert(0, {"name": "確認の窓", "kind": "confirm", "box": list(box),
                       "template": "tpl_confirm.png", "threshold": 0.97, "match": "pixel"})

    rows = {str(n): [list(b) for b in choice_rows(style, n)] for n in (2, 3, 4, 5)}
    all_rows = choice_rows(style, 5)
    band = list(style.band)
    data = {
        "name": f"synth_{style.name}",
        "size": list(style.size),
        "window_title": "synth",
        "process": "synth.exe",
        "voice_rule": voice_rule,
        "glyph": {"bright": 190, "dark": 110, "radius": 2},
        "text": {"band": band, "stable_s": 0.3},
        "scene": {"box": list(style.scene_box)},
        "screens": screens,
        "choice": {"box": [style.choice_x[0], all_rows[0][1], style.choice_x[1], all_rows[-1][3]],
                   "rows_by_count": rows},
        "keys": {"advance": ["enter"], "cursor": ["up", "down"]},
    }
    if style.tag_kind == "hline":
        data["tag"] = {"kind": "hline", "y": style.tag_y, "x0": 36,
                       "x1": style.size[0] - 36, "min_cols": 600}
    else:
        data["tag"] = {"kind": "glyph", "box": [170, 492, 420, 526], "min_cols": 30}
    path = folder / f"synth_{style.name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
