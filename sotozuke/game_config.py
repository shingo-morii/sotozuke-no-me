"""ゲームごとの設定。座標と見分け方は全部ここから読む。本体に特定のゲームの値を書かない。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Box = tuple[int, int, int, int]  # x0, y0, x1, y1。x1 と y1 は含まない

VOICE_RULES = ("tag", "audio", "all")
SCREEN_KINDS = (
    "choice",       # 選択肢
    "map",          # 行き先の地図
    "walk",         # 歩く画面
    "menu",         # 項目を選ぶ画面（タイトル、設定、セーブ、ロード）
    "confirm",      # 確認の窓
    "card",         # 字だけを読む札（日付など）
    "backlog",      # 文字の履歴
)


class ConfigError(ValueError):
    pass


def _box(value: Any, label: str, size: tuple[int, int]) -> Box:
    if not (isinstance(value, (list, tuple)) and len(value) == 4):
        raise ConfigError(f"{label}: 位置は4つの数で書く")
    x0, y0, x1, y1 = (int(v) for v in value)
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ConfigError(f"{label}: 位置が画面の外です {value}")
    return (x0, y0, x1, y1)


@dataclass(frozen=True)
class GlyphRule:
    """字の形の見分け方。明るい画素のそばに暗い画素があるものを字と見なす。"""
    bright: int = 190
    dark: int = 110
    radius: int = 2


@dataclass(frozen=True)
class TextRule:
    band: Box
    stable_s: float = 0.3
    new_overlap: float = 0.7     # 前の文との重なりがこれ未満なら新しい文
    still_overlap: float = 0.93  # 同じ字が続いていると見なす重なり
    min_pixels: int = 40         # これより字の点が少なければ空


@dataclass(frozen=True)
class TagRule:
    """名前の札の見分け方。"""
    kind: str = "hline"          # 横の線
    y: int = 0
    x0: int = 0
    x1: int = 0
    min_cols: int = 0
    contrast: int = 25
    box: Box | None = None       # 札の字の位置を字の形で見るとき


@dataclass(frozen=True)
class SceneRule:
    box: Box
    dark_below: float = 3.0
    bright_above: float = 250.0
    min_dark_s: float = 0.3
    settle_frames: int = 3
    settle_diff: float = 6.0


@dataclass(frozen=True)
class ScreenRule:
    name: str
    kind: str
    box: Box
    template: Path
    threshold: float = 0.7
    prompt: str = ""


@dataclass(frozen=True)
class ChoiceRule:
    box: Box                         # 全項目が入る範囲
    rows_by_count: dict[int, list[Box]] = field(default_factory=dict)
    strip_label: bool = True

    def rows_for(self, count: int) -> list[Box]:
        """項目の数ごとの行の位置。知らない数は、知っている2つの数から伸ばして求める。"""
        if count in self.rows_by_count:
            return self.rows_by_count[count]
        known = sorted(self.rows_by_count)
        if len(known) < 2:
            return []
        a, b = known[0], known[-1]
        first_a, first_b = self.rows_by_count[a][0], self.rows_by_count[b][0]
        pitch = self.rows_by_count[a][1][1] - first_a[1]
        shift = (first_b[1] - first_a[1]) / (b - a)
        top = round(first_a[1] + shift * (count - a))
        height = first_a[3] - first_a[1]
        return [(first_a[0], top + i * pitch, first_a[2], top + i * pitch + height)
                for i in range(count)]


@dataclass(frozen=True)
class GameConfig:
    name: str
    size: tuple[int, int]
    window_title: str
    process: str
    glyph: GlyphRule
    text: TextRule
    voice_rule: str
    tag: TagRule | None
    scene: SceneRule | None
    screens: tuple[ScreenRule, ...]
    choice: ChoiceRule | None
    next_marker: Box | None
    advance_keys: tuple[str, ...]
    cursor_keys: tuple[str, ...]
    source: Path | None = None


def _load_template_path(base: Path, value: str, label: str) -> Path:
    path = (base / value).resolve()
    if not path.exists():
        raise ConfigError(f"{label}: 見本のファイルがありません {value}")
    return path


def parse_config(data: dict[str, Any], base: Path = Path(".")) -> GameConfig:
    try:
        size = (int(data["size"][0]), int(data["size"][1]))
        voice_rule = data.get("voice_rule", "all")
        if voice_rule not in VOICE_RULES:
            raise ConfigError(f"voice_rule は {VOICE_RULES} のどれか: {voice_rule}")

        glyph = GlyphRule(**data.get("glyph", {}))
        text_raw = dict(data["text"])
        text = TextRule(band=_box(text_raw.pop("band"), "text.band", size), **text_raw)

        tag = None
        if data.get("tag"):
            tag_raw = dict(data["tag"])
            if tag_raw.get("box") is not None:
                tag_raw["box"] = _box(tag_raw["box"], "tag.box", size)
            tag = TagRule(**tag_raw)
        if voice_rule == "tag" and tag is None:
            raise ConfigError("voice_rule が tag なのに tag の設定がありません")

        scene = None
        if data.get("scene"):
            scene_raw = dict(data["scene"])
            scene = SceneRule(box=_box(scene_raw.pop("box"), "scene.box", size), **scene_raw)

        screens = []
        for i, raw in enumerate(data.get("screens", [])):
            label = f"screens[{i}]"
            if raw["kind"] not in SCREEN_KINDS:
                raise ConfigError(f"{label}: 種類が違います {raw['kind']}")
            screens.append(ScreenRule(
                name=raw["name"], kind=raw["kind"],
                box=_box(raw["box"], label, size),
                template=_load_template_path(base, raw["template"], label),
                threshold=float(raw.get("threshold", 0.7)),
                prompt=raw.get("prompt", ""),
            ))

        choice = None
        if data.get("choice"):
            raw = data["choice"]
            rows = {
                int(count): [_box(b, f"choice.rows[{count}]", size) for b in boxes]
                for count, boxes in raw.get("rows_by_count", {}).items()
            }
            choice = ChoiceRule(
                box=_box(raw["box"], "choice.box", size), rows_by_count=rows,
                strip_label=bool(raw.get("strip_label", True)),
            )

        marker = data.get("next_marker")
        return GameConfig(
            name=data["name"], size=size,
            window_title=data.get("window_title", ""),
            process=data.get("process", ""),
            glyph=glyph, text=text, voice_rule=voice_rule, tag=tag, scene=scene,
            screens=tuple(screens), choice=choice,
            next_marker=_box(marker, "next_marker", size) if marker else None,
            advance_keys=tuple(data.get("keys", {}).get("advance", ["enter"])),
            cursor_keys=tuple(data.get("keys", {}).get("cursor", ["up", "down"])),
        )
    except KeyError as exc:
        raise ConfigError(f"設定の項目がありません: {exc}") from exc


def load_config(path: str | Path) -> GameConfig:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    config = parse_config(data, base=path.parent)
    return GameConfig(**{**config.__dict__, "source": path})
