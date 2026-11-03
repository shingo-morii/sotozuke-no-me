"""見張り。API を呼ばず、10コマ毎秒の画面から出来事を取り出す。

画面の種類を先に決める。名前の札の線は選択肢や履歴の画面でも出るので、
案内の字の形で種類を決めてから、文の画面でだけ札を見る。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .game_config import GameConfig, ScreenRule
from .image_ops import crop, dilate, glyph_mask, new_pixels, overlap, small_gray, to_gray

TEXT_KIND = "text"
KIND_CONFIRM_FRAMES = 2
SETTLE_FRAMES = 2
SETTLE_CELL_DIFF = 25
SETTLE_CELLS = 3
FORCE_SETTLE_S = 1.2  # 背景が動き続ける画面でも、この時間で読む
CURSOR_WAIT_S = 0.35  # カーソルのあと、選択の光が移り終わるまで読まない
FINE_CELL_DIFF = 6    # カーソルのあとは、少しずつ移る光も動きと見る
FORCE_JUDGE_S = 1.2   # キーのあと字の形が揺れて止まらなくても、この時間で文を判定する
FORCE_AGAIN_S = 0.6
RECENT_FRAMES = 3
AGAIN_WITHOUT_KEY_S = 1.0  # キーなしで次の文と見なすには、これだけ止まっていること


@dataclass
class TextReady:
    """文が出きった。"""
    t: float
    band: np.ndarray
    has_tag: bool
    frame: np.ndarray = field(repr=False, default=None)


@dataclass
class KindChanged:
    t: float
    old: str
    new: str
    screen: ScreenRule | None
    frame: np.ndarray = field(repr=False, default=None)


@dataclass
class SceneDawn:
    """暗転や白飛びが明け、絵が落ち着いた。"""
    t: float
    frame: np.ndarray = field(repr=False, default=None)
    kind: str = TEXT_KIND


@dataclass
class ScreenSettled:
    """文の画面でないときに、画面の変化が落ち着いた。"""
    t: float
    kind: str
    screen: ScreenRule | None
    frame: np.ndarray = field(repr=False, default=None)


class TextTracker:
    """本文の帯の字の形を見て、文が出きった瞬間を知らせる。"""

    def __init__(self, config: GameConfig):
        self.rule = config.text
        self.glyph = config.glyph
        self.reference: np.ndarray | None = None
        self.core: np.ndarray | None = None
        self.age = 0
        self.reference_t = 0.0
        self.judged = False
        self.last: np.ndarray | None = None
        self.emptied = False
        self.recent: deque[np.ndarray] = deque(maxlen=RECENT_FRAMES)
        self.key_pressed = False
        self.force_at: float | None = None
        self.fired_since_key = False
        self.empty_frames = 0

    def on_key(self) -> None:
        """キーが押されたら、そこから止まるのを待ち直す。連打の途中の文を読まないため。"""
        self.reference = None
        self.judged = False
        self.key_pressed = True
        self.fired_since_key = False

    def feed(self, band_rgb: np.ndarray, t: float) -> bool:
        mask = glyph_mask(to_gray(band_rgb), self.glyph)
        self.recent.append(mask)
        if self.key_pressed:
            self.key_pressed = False
            self.force_at = t + FORCE_JUDGE_S
        if int(mask.sum()) < self.rule.min_pixels:
            self.reference = None
            self.judged = False
            self.empty_frames += 1
            if self.empty_frames >= 2:  # 1コマだけ字が薄れたときは空と見なさない
                self.emptied = True
            if self.force_at is not None and t >= self.force_at:
                self.force_at = t + FORCE_AGAIN_S
            return False
        self.empty_frames = 0
        if self.reference is None or self._changed(mask):
            self.reference = self.core = mask
            self.age = 0
            self.reference_t = t
            self.judged = False
            return self._forced(t)
        self.age += 1
        self.core = self.core & dilate(mask, 1)
        stable_s = self.rule.stable_s
        if not self.rule.typed and self.fired_since_key:
            stable_s = AGAIN_WITHOUT_KEY_S
        if self.judged or t - self.reference_t < stable_s:
            return self._forced(t)
        self.judged = True
        final = mask if self.rule.typed else self.core
        if int(final.sum()) < self.rule.min_pixels:
            self.emptied = True  # 見続けて残らなかった点は背景のゆらぎ。字は無かった
            return False
        is_new = self._judge(final)
        if is_new:
            self.force_at = None  # キーの直後は古い文がまだ残っているので、見つけるまで保険を残す
        return is_new

    def shows_last(self) -> bool:
        """今の字が、最後に新しい文と判定した字と同じか。"""
        if self.last is None or self.reference is None or self.core is None:
            return False
        return int(self.core.sum()) >= self.rule.min_pixels and overlap(self.core, self.last) >= self.rule.new_overlap

    def _forced(self, t: float) -> bool:
        """動く背景で字の形が止まらないときの保険。文はキーのあとにしか変わらないので、一度だけ判定する。"""
        if self.rule.typed or self.force_at is None or t < self.force_at:
            return False
        # 字がキーから遅れて出ることもあるので、文が見つかるまで見直す
        self.force_at = t + FORCE_AGAIN_S
        self.judged = True
        steady = np.logical_and.reduce(list(self.recent))
        if int(steady.sum()) < self.rule.min_pixels:
            return False
        is_new = self._judge(steady)
        if is_new:
            self.force_at = None
        return is_new

    def _judge(self, mask: np.ndarray) -> bool:
        if not self.rule.typed and self.fired_since_key:
            # キーなしで次の文と見なすのは、字がほとんど入れ替わったときだけ
            is_new = self.emptied or overlap(mask, self.last) < 0.4
        else:
            grown = 0.15 if self.rule.typed else 0.25
            is_new = (self.last is None or self.emptied
                      or overlap(mask, self.last) < self.rule.new_overlap
                      # 左にそろえたゲームでは、前の文に字が足されただけの文もある
                      or new_pixels(mask, self.last) > int(self.last.sum() * grown))
        if is_new:
            self.last = mask
            self.emptied = False
            self.fired_since_key = True
        return is_new

    def flush(self) -> bool:
        """止まりきる前に別の画面が重なったとき、見ていた文が新しい文かを決める。"""
        if self.reference is None or self.judged:
            return False
        self.judged = True
        return self._judge(self.core)

    def _changed(self, mask: np.ndarray) -> bool:
        if self.rule.typed:
            size = int(self.reference.sum())
            added = new_pixels(mask, self.reference)
            kept = size - new_pixels(self.reference, mask)
            return added > max(30, int(size * 0.02)) or kept < size * 0.75
        # 文字が一度に出るゲーム。背景が動くと字の形が揺れるので、見続けても残る点を字と見なし、
        # その点が消えたら別の文と判断する
        if self.age == 0:
            return False
        size = int(self.core.sum())
        kept = size - new_pixels(self.core, mask)
        # 見続けて残った点はほとんど消えない。同じ位置の短い文どうしも見分けられるよう高くする
        limit = 0.7 if self.age >= 2 else 0.5
        return kept < size * limit or new_pixels(mask, self.core) > max(1000, size)


class SceneTracker:
    """画面の上の部分が真っ黒か真っ白を通り、そのあと落ち着いた瞬間を知らせる。"""

    def __init__(self, config: GameConfig):
        self.rule = config.scene
        self.state = "lit"
        self.extreme_since = 0.0
        self.previous: np.ndarray | None = None
        self.quiet = 0

    def feed(self, frame: np.ndarray, t: float) -> bool:
        rule = self.rule
        sample = small_gray(frame, rule.box)
        level = float(sample.mean())
        extreme = level < rule.dark_below or level > rule.bright_above
        previous, self.previous = self.previous, sample

        if self.state == "lit":
            if extreme:
                self.state = "extreme"
                self.extreme_since = t
            return False
        if self.state == "extreme":
            if not extreme:
                if t - self.extreme_since >= rule.min_dark_s:
                    self.state = "settling"
                    self.quiet = 0
                else:
                    self.state = "lit"
            return False
        # 明けたあと、絵が落ち着くのを待つ
        if extreme:
            self.state = "extreme"
            return False
        diff = float(np.abs(sample.astype(np.int16) - previous.astype(np.int16)).mean())
        self.quiet = self.quiet + 1 if diff < rule.settle_diff else 0
        if self.quiet >= rule.settle_frames:
            self.state = "lit"
            return True
        return False


class ScreenClassifier:
    """案内の字の形を見本と比べて、画面の種類を決める。"""

    def __init__(self, config: GameConfig):
        self.glyph = config.glyph
        self.screens = [(s, self._template(s)) for s in config.screens]

    def _template(self, screen: ScreenRule) -> np.ndarray:
        gray = to_gray(np.asarray(Image.open(screen.template).convert("RGB")))
        if screen.match == "pixel":
            return gray
        mask = glyph_mask(gray, self.glyph)
        if not mask.any():
            raise ValueError(f"見本に字がありません: {screen.template}")
        return mask

    def _score(self, frame: np.ndarray, screen: ScreenRule, template: np.ndarray, masks: dict) -> float:
        if screen.match == "pixel":
            gray = to_gray(crop(frame, screen.box)).astype(np.int16)
            return 1.0 - float(np.abs(gray - template).mean()) / 255.0
        mask = masks.get(screen.box)
        if mask is None:
            mask = glyph_mask(to_gray(crop(frame, screen.box)), self.glyph)
            masks[screen.box] = mask
        return overlap(mask, template)

    def scores(self, frame: np.ndarray) -> list[tuple[ScreenRule, float]]:
        masks: dict = {}
        return [(s, self._score(frame, s, t, masks)) for s, t in self.screens]

    def classify(self, frame: np.ndarray) -> ScreenRule | None:
        """基準を超えたもののうち、いちばん重なる見本を選ぶ。似た案内の画面を取り違えないため。

        確認の窓のように重なる画面は、下の画面より先に見る。設定の並びで pixel を前に置く。
        """
        best = None
        best_score = 0.0
        masks: dict = {}
        for screen, template in self.screens:
            score = self._score(frame, screen, template, masks)
            if score >= screen.threshold:
                if screen.match == "pixel":
                    return screen
                if score > best_score:
                    best, best_score = screen, score
        return best


def has_name_tag(frame: np.ndarray, config: GameConfig) -> bool:
    """名前の札があるか。文の画面でだけ呼ぶ。"""
    tag = config.tag
    if tag is None:
        return False
    if tag.kind == "hline":
        return _has_line(frame, tag)
    if tag.kind == "glyph":
        return _has_name_glyphs(frame, tag, config)
    if tag.kind == "either":
        # 線は明るい背景で消え、名前の字は短い名前で少なくなる。どちらかが見えれば札がある
        return _has_line(frame, tag) or _has_name_glyphs(frame, tag, config)
    raise ValueError(f"札の見分け方が違います: {tag.kind}")


def _has_line(frame: np.ndarray, tag) -> bool:
    rows = to_gray(frame[tag.y - 3:tag.y + 4, tag.x0:tag.x1]).astype(np.int16)
    line = rows[2:5].max(axis=0)
    around = (rows[0] + rows[6]) / 2
    return int((line - around >= tag.contrast).sum()) >= tag.min_cols


def _has_name_glyphs(frame: np.ndarray, tag, config: GameConfig) -> bool:
    mask = glyph_mask(to_gray(crop(frame, tag.box)), config.glyph)
    return int(mask.sum()) >= tag.min_glyphs


class Watcher:
    """1コマごとに呼ぶ。出来事の一覧を返す。"""

    def __init__(self, config: GameConfig):
        self.config = config
        self.text = TextTracker(config)
        self.scene = SceneTracker(config) if config.scene else None
        self.classifier = ScreenClassifier(config)
        self.kind = TEXT_KIND
        self.screen: ScreenRule | None = None
        self.pending: ScreenRule | None | str = "none"
        self.pending_count = 0
        self.settle_previous: np.ndarray | None = None
        self.settle_count = 0
        self.settled_sample: np.ndarray | None = None
        self.force_at: float | None = None
        self.expecting = False
        self.cursor_moved = False
        self.cursor_until: float | None = None

    def on_key(self) -> None:
        self.text.on_key()
        self.expecting = True

    def expect_change(self) -> None:
        """カーソルが動いたので、画面が落ち着かなくても読む。"""
        self.expecting = True
        self.cursor_moved = True

    def settle_again(self) -> None:
        """読めなかったので、画面が変わっていなくてももう一度知らせる。"""
        self.settled_sample = None
        self.settle_count = 0
        self.expecting = True

    def feed(self, frame: np.ndarray, t: float) -> list:
        events: list = []
        screen = self.classifier.classify(frame)
        new_kind = screen.kind if screen else TEXT_KIND
        if new_kind != self.kind or (screen and self.screen and screen.name != self.screen.name):
            if self.pending == (screen.name if screen else TEXT_KIND):
                self.pending_count += 1
            else:
                self.pending = screen.name if screen else TEXT_KIND
                self.pending_count = 1
            if self.pending_count >= KIND_CONFIRM_FRAMES:
                # 文のすぐあとに選択肢が重なると、文が止まりきらない。読み落とさないように知らせる
                if self.kind == TEXT_KIND and new_kind == "choice" and self.text.flush():
                    band = crop(frame, self.config.text.band)
                    events.append(TextReady(t, band.copy(), has_name_tag(frame, self.config), frame))
                events.append(KindChanged(t, self.kind, new_kind, screen, frame))
                self.kind = new_kind
                self.screen = screen
                self.settled_sample = None
                self.settle_count = 0
                self.expecting = True
        else:
            self.pending = "none"
            self.pending_count = 0

        if self.scene is not None and self.scene.feed(frame, t):
            events.append(SceneDawn(t, frame, self.kind))

        if self.kind == TEXT_KIND:
            # 種類が確定する前のコマも、文の画面でなければ本文として見ない
            if screen is None:
                band = crop(frame, self.config.text.band)
                if self.text.feed(band, t):
                    events.append(TextReady(t, band.copy(), has_name_tag(frame, self.config), frame))
        else:
            settled = self._screen_settled(frame, t)
            if settled:
                events.append(settled)
        return events

    def _screen_settled(self, frame: np.ndarray, t: float) -> ScreenSettled | None:
        sample = small_gray(frame)
        previous, self.settle_previous = self.settle_previous, sample
        if self.expecting:
            self.expecting = False
            self.force_at = t + FORCE_SETTLE_S
        if self.cursor_moved:
            # 光が少しずつ移ると1コマの差は小さく、移る途中の絵を読んでしまう。少し待ち、細かい差も見る
            self.cursor_moved = False
            self.cursor_until = t + CURSOR_WAIT_S
            self.settle_count = 0
        fine = self.cursor_until is not None
        if fine and t < self.cursor_until:
            return None
        forced = self.force_at is not None and t >= self.force_at
        if not forced:
            moving = previous is None or _changed(sample, previous, FINE_CELL_DIFF if fine else SETTLE_CELL_DIFF)
            if moving:
                self.settle_count = 0
                return None
            self.settle_count += 1
            if self.settle_count < SETTLE_FRAMES:
                return None
            if self.settled_sample is not None and not _changed(sample, self.settled_sample):
                self.force_at = None  # 動かしても画面が変わらなかった
                self.cursor_until = None
                return None
        self.force_at = None
        self.cursor_until = None
        self.settled_sample = sample
        return ScreenSettled(t, self.kind, self.screen, frame)


def _changed(a: np.ndarray, b: np.ndarray, cell_diff: int = SETTLE_CELL_DIFF) -> bool:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return int((diff > cell_diff).sum()) >= SETTLE_CELLS


__all__ = [
    "KindChanged", "SceneDawn", "ScreenClassifier", "ScreenSettled", "TEXT_KIND",
    "TextReady", "TextTracker", "Watcher", "dilate", "has_name_tag",
]
