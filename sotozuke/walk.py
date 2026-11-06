"""歩く画面の道案内。目当ての人の向きと近さ、回した角度、進めないことを、声と音で伝える。

道案内はゲームに何も送らない。利用者が押しているキーを見て、押している長さから回した角度を見積もる。
回る速さと画角はゲームごとの設定から読む。目当ての人は、地図で決めたときの画面から読み手に聞いておく。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .game_config import WalkRule
from .image_ops import small_gray

if TYPE_CHECKING:
    from .director import Director

CHECK_AFTER_S = 0.6     # キーを全部離してからこれだけたったら、目当ての人を探す
BUMP_AFTER_S = 0.4      # 歩くキーを押しているのに画面がこれだけ変わらなければ、進めないと知らせる
TARGET_WAIT_S = 6.0     # 歩く画面に入ったとき、目当ての人の見た目が届くのをこれだけ待つ
FRONT_DEG = 8.0         # これより小さい角度は正面と言う
LAST_SEEN_S = 20.0      # 近くで見えた人が見えなくなったら、この間は見えた向きを言い添える
DISTANCES = ("すぐ前", "近い", "中くらい", "遠い")


def bearing_text(x: float, view_deg: float) -> str:
    """画面の横の位置（左端0、右端100）を、向きの言葉にする。"""
    offset = (x - 50.0) / 50.0
    side = "右" if offset > 0 else "左"
    if view_deg > 0:
        degrees = abs(offset) * view_deg / 2
        if degrees < FRONT_DEG:
            return "正面"
        return f"{side}へ{max(10, int(round(degrees / 5.0)) * 5)}度"
    if abs(offset) < 0.2:
        return "正面"
    return f"やや{side}" if abs(offset) < 0.6 else f"{side}の端"


def bearing_deg(x: float, view_deg: float) -> float:
    """画面の横の位置を、画面の中央からの角度にする。右が正。"""
    return (min(max(x, 0.0), 100.0) - 50.0) / 50.0 * view_deg / 2


def direction_text(degrees: float) -> str:
    """今の向きから見た方向。"""
    degrees = (degrees + 180.0) % 360.0 - 180.0
    if abs(degrees) < 15:
        return "正面"
    if abs(degrees) > 150:
        return "後ろ"
    side = "右" if degrees > 0 else "左"
    return f"{side}へ{int(round(abs(degrees) / 15.0)) * 15}度"


def heading_text(degrees: float) -> str:
    """初めの向きからどれだけ回ったか。"""
    degrees = (degrees + 180.0) % 360.0 - 180.0
    if abs(degrees) < 15:
        return "初めの向き"
    if abs(degrees) > 165:
        return "初めの向きの逆"
    side = "右" if degrees > 0 else "左"
    return f"初めの向きから{side}へ{int(round(abs(degrees) / 15.0)) * 15}度"


def target_text(data: dict, view_deg: float) -> str:
    """読み手の答えから、目当ての人の向きと近さの文を作る。"""
    if not data.get("found"):
        return "見えません。"
    try:
        x = float(data.get("x", 50))
    except (TypeError, ValueError):
        x = 50.0
    where = bearing_text(min(max(x, 0.0), 100.0), view_deg)
    distance = str(data.get("distance", ""))
    distance = distance if distance in DISTANCES else ""
    text = f"{where}、{distance}。" if distance else f"{where}。"
    if where == "正面" and distance == "すぐ前":
        text += "話しかけられそうです。"
    return text


@dataclass
class Turn:
    sign: int          # 右は1、左は-1
    start: float
    ticks: int = 0


class WalkGuide:
    """判断の中心から呼ばれる。結果は判断の中心の世代で古いものを捨てる。"""

    def __init__(self, rule: WalkRule, director: "Director"):
        self.rule = rule
        self.d = director
        self.s = director.s
        self.target = ""
        self.target_ready = threading.Event()
        self.target_ready.set()
        self.hint_said = False
        self.heading = 0.0
        self.held: set[str] = set()
        self.turn: Turn | None = None
        self.check_at: float | None = None
        self.previous: np.ndarray | None = None
        self.still_since: float | None = None
        self.bumped = False
        self.last_seen: tuple[float, str, float] | None = None  # 見えた向き（初めの向きから）、近さ、時刻

    # 地図

    def remember_target(self, frame: np.ndarray | None) -> None:
        """地図で決めたとき、選ばれている人の見た目を聞いておく。"""
        if frame is None:
            return
        self.target = ""
        self.target_ready.clear()
        prompt = self.rule.target_prompt

        def work():
            try:
                self.target = self.s.reader.read_target(frame, prompt).text
            finally:
                self.target_ready.set()
            return self.target

        def done(value, error):
            self.s.log("walk_target", text=value or "", error=repr(error) if error else "")

        self.s.tasks.submit(work, done)

    # 歩く画面に入ったとき

    def enter(self) -> None:
        self.last_seen = None
        self.heading = 0.0
        self.turn = None
        self.check_at = None
        self.previous = None
        self.still_since = None
        self.bumped = False

    def entry_text(self, frame: np.ndarray) -> str:
        """入ったときに言う文の続き。読み手の仕事の流れの中で呼ぶ。"""
        parts = []
        if self.rule.hint and not self.hint_said:
            self.hint_said = True
            parts.append(self.rule.hint)
        self.target_ready.wait(TARGET_WAIT_S)
        if self.target:
            data, _ = self.s.reader.find_target(frame, self.target)
            parts.append("目当ての人は" + self.words(data, 0.0, self.s.clock()))
        return "".join(parts)

    def words(self, data: dict, heading: float, now: float) -> str:
        """目当ての人の向きと近さ。近くで見えていた人が見えなくなったら、見えた向きを言い添える。

        近づくと、相手は画面の端で切れたり、手前の主人公の陰に入ったりして見えなくなる。
        """
        text = target_text(data, self.rule.view_deg)
        if self.rule.turn_deg_per_s <= 0 or self.rule.view_deg <= 0:
            return text
        if data.get("found"):
            try:
                x = float(data.get("x", 50))
            except (TypeError, ValueError):
                x = 50.0
            self.last_seen = (heading + bearing_deg(x, self.rule.view_deg), str(data.get("distance", "")), now)
            return text
        seen = self.last_seen
        if seen and now - seen[2] <= LAST_SEEN_S and seen[1] in ("すぐ前", "近い", "中くらい"):
            where = direction_text(seen[0] - heading)
            text = text.rstrip("。") + f"。さっきは{where}の{seen[1]}に見えました。"
        return text

    # キー

    def key_down(self, name: str, now: float) -> None:
        if name in self.held:
            return
        self.held.add(name)
        self.check_at = None
        self.d._next_generation()  # 前の向きの答えはもう古い
        self.d._stop_sound()
        if name == self.rule.turn_right or name == self.rule.turn_left:
            self._end_turn(now)
            self.turn = Turn(1 if name == self.rule.turn_right else -1, now)
        else:
            self.still_since = None
            self.bumped = False

    def key_up(self, name: str, now: float) -> None:
        if name not in self.held:
            return
        self.held.discard(name)
        if self.turn and name in (self.rule.turn_right, self.rule.turn_left):
            self._end_turn(now)
        if not self.held:
            self.check_at = now + CHECK_AFTER_S

    def _end_turn(self, now: float) -> None:
        turn, self.turn = self.turn, None
        if turn is None or self.rule.turn_deg_per_s <= 0:
            return
        turned = (now - turn.start) * self.rule.turn_deg_per_s
        self.heading += turn.sign * turned
        self.s.log("walk_turn", degrees=round(turn.sign * turned, 1), heading=round(self.heading, 1))

    def moving(self) -> bool:
        return any(name in self.rule.move_keys for name in self.held)

    # 1コマごと

    def step(self, frame: np.ndarray, now: float) -> None:
        turn = self.turn
        if turn and self.rule.turn_deg_per_s > 0 and self.rule.tick_deg > 0:
            ticks = int((now - turn.start) * self.rule.turn_deg_per_s / self.rule.tick_deg + 1e-6)
            if ticks > turn.ticks:
                turn.ticks = ticks
                big = ticks * self.rule.tick_deg % 90 < 1e-6
                self.d._tone_now(self.d.walk_tones["tick90" if big else "tick"], "walk_tick")
        self._watch_bump(frame, now)
        if self.check_at is not None and now >= self.check_at and not self.held:
            self.check_at = None
            if self.target:
                self.check(frame, around=False)

    def _watch_bump(self, frame: np.ndarray, now: float) -> None:
        sample = small_gray(frame)
        previous, self.previous = self.previous, sample
        if not self.moving() or previous is None or self.turn:
            self.still_since = None
            return
        diff = float(np.abs(sample.astype(np.float32) - previous.astype(np.float32)).mean())
        if diff >= self.rule.bump_diff:
            self.still_since = None
            self.bumped = False
            return
        if self.still_since is None:
            self.still_since = now
        elif not self.bumped and now - self.still_since >= BUMP_AFTER_S:
            self.bumped = True
            self.s.log("walk_bump", diff=round(diff, 2))
            self.d._tone_now(self.d.walk_tones["bump"], "walk_bump")

    # 目当ての人を探す

    def check(self, frame: np.ndarray, around: bool) -> None:
        """目当ての人の向きと近さを言う。around なら周りの様子と今の向きも言う。"""
        d = self.d
        self.check_at = None  # 説明キーで聞いたら、離したあとの自動の確認はしない
        generation = d._next_generation()
        target = self.target
        heading = self.heading if self.rule.turn_deg_per_s > 0 else None
        now = self.s.clock()

        def work():
            if target:
                data, _ = self.s.reader.find_target(frame, target)
                found = self.words(data, heading or 0.0, now)
                text = "目当ての人は" + found if around else found
                if around:
                    near = str(data.get("around", "")).strip()
                    text += near if near.endswith("。") or not near else near + "。"
            else:
                from . import reader as prompts
                text = self.s.reader.read_plain(frame, prompts.PROMPT_WALK_VIEW, "walk_view").text
            if around and heading is not None:
                text += f"いまは{heading_text(heading)}。"
            return (text, self.s.speech(text)) if text else None

        def done(value, error):
            if generation != d.generation or d.watcher.kind != "walk":
                # 場面が始まって歩く画面を出たあとに、古い向きを言わない
                self.s.log("drop_stale", what="walk_check", generation=generation)
                return
            if error is not None or value is None:
                self.s.log("fail", reason=repr(error) if error else "empty", what="walk_check")
                d._play_tone(self.s.fail_tone, "fail")
                return
            d._say(value[0], value[1], "walk" if not around else "description")

        self.s.log("walk_check", around=around, target=target)
        self.s.tasks.submit(work, done)
