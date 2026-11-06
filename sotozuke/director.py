"""判断の中心。読み上げのきまりをここに集める。

ノベルゲームは、利用者が押すまで先へ進まない。だから、利用者が押した先の今の画面をいつも優先し、
古いものは捨てる。順番待ちの列は作らない。読むのはいつも最新の1つだけ。

このクラスは1本の流れの中でだけ動く。読み手や声の結果は tasks を通して同じ流れに戻る。
古い結果を鳴らさないために、結果には作り始めたときの世代の番号を付けて、今の世代と比べる。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from . import reader as prompts
from .game_config import GameConfig, InputRule, MouseAction, ScreenRule
from .image_ops import crop, fingerprint, glyph_mask, small_gray, to_gray
from .inputs import Command
from .walk import WalkGuide
from .watcher import (KindChanged, SceneDawn, ScreenSettled, TEXT_KIND, TextReady, Watcher,
                      has_name_tag)

FAIL_AFTER_S = 3.0        # これだけ待っても読めなければ短い音で知らせる
DESCRIPTION_HOLD_S = 3.0  # 説明を待って文を止めておく長さ
SAME_PICTURE_DIFF = 10.0  # 粗い絵の差がこれ未満なら同じ絵
KEY_RECENT_S = 1.5        # キーのあとこの間に落ち着いた画面の変化は、利用者が動かしたものと見なす
CHECKING_EXTRA_S = 1.5    # 読み手が確認に入ったら、知らせる音をこれだけ待つ
TAG_RECHECK_S = 0.35      # 声ありと判定した文は、前の文の札が残っていただけかをこのあと見直す
SAME_AGAIN_S = 1.6        # キーのあとこの時間たっても字が同じなら、同じ文がまた出たと見なす


class Tasks(Protocol):
    def submit(self, work: Callable[[], Any], done: Callable[[Any, Exception | None], None]) -> None:
        ...

    def post(self, call: Callable[[], None]) -> None:
        ...


class Sound(Protocol):
    def play(self, audio: np.ndarray, on_end: Callable[[], None] | None = None) -> None:
        ...

    def stop(self) -> None:
        ...

    @property
    def playing(self) -> bool:
        ...


@dataclass
class Services:
    reader: Any
    speech: Callable[[str], np.ndarray]   # 文を音声の配列にする
    sound: Sound
    tasks: Tasks
    clock: Callable[[], float]
    log: Callable[..., None]
    chime: np.ndarray
    fail_tone: np.ndarray
    is_voice_playing: Callable[[], bool] = lambda: False
    walk_tones: dict[str, np.ndarray] = field(default_factory=dict)  # 歩く画面の刻みと、進めないときの音
    send: Callable[[MouseAction], None] = lambda action: None  # ゲームへ動きを送る。sender.py


@dataclass
class Pending:
    generation: int
    deadline: float
    failed: bool = False


@dataclass
class Scene:
    sample: np.ndarray
    frame: np.ndarray
    hold_until: float
    state: str = "generating"    # generating、speaking、done
    auto: bool = True            # 自動で読んでよいか
    text: str = ""
    audio: np.ndarray | None = None
    held: tuple[str, np.ndarray] | None = None
    discarded: bool = False


@dataclass
class ChoiceSession:
    frame: np.ndarray
    items: list[str] = field(default_factory=list)
    audios: list[np.ndarray] = field(default_factory=list)
    intro_text: str = ""
    intro_audio: np.ndarray | None = None
    rows: list[tuple[int, int, int, int]] = field(default_factory=list)
    selected: int | None = None
    candidate: int | None = None
    moved: bool = False
    ready: bool = False
    closing: bool = False


@dataclass
class ScreenState:
    screen: ScreenRule
    entered: bool = False
    title: str = ""
    text: str = ""
    audio: np.ndarray | None = None


def selected_row(frame: np.ndarray, rows: list[tuple[int, int, int, int]],
                 mode: str = "bright", margin: float = 0.0, previous: int = 0) -> int:
    """行の明るさの1位を選ばれている行とする。閾値で決めると背景で外れる。

    乗せると暗くなるゲームは、ほかの行よりはっきり暗い行を選ぶ。マウスが行から外れると
    どの行も同じ明るさになるので、そのときは前の行のままにする。
    """
    levels = [float(to_gray(crop(frame, box)).mean()) for box in rows]
    if mode != "dark":
        return int(np.argmax(levels))
    darkest = int(np.argmin(levels))
    others = [v for i, v in enumerate(levels) if i != darkest]
    if others and min(others) - levels[darkest] >= margin:
        return darkest
    return min(previous, len(rows) - 1)


class Director:
    def __init__(self, config: GameConfig, watcher: Watcher, services: Services):
        self.config = config
        self.watcher = watcher
        self.s = services
        self.generation = 0
        self.speaking: str | None = None
        self.pending: Pending | None = None
        self.spoken: tuple[str, np.ndarray] | None = None
        self.band: np.ndarray | None = None
        self.scene: Scene | None = None
        self.scene_memory: list[tuple[np.ndarray, str, np.ndarray]] = []
        self.last_description: tuple[str, np.ndarray] | None = None
        self.latest_frame: np.ndarray | None = None
        self.choice: ChoiceSession | None = None
        self.screen: ScreenState | None = None
        self.last_key_t = -100.0
        self.tag_recheck_at: float | None = None
        self.same_check_at: float | None = None
        self.text_since_scene = True
        self.text_sample: np.ndarray | None = None  # 文ごとの場面の比べに使う、前の文のときの絵
        self.after_speech: Callable[[], None] | None = None
        self.screen_cache: dict[tuple[str, bool, str], tuple[str, str, np.ndarray]] = {}
        self.names: dict[str, str] = {}  # 札の字の形ごとに読んだ名前。同じ札は読み直さない
        self.row_index: int | None = None  # 行を選ぶ画面で、本体がマウスを乗せた行
        self.walk = WalkGuide(config.walk, self) if config.walk else None

    @property
    def walk_tones(self) -> dict[str, np.ndarray]:
        tones = self.s.walk_tones
        return {name: tones.get(name, self.s.chime) for name in ("tick", "tick90", "bump")}

    # ここから入口。app か試験の土台が呼ぶ

    def step(self, frame: np.ndarray, t: float) -> None:
        """1コマ分の処理。"""
        self.latest_frame = frame
        for event in self.watcher.feed(frame, t):
            if isinstance(event, TextReady):
                self._on_text(event)
            elif isinstance(event, KindChanged):
                self._on_kind(event)
            elif isinstance(event, SceneDawn):
                self._on_dawn(event)
            elif isinstance(event, ScreenSettled):
                self._on_settled(event)
        if self.choice and self.choice.ready:
            self._track_choice(frame)
        self._recheck_tag(frame)
        if self.walk and self.watcher.kind == "walk":
            self.walk.step(frame, self.s.clock())
        self.tick()

    def tick(self) -> None:
        now = self.s.clock()
        if self.same_check_at is not None and now >= self.same_check_at:
            self.same_check_at = None
            frame = self.latest_frame
            if (frame is not None and self.watcher.kind == TEXT_KIND and self.pending is None
                    and self.watcher.text.shows_last() and not self._voiced_now(frame)):
                # 背景の模様まで字の形に入ると、別の文でも同じ文に見える。前の結果を使わず読み手に読ませる
                self.s.log("same_again", text=self.spoken[0] if self.spoken else "")
                self.band = crop(frame, self.config.text.band).copy()
                self._read_sentence(self.band)
        if self.pending and not self.pending.failed and now >= self.pending.deadline:
            self.pending.failed = True
            self.s.log("fail", reason="timeout")
            self._play_tone(self.s.fail_tone, "fail")
        scene = self.scene
        if scene and scene.state == "generating" and now >= scene.hold_until:
            # 説明が遅れたら文を先に読む。説明はあとで合図の音だけ鳴らす
            scene.auto = False
            if scene.held:
                text, audio = scene.held
                scene.held = None
                self._say(text, audio, "sentence")

    def on_command(self, command: Command) -> None:
        self.s.log("key", kind=command.kind, source=command.source)
        if self.walk:
            now = self.s.clock()
            if command.kind == "release":
                self.walk.key_up(command.source, now)
            elif command.kind in ("cursor", "move") and self.watcher.kind == "walk":
                self.walk.key_down(command.source, now)
            elif command.kind == "advance" and self.watcher.kind == "map":
                self.walk.remember_target(self.latest_frame)
        if command.kind in ("release", "move"):
            return
        action = self._action_for(command.source)
        kind = command.kind
        if action is not None and action.as_kind:
            kind = action.as_kind
        if kind == "key":  # 対応表にだけあるキー。動きがあれば、画面が変わるのを待って読む
            kind = "cursor" if action is not None else ""
        handler = {
            "advance": self._on_advance, "cursor": self._on_cursor, "repeat": self._on_repeat,
            "describe": self._on_describe, "stop": self._on_stop,
        }.get(kind)
        if handler:
            handler()  # 文を進めるなら、ここで読み上げを止めてから送る
        if action is not None:
            self._send_for_key(command, action)

    # ゲームへ送る

    def _send_for_key(self, command: Command, action: MouseAction) -> None:
        """ゲームへ入力を送るのはここだけ。利用者のキー1回に1つの動き。"""
        self.s.log("send", key=command.source, do=action.do, at=list(action.at))
        self.s.send(action)

    def _input_rule(self) -> InputRule | None:
        screen = self.watcher.screen if self.watcher.kind != TEXT_KIND else None
        if screen is not None and screen.name in self.config.input:
            return self.config.input[screen.name]
        return self.config.input.get(self.watcher.kind)

    def _action_for(self, key: str) -> MouseAction | None:
        """今の画面の対応表で、このキーを何の動きに言い換えるか。無ければ None。"""
        rule = self._input_rule()
        if rule is None:
            return None
        rows = self._input_rows(rule)
        if rows:
            if key in (rule.prev, rule.next):
                step = -1 if key == rule.prev else 1
                index = 0 if self.row_index is None else self.row_index + step
                self.row_index = max(0, min(len(rows) - 1, index))
                return MouseAction("move", _center(rows[self.row_index]))
            if key == rule.pick:
                if self.row_index is None:
                    return None
                return MouseAction("click", _center(rows[self.row_index]))
        return rule.actions.get(key)

    def _input_rows(self, rule: InputRule) -> list:
        if rule.rows == "choice":
            return self.choice.rows if self.choice and self.choice.ready else []
        return list(rule.rows)

    # 文

    def _on_text(self, event: TextReady) -> None:
        rule = self.config.scene
        if rule is not None and rule.text_diff > 0:
            self._scene_by_text(event, rule.box, rule.text_diff)
        self.band = event.band
        self.spoken = None
        self.same_check_at = None
        self.text_since_scene = True
        voiced = self._is_voiced(event)
        self.s.log("text_ready", voiced=voiced)
        scene = self.scene
        self.tag_recheck_at = None
        if voiced:
            self._next_generation()
            if self.config.voice_rule == "tag":
                self.tag_recheck_at = self.s.clock() + TAG_RECHECK_S
            if self.speaking == "sentence":
                self._stop_sound()
            if scene and scene.state == "speaking":
                self._stop_sound()
                scene.state = "done"
                scene.held = None
                self._play_tone(self.s.chime, "chime")
            elif scene and scene.state == "generating":
                scene.auto = False
            return
        self._read_sentence(event.band)

    def _read_sentence(self, band: np.ndarray) -> None:
        generation = self._next_generation()
        if self.speaking == "sentence":
            self._stop_sound()
        self.after_speech = None
        self.pending = Pending(generation, self.s.clock() + FAIL_AFTER_S)

        def checking():
            self.s.tasks.post(lambda: self._extend_deadline(generation))

        key, tag_image = self._name_tag()
        known = self.names.get(key) if key is not None else None
        learned: dict[str, str] = {}

        def work():
            name = known
            if key is not None and known is None:
                result = self.s.reader.read_text(band, on_check=checking, name_image=tag_image)
                name = learned[key] = result.name
            else:
                result = self.s.reader.read_text(band, on_check=checking)
            if not result.text:
                return None
            body = speakable(strip_end_mark(result.text, self.config.text.end_marks))
            text = self._with_name(key, name or "", body)
            return text, self.s.speech(text)

        def done(value, error):
            self.names.update(learned)  # 1本の流れで覚える
            self._on_sentence(generation, value, error)

        self.s.log("read_start", generation=generation)
        self.s.tasks.submit(work, done)

    def _name_tag(self) -> tuple[str | None, np.ndarray | None]:
        """札の字も読むゲームで、今の画面に札があれば、札の字の形の印と札の絵を返す。"""
        tag = self.config.tag
        frame = self.latest_frame
        if tag is None or not tag.read_name or frame is None or not has_name_tag(frame, self.config):
            return None, None
        image = crop(frame, tag.box).copy()
        mask = glyph_mask(to_gray(image), self.config.glyph)
        return hashlib.md5(np.packbits(mask).tobytes()).hexdigest(), image

    def _with_name(self, key: str | None, name: str, body: str) -> str:
        if key is None or not name or body in self.config.text.ignore:
            return body
        return f"{name}。{body}"

    def _extend_deadline(self, generation: int) -> None:
        if self.pending and self.pending.generation == generation:
            self.pending.deadline += CHECKING_EXTRA_S

    def _recheck_tag(self, frame: np.ndarray) -> None:
        """札は前の文から少し残ることがある。消えていたら声の無い文なので読む。"""
        if self.tag_recheck_at is None or self.s.clock() < self.tag_recheck_at:
            return
        self.tag_recheck_at = None
        if self.watcher.kind != TEXT_KIND or has_name_tag(frame, self.config):
            return
        self.s.log("tag_gone")
        self.band = crop(frame, self.config.text.band).copy()
        self._read_sentence(self.band)

    def _voiced_now(self, frame: np.ndarray) -> bool:
        rule = self.config.voice_rule
        if rule == "tag":
            return has_name_tag(frame, self.config)
        return rule != "all" and self.s.is_voice_playing()

    def _is_voiced(self, event: TextReady) -> bool:
        rule = self.config.voice_rule
        if rule == "all":
            return False
        if rule == "tag":
            return event.has_tag
        return self.s.is_voice_playing()

    def _on_sentence(self, generation: int, value, error: Exception | None) -> None:
        if generation != self.generation:
            self.s.log("drop_stale", what="sentence", generation=generation)
            return
        self.pending = None
        if error is not None or value is None:
            self.s.log("fail", reason=repr(error) if error else "empty")
            if error is not None:
                self._play_tone(self.s.fail_tone, "fail")
            return
        text, audio = value
        self.s.log("read_done", text=text, generation=generation)
        if text in self.config.text.ignore:
            return
        self.spoken = (text, audio)
        scene = self.scene
        waiting = scene and (scene.state == "speaking" or
                             (scene.state == "generating" and scene.auto))
        if waiting:
            scene.held = (text, audio)
            return
        self._say(text, audio, "sentence")

    # 場面

    def _scene_by_text(self, event: TextReady, box, limit: float) -> None:
        """暗転も白飛びも通らない場面の切り替わり。文が出るたびに、前の文のときの絵と比べる。

        比べる絵は、文でない画面を通っても捨てない。捨てると、選択肢のあとに同じ場面をもう一度説明する。
        ロードやタイトルからの始め直しは絵が変わるので説明される。
        """
        frame = event.frame if event.frame is not None else self.latest_frame
        if frame is None:
            return
        sample = small_gray(frame, box).astype(np.int16)
        before, self.text_sample = self.text_sample, sample
        # 始めて最初の文は、比べる絵が無いので場面とする
        diff = float(np.abs(sample - before).mean()) if before is not None else float("inf")
        if diff < limit:
            return
        self.s.log("scene_by_text", diff=round(diff, 1) if before is not None else None)
        self.text_since_scene = True  # 文が出たので、映像の区間ではない
        self._on_dawn(SceneDawn(event.t, frame))

    def _on_dawn(self, event: SceneDawn) -> None:
        if event.kind != TEXT_KIND or self.config.scene is None:
            return
        sample = small_gray(event.frame, self.config.scene.box)
        if self.scene is not None and _same_picture(sample, self.scene.sample):
            return
        if not self.text_since_scene:
            # 文が出ないまま絵が変わり続けるのは映像の区間。説明を作らない
            self.s.log("dawn_skipped")
            return
        self.text_since_scene = False
        scene = Scene(sample, event.frame, self.s.clock() + DESCRIPTION_HOLD_S)
        self.scene = scene
        for known_sample, text, audio in self.scene_memory:
            if _same_picture(sample, known_sample):
                self.s.log("description_reused", text=text)
                self._scene_ready(scene, text, audio)
                return
        frame = event.frame

        def work():
            result = self.s.reader.describe(frame)
            return result.text, self.s.speech(result.text)

        self.s.log("description_start")
        self.s.tasks.submit(work, lambda value, error: self._on_description(scene, value, error))

    def _on_description(self, scene: Scene, value, error: Exception | None) -> None:
        if error is not None:
            self.s.log("fail", reason=repr(error), what="description")
            scene.state = "done"
            self._release_held(scene)
            return
        text, audio = value
        self.s.log("description_done", text=text, discarded=scene.discarded)
        if scene.discarded:
            return
        self.scene_memory.append((scene.sample, text, audio))
        self._scene_ready(scene, text, audio)

    def _scene_ready(self, scene: Scene, text: str, audio: np.ndarray) -> None:
        scene.text, scene.audio = text, audio
        self.last_description = (text, audio)
        if scene is not self.scene:
            return
        if not scene.auto or self._in_other_screen():
            scene.state = "done"
            self._play_tone(self.s.chime, "chime")
            self._release_held(scene)
            return
        if self.speaking == "sentence":
            # 文の途中なら、読み終わってから説明を読む
            self.after_speech = lambda: self._say_description(scene)
            return
        self._say_description(scene)

    def _say_description(self, scene: Scene) -> None:
        if scene.state == "done" or scene.audio is None:
            return
        scene.state = "speaking"
        self._say(scene.text, scene.audio, "description", lambda: self._on_description_end(scene))

    def _on_description_end(self, scene: Scene) -> None:
        scene.state = "done"
        self._release_held(scene)

    def _release_held(self, scene: Scene) -> None:
        if scene.held:
            text, audio = scene.held
            scene.held = None
            self._say(text, audio, "sentence")

    # 選択肢

    def _on_kind(self, event: KindChanged) -> None:
        self.s.log("kind", old=event.old, new=event.new)
        self.same_check_at = None
        self.choice = None
        self.screen = None
        self.row_index = None
        if event.new == "choice":
            self._start_choice(event)
        elif event.new == "card":
            self._read_card(event)
        elif event.new != TEXT_KIND:
            if event.new != "loading":
                self.screen = ScreenState(event.screen)
            if event.new == "walk" and self.walk:
                self.walk.enter()
            scene = self.scene
            if scene and scene.state == "generating":
                # 暗転のあとに出たのは場面でなくメニューだった。説明は捨てる
                scene.state = "done"
                scene.discarded = True

    def _start_choice(self, event: KindChanged) -> None:
        rule = self.config.choice
        if rule is None:
            return
        session = ChoiceSession(event.frame)
        self.choice = session
        area = crop(event.frame, rule.box)

        def work():
            items, _ = self.s.reader.read_items(area)
            audios = [self.s.speech(item) for item in items]
            parts = self.s.speech("選択肢。"), self.s.speech("選んでいるのは、")
            return items, audios, parts

        self.s.log("choice_start")
        self.s.tasks.submit(work, lambda value, error: self._on_items(session, value, error))

    def _on_items(self, session: ChoiceSession, value, error: Exception | None) -> None:
        if session is not self.choice:
            self.s.log("drop_stale", what="choice")
            return
        if error is not None or not value[0]:
            self.s.log("fail", reason=repr(error), what="choice")
            self._play_tone(self.s.fail_tone, "fail")
            return
        items, audios, (opening, closing) = value
        session.items, session.audios = items, audios
        session.rows = self.config.choice.rows_for(len(items))
        frame = self.latest_frame if self.latest_frame is not None else session.frame
        session.selected = self._selected_row(frame, session.rows, 0)  # 選択肢が始まるたびに1行目から
        self.row_index = session.selected
        session.ready = True
        self.s.log("choice_ready", items=items, selected=session.selected)
        # 音声は項目ごとに作ってあるので、つなぐだけで紹介ができる。待たせない
        chosen = items[session.selected]
        session.intro_text = "選択肢。" + "".join(f"{i}。" for i in items) + f"選んでいるのは、{chosen}。"
        session.intro_audio = np.concatenate([opening, *audios, closing, audios[session.selected]])
        if self.speaking == "sentence":
            self.after_speech = lambda: self._say_intro(session)
        else:
            self._say_intro(session)

    def _say_intro(self, session: ChoiceSession) -> None:
        if session is self.choice and not session.moved:
            self._say(session.intro_text, session.intro_audio, "choice")

    def _track_choice(self, frame: np.ndarray) -> None:
        session = self.choice
        if session.closing:
            return
        index = self._selected_row(frame, session.rows, session.selected)
        if index == session.selected:
            session.candidate = None
            return
        # 帯が動く途中のコマで順位が揺れるので、同じ行が2コマ続いたら確定する
        if index != session.candidate:
            session.candidate = index
            return
        session.candidate = None
        session.selected = index
        self.row_index = index
        session.moved = True
        self._next_generation()
        self.after_speech = None
        self.s.log("choice_move", selected=index, text=session.items[index])
        self._say(session.items[index], session.audios[index], "choice")

    def _selected_row(self, frame: np.ndarray, rows, previous: int) -> int:
        rule = self.config.choice
        return selected_row(frame, rows, rule.selected, rule.dark_margin, previous)

    # 札（字だけを読む画面）

    def _read_card(self, event: KindChanged) -> None:
        generation = self._next_generation()
        frame = event.frame

        def work():
            result = self.s.reader.read_plain(frame, prompts.PROMPT_ALL, "card")
            return (result.text, self.s.speech(result.text)) if result.text else None

        self.s.tasks.submit(work, lambda v, e: self._on_screen_text(generation, v, e))

    def _on_screen_text(self, generation: int, value, error: Exception | None) -> None:
        if generation != self.generation:
            self.s.log("drop_stale", what="screen", generation=generation)
            return
        self.pending = None
        if error is not None:
            self.s.log("fail", reason=repr(error), what="screen")
            self._play_tone(self.s.fail_tone, "fail")
            return
        if value is None:
            return
        text, audio = value
        if self.screen is not None:
            self.screen.text, self.screen.audio = text, audio
        self.spoken = (text, audio)
        self._say(text, audio, "screen")

    # メニュー、確認の窓、地図、歩く画面、履歴

    def _on_settled(self, event: ScreenSettled) -> None:
        state = self.screen
        if state is None or event.screen is None or event.kind == "choice":
            return
        first = not state.entered
        if not first and self.s.clock() - self.last_key_t > KEY_RECENT_S:
            return  # 利用者が動かしていない変化は読まない。絵の動きで費用が増えるのを防ぐ
        state.entered = True
        if event.kind == "walk" and not first:
            return
        frame = event.frame
        generation = self._next_generation()
        self._stop_sound()
        self.after_speech = None
        key = (event.screen.name, first, fingerprint(frame))
        cached = self.screen_cache.get(key) if event.kind != "walk" else None  # 目当ての人の向きは毎回違う
        if cached:
            text, title, audio = cached
            state.title = title
            self.s.log("screen_cached", text=text)
            self._on_screen_text(generation, (text, audio), None)
            return
        self.pending = Pending(generation, self.s.clock() + FAIL_AFTER_S + 2.0)
        kind = event.kind
        title = state.title
        screen = event.screen

        def work():
            text, new_title = self._read_screen(kind, screen, frame, first, title)
            if not text:
                return None
            return text, new_title, self.s.speech(text)

        def done(value, error):
            if value is None:
                if first:
                    state.entered = False  # 読めなかったので、次も入った直後として読む
                    self.watcher.settle_again()
                self._on_screen_text(generation, None, error)
                return
            text, new_title, audio = value
            if generation == self.generation:
                state.title = new_title
                self.screen_cache[key] = (text, new_title, audio)
            self._on_screen_text(generation, (text, audio), None)

        self.s.log("screen_read", kind=kind, first=first)
        self.s.tasks.submit(work, done)

    def _read_screen(self, kind: str, screen: ScreenRule, frame: np.ndarray, first: bool,
                     title: str) -> tuple[str, str]:
        reader = self.s.reader
        if kind == "menu":
            data, _ = reader.read_menu(frame)
            new_title = str(data.get("title", ""))
            selected = str(data.get("selected", ""))
            if not new_title and not selected:
                return "", title  # 画面が動いている途中で読めなかった
            if first or new_title != title:
                return f"{new_title}。{selected}", new_title
            return selected, new_title
        if kind == "confirm":
            data, _ = reader.read_confirm(frame)
            selected = str(data.get("selected", ""))
            message = prompts.clean_text(str(data.get("message", "")))
            sep = "" if message.endswith(("。", "？", "！", "?", "!")) else "。"
            return (f"{message}{sep}{selected}" if first else selected), message
        if kind == "map":
            prompt = screen.prompt if first else screen.prompt_move
            return reader.read_map(frame, moved=not first, prompt=prompt).text, title
        if kind == "walk":
            name = reader.read_plain(frame, prompts.PROMPT_WALK, "walk").text
            text = f"{name}。歩く画面です。" if name else "歩く画面です。"
            if self.walk:
                text += self.walk.entry_text(frame)
            return text, title
        if kind == "backlog":
            return reader.read_plain(frame, prompts.PROMPT_BACKLOG, "backlog").text, title
        return reader.read_plain(frame, prompts.PROMPT_ALL, "screen").text, title

    # キー

    def _on_advance(self) -> None:
        """押した瞬間に読み上げを止める。作っている途中の結果は鳴らさない。"""
        self.watcher.on_key()
        self.last_key_t = self.s.clock()
        self.tag_recheck_at = None
        self.same_check_at = self.s.clock() + SAME_AGAIN_S
        if self.choice:
            self.choice.closing = True  # 決定のあとの動きは読まない
        self._next_generation()
        self._stop_sound()
        self.after_speech = None
        self.pending = None
        scene = self.scene
        if scene:
            scene.held = None
            if scene.state == "generating":
                scene.auto = False
            elif scene.state == "speaking":
                scene.state = "done"

    def _on_cursor(self) -> None:
        self.last_key_t = self.s.clock()
        self.watcher.expect_change()

    def _on_stop(self) -> None:
        self._stop_sound()
        self.after_speech = None
        scene = self.scene
        if scene and scene.state == "speaking":
            scene.state = "done"
            self._release_held(scene)
            return
        self._next_generation()
        self.pending = None

    def _on_repeat(self) -> None:
        if self.choice and self.choice.intro_audio is not None:
            self._say(self.choice.intro_text, self.choice.intro_audio, "choice")
            return
        if self.screen and self.screen.audio is not None:
            self._say(self.screen.text, self.screen.audio, "screen")
            return
        if self.spoken:
            self._say(self.spoken[0], self.spoken[1], "sentence")
            return
        band = self.band
        if band is not None and self._band_has_text():
            event = TextReady(self.s.clock(), band, False)
            self._read_again(event)
            return
        self._read_whole_screen()

    def _band_has_text(self) -> bool:
        return self.watcher.text.reference is not None

    def _read_again(self, event: TextReady) -> None:
        """声のある台詞や、読めなかった文を、頼まれたときに読む。"""
        generation = self._next_generation()
        band = event.band
        self.pending = Pending(generation, self.s.clock() + FAIL_AFTER_S)

        def work():
            result = self.s.reader.read_text(band)
            text = strip_end_mark(result.text, self.config.text.end_marks)
            return (text, self.s.speech(text)) if text else None

        self.s.tasks.submit(work, lambda v, e: self._on_sentence(generation, v, e))

    def _read_whole_screen(self) -> None:
        frame = self.latest_frame
        if frame is None:
            return
        generation = self._next_generation()
        self.pending = Pending(generation, self.s.clock() + FAIL_AFTER_S)

        def work():
            result = self.s.reader.read_plain(frame, prompts.PROMPT_ALL, "screen")
            return (result.text, self.s.speech(result.text)) if result.text else None

        self.s.tasks.submit(work, lambda v, e: self._on_screen_text(generation, v, e))

    def _on_describe(self) -> None:
        if self.watcher.kind == "walk":
            self._describe_view()
            return
        scene = self.scene
        # 場面の見分けが無いゲームでは、前の説明は今の画面の説明とは限らない。押すたびに今の画面を説明する
        fresh = self.config.scene is None
        if not fresh and self.last_description is not None and (scene is None or scene.state != "generating"):
            text, audio = self.last_description
            self._say(text, audio, "description")
            return
        if scene is not None and scene.state == "generating":
            scene.auto = True
            return
        frame = self.latest_frame
        if frame is None:
            return

        def work():
            result = self.s.reader.describe(frame)
            return result.text, self.s.speech(result.text)

        def done(value, error):
            if error is not None:
                self._play_tone(self.s.fail_tone, "fail")
                return
            self.last_description = value
            self._say(value[0], value[1], "description")

        self.s.tasks.submit(work, done)

    def _describe_view(self) -> None:
        """歩く画面は動くので、前の説明を使わず今の画面を言葉にする。"""
        frame = self.latest_frame
        if frame is None:
            return
        if self.walk:
            self.walk.check(frame, around=True)
            return
        generation = self._next_generation()
        self._stop_sound()

        def work():
            result = self.s.reader.read_plain(frame, prompts.PROMPT_WALK_VIEW, "walk_view")
            return (result.text, self.s.speech(result.text)) if result.text else None

        def done(value, error):
            if generation != self.generation:
                self.s.log("drop_stale", what="walk_view", generation=generation)
                return
            if error is not None or value is None:
                self.s.log("fail", reason=repr(error) if error else "empty", what="walk_view")
                self._play_tone(self.s.fail_tone, "fail")
                return
            self._say(value[0], value[1], "description")

        self.s.tasks.submit(work, done)

    # 音

    def _next_generation(self) -> int:
        self.generation += 1
        return self.generation

    def _in_other_screen(self) -> bool:
        return self.choice is not None or self.screen is not None

    def _say(self, text: str, audio: np.ndarray, what: str,
             on_end: Callable[[], None] | None = None) -> None:
        self.speaking = what
        self.s.log("speak", what=what, text=text)

        def ended():
            self.s.tasks.post(lambda: self._on_speech_end(on_end))

        self.s.sound.play(audio, ended)

    def _on_speech_end(self, on_end: Callable[[], None] | None) -> None:
        self.speaking = None
        self.s.log("speak_end")
        if on_end:
            on_end()
        follow, self.after_speech = self.after_speech, None
        if follow and self.speaking is None:
            follow()

    def _stop_sound(self) -> None:
        if self.s.sound.playing:
            self.s.log("stop")
        self.s.sound.stop()
        self.speaking = None

    def _tone_now(self, tone: np.ndarray, name: str) -> None:
        """読み上げの途中でも鳴らす合図。歩く画面で回した角度や、進めないことを知らせる。"""
        self.s.log("tone", name=name)
        self.speaking = None
        self.s.sound.play(tone)

    def _play_tone(self, tone: np.ndarray, name: str) -> None:
        self.s.log("tone", name=name)
        if not self.s.sound.playing:
            self.speaking = None
            self.s.sound.play(tone)


def strip_end_mark(text: str, marks: str) -> str:
    """文の終わりの「次へ」の印を、読み手が字として返すことがある。終わりの1字だけ外す。

    同じ字が続くとき（・・・のような点の文）は印ではないので外さない。
    """
    text = text.rstrip()
    if marks and len(text) >= 2 and text[-1] in marks and text[-2] != text[-1]:
        return text[:-1].rstrip()
    return text


def speakable(text: str) -> str:
    """点だけの文は声にならないので、無言と伝える。"""
    if re.fullmatch(r"[…‥・。、．.！？!?ー―－\-〜～\s]*", text):
        return "無言。"
    # 点だけの文は、読み手が同じ字の並びに読み違えることがある
    if len(text) >= 4 and len(set(text)) == 1:
        return "無言。"
    return text


def _center(box) -> tuple[int, int]:
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def _same_picture(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) < SAME_PICTURE_DIFF
