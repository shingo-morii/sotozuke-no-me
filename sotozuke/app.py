"""常駐の本体。画面を撮り、判断の中心へ渡し、声で伝える。

ゲームへ入力を送るのは、利用者が対応表のキーを押した直後だけ（sender.py）。対応表が空なら送る部品を作らない。
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .capture import WindowCapture, make_dpi_aware
from .costs import BudgetExceeded, CostLog
from .director import Director, Services
from .game_audio import GameAudioMonitor
from .game_config import GameConfig, load_config
from .inputs import InputWatcher, KeyMapper
from .nvda import BrowseModeNotice
from .player import Player, load_audio, tone
from .reader import ClaudeReader
from .sender import MouseSender
from .voice import CommandVoice, SapiVoice, VoiceCache
from .watcher import Watcher

FRAME_PERIOD_S = 0.1
STALL_S = 30.0  # 1本の流れがこれだけ止まったら、呼び出しの順を書き残す


class ThreadTasks:
    """仕事は別の流れで行い、結果は1本の流れへ戻す。"""

    def __init__(self, workers: int = 4):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="work")
        self.results: queue.Queue = queue.Queue()

    def submit(self, work, done) -> None:
        def run():
            try:
                value, error = work(), None
            except Exception as exc:  # noqa: BLE001  読み手の失敗は音で知らせる
                value, error = None, exc
            self.results.put(lambda: done(value, error))
        self.pool.submit(run)

    def post(self, call) -> None:
        self.results.put(call)

    def wait_and_run(self, timeout: float) -> None:
        """結果が来るまで待ち、来たら全部行う。"""
        try:
            call = self.results.get(timeout=max(timeout, 0.0))
        except queue.Empty:
            return
        call()
        while True:
            try:
                self.results.get_nowait()()
            except queue.Empty:
                return


class StallDump:
    """1本の流れが止まったら、全部の流れの呼び出しの順をファイルに書く。止まった原因をあとで調べるため。"""

    def __init__(self, path: Path | None, seconds: float = STALL_S):
        self.seconds = seconds
        self.file = open(path, "a", encoding="utf-8") if path else None
        # この名前のファイルが置かれたら、止まっていなくても書く。1本の流れ以外が止まったときに使う
        self.request = path.with_suffix(".request") if path else None
        self.checked_at = 0.0

    def arm(self) -> None:
        if not self.file:
            return
        faulthandler.dump_traceback_later(self.seconds, file=self.file)
        now = time.monotonic()
        if now - self.checked_at >= 1.0:
            self.checked_at = now
            if self.request.exists():
                self.request.unlink()
                self.file.write(f"--- 頼まれて書いた {time.strftime('%H:%M:%S')}\n")
                faulthandler.dump_traceback(file=self.file, all_threads=True)
                self.file.flush()

    def close(self) -> None:
        if self.file:
            faulthandler.cancel_dump_traceback_later()
            self.file.close()
            self.file = None


class JsonLog:
    """読み上げの記録。通しの試験で画面と照らす。"""

    def __init__(self, path: Path | None):
        self.file = None
        self.lock = threading.Lock()
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.file = open(path, "a", encoding="utf-8")

    def __call__(self, event: str, **fields) -> None:
        if self.file is None:
            return
        row = {"wall": round(time.time(), 3), "event": event, **fields}
        with self.lock:
            self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.file.flush()


def build_voice(args):
    if args.voice_command:
        return CommandVoice(args.voice_command, name="command")
    return SapiVoice(args.voice)


class Announcer:
    """本体からの知らせ（始めの挨拶、NVDA の案内、大きさの違い）を、前の知らせを切らずに順に鳴らす。

    前は続けて知らせると、あとの知らせが前の知らせを置き換えた（11-07、1本目で挨拶と NVDA の案内が切れた）。
    ゲームの文の読み上げは、今のまま知らせを置き換える。
    """

    def __init__(self, player, post):
        self.player = player
        self.post = post
        self.waiting: list = []
        self.current = False

    def say(self, audio) -> None:
        if self.current and self.player.playing:
            self.waiting.append(audio)
            return
        self._play(audio)

    def _play(self, audio) -> None:
        self.current = True
        self.player.play(audio, on_end=lambda: self.post(self._next))

    def _next(self) -> None:
        if self.waiting:
            self._play(self.waiting.pop(0))
        else:
            self.current = False


class App:
    def __init__(self, config: GameConfig, args):
        self.config = config
        self.args = args
        self.log = JsonLog(args.log)
        self.tasks = ThreadTasks()
        self.player = Player()
        self.cache = VoiceCache(build_voice(args), Path(args.cache))
        self.capture = WindowCapture(config.process)
        self.audio = None
        if config.voice_rule == "audio":
            self.audio = GameAudioMonitor(threshold=config.audio_threshold,
                                          is_own_sound=lambda: self.player.playing)
        reader = ClaudeReader(CostLog(Path(args.costs), limit_usd=args.budget))
        self.sender = MouseSender(self.capture.origin) if config.input else None
        self.services = Services(
            reader=reader, speech=self.speech, sound=self.player, tasks=self.tasks,
            clock=time.monotonic, log=self.log,
            chime=tone(880, 0.12), fail_tone=tone(330, 0.25),
            is_voice_playing=self.audio.voice_playing if self.audio else (lambda: False),
            walk_tones={"tick": tone(1400, 0.03, 0.15), "tick90": tone(1900, 0.06, 0.2),
                        "bump": tone(140, 0.15, 0.35)},
            send=self.sender.send if self.sender else (lambda action: None),
        )
        self.director = Director(config, Watcher(config), self.services)
        walk = config.walk
        self.mapper = KeyMapper(config.advance_keys, config.cursor_keys,
                                move=walk.move_keys if walk else (),
                                hold=(walk.turn_left, walk.turn_right) if walk else (),
                                mapped=sorted(config.input_keys()))
        self.inputs = InputWatcher(self.mapper, self._on_command, is_active=self.capture.in_front,
                                   is_own=self.sender.is_own if self.sender else (lambda name: False))
        self.size_warned = False
        self.browse_notice = BrowseModeNotice()
        self.announcer = Announcer(self.player, self.tasks.post)
        self.running = True

    def speech(self, text: str):
        return load_audio(self.cache.get(text))

    def _on_command(self, command) -> None:
        # 止めるのは入力の流れからすぐ行う。判断は1本の流れで行う
        if command.kind == "advance":
            playing = self.player.playing
            self.player.stop()
            self.log("stop_now", playing=playing)
        self.tasks.post(lambda: self.director.on_command(command))

    def announce(self, text: str) -> None:
        self.log("announce", text=text)
        self.announcer.say(self.speech(text))

    def run(self) -> None:
        self.announce(f"{self.args.greeting}")
        self.inputs.start()
        if self.audio:
            self.audio.start()
        next_frame = time.monotonic()
        stall = StallDump(Path(self.args.log).with_suffix(".stacks.txt") if self.args.log else None)
        try:
            while self.running:
                stall.arm()
                now = time.monotonic()
                if now >= next_frame:
                    next_frame = max(next_frame + FRAME_PERIOD_S, now)
                    self._frame(now)
                self.tasks.wait_and_run(next_frame - time.monotonic())
                if self.args.seconds and time.monotonic() - self.started > self.args.seconds:
                    break
        except BudgetExceeded:
            self.announce("費用の上限に届いたので止まります。")
            time.sleep(2)
        finally:
            stall.close()
            self.inputs.stop()
            if self.sender:
                self.sender.close()
            if self.audio:
                self.audio.stop()
            self.player.close()

    started = 0.0

    def _frame(self, now: float) -> None:
        if not self.browse_notice.checked and self.capture.refresh():
            notice = self.browse_notice.check(self.capture.hwnd, self.log)
            if notice:
                self.announce(notice)
        if not self.capture.in_front():
            return
        size = self.capture.size()
        if size != self.config.size:
            if not self.size_warned:
                self.size_warned = True
                self.announce("画面の大きさが設定と違うので、読めません。設定を作り直してください。")
            return
        frame = self.capture.grab()
        if frame is not None:
            self.director.step(frame, now)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="sotozuke", description="外付けの目")
    parser.add_argument("--game", required=True, help="ゲームの設定の JSON")
    parser.add_argument("--log", help="読み上げの記録の置き場")
    parser.add_argument("--cache", default=str(Path(os.environ.get("LOCALAPPDATA", ".")) / "sotozuke" / "voice"))
    parser.add_argument("--costs", default="costs.jsonl")
    parser.add_argument("--budget", type=float, default=4.5, help="API の費用の上限（ドル）")
    parser.add_argument("--voice", default="Haruka")
    parser.add_argument("--voice-command", help="外のコマンドで声を作る。{text} と {out} を含める")
    parser.add_argument("--seconds", type=float, default=0, help="この秒数で終わる。試験用")
    parser.add_argument("--greeting", default="外付けの目を始めます。")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    make_dpi_aware()
    args = parse_args(argv)
    config = load_config(args.game)
    app = App(config, args)
    App.started = time.monotonic()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
