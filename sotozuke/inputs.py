"""入力を見る。キーボードとコントローラの状態を読むだけ。ゲームには何も渡さない。

読み取りは GetAsyncKeyState と XInputGetState の見回り。フックを入れないので、
スクリーンリーダーのキー操作と干渉しない。無変換キーは NVDA のキーなので割り当てない。
"""
from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable

# 仮想キーの番号
VK = {
    "enter": 0x0D, "space": 0x20, "escape": 0x1B, "backspace": 0x08, "delete": 0x2E,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "shift": 0x10, "control": 0x11, "tab": 0x09,
    "page_up": 0x21, "page_down": 0x22,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
}
for letter in "abcdefghijklmnopqrstuvwxyz":
    VK[letter] = ord(letter.upper())

VK_NONCONVERT = 0x1D  # 無変換
RESERVED_KEYS = {"nonconvert"}

# XInput のボタンの番号
PAD = {
    "pad_up": 0x0001, "pad_down": 0x0002, "pad_left": 0x0004, "pad_right": 0x0008,
    "pad_start": 0x0010, "pad_back": 0x0020, "pad_l3": 0x0040, "pad_r3": 0x0080,
    "pad_lb": 0x0100, "pad_rb": 0x0200, "pad_a": 0x1000, "pad_b": 0x2000,
    "pad_x": 0x4000, "pad_y": 0x8000,
}

DEFAULT_BINDINGS = {
    "repeat": ("f6", "pad_l3"),       # 読み直す
    "describe": ("f7", "pad_r3"),     # 絵の説明を聞く
    "stop": ("f8",),                  # 読み上げを止める
}


@dataclass(frozen=True)
class Command:
    """入力から作る命令。kind は repeat、describe、stop、advance、cursor のどれか。"""
    kind: str
    source: str
    t: float


class KeyMapper:
    """押された瞬間の入力を命令に変える。設定の値はゲームごとの設定から受け取る。"""

    def __init__(self, advance: Iterable[str], cursor: Iterable[str],
                 bindings: dict[str, tuple[str, ...]] | None = None,
                 pad_advance: Iterable[str] = ("pad_a",),
                 pad_cursor: Iterable[str] = ("pad_up", "pad_down", "pad_left", "pad_right"),
                 move: Iterable[str] = (), hold: Iterable[str] = ()):
        self.table: dict[str, str] = {}
        # 離した瞬間も知らせるキー。歩く画面で、押している長さから回した角度を見積もる
        self.hold: set[str] = set()
        for name in move:
            self._bind(name, "move")
            self.hold.add(name)
        for name in hold:
            if name not in self.table:
                self._bind(name, "cursor")
            self.hold.add(name)
        for command, names in (bindings or DEFAULT_BINDINGS).items():
            for name in names:
                self._bind(name, command)
        for name in advance:
            self._bind(name, "advance")
        for name in pad_advance:
            self._bind(name, "advance")
        for name in cursor:
            self._bind(name, "cursor")
        for name in pad_cursor:
            self._bind(name, "cursor")

    def _bind(self, name: str, command: str) -> None:
        if name in RESERVED_KEYS:
            raise ValueError(f"{name} は割り当てられません")
        if name not in VK and name not in PAD:
            raise ValueError(f"知らないキーです: {name}")
        self.table[name] = command

    def names(self) -> list[str]:
        return list(self.table)

    def on_press(self, name: str, t: float) -> Command | None:
        kind = self.table.get(name)
        return Command(kind, name, t) if kind else None

    def on_release(self, name: str, t: float) -> Command | None:
        return Command("release", name, t) if name in self.hold else None


class EdgeDetector:
    """状態の一覧から、押された瞬間だけを取り出す。"""

    def __init__(self):
        self.down: set[str] = set()

    def update(self, pressed: set[str]) -> list[str]:
        new = sorted(pressed - self.down)
        self.released = sorted(self.down - pressed)
        self.down = pressed
        return new

    released: list[str] = []


def _key_state(names: Iterable[str]) -> set[str]:
    get = ctypes.windll.user32.GetAsyncKeyState
    return {n for n in names if n in VK and get(VK[n]) & 0x8000}


class _XInputGamepad(ctypes.Structure):
    _fields_ = [("buttons", ctypes.c_ushort), ("left_trigger", ctypes.c_ubyte),
                ("right_trigger", ctypes.c_ubyte), ("thumb_lx", ctypes.c_short),
                ("thumb_ly", ctypes.c_short), ("thumb_rx", ctypes.c_short),
                ("thumb_ry", ctypes.c_short)]


class _XInputState(ctypes.Structure):
    _fields_ = [("packet", ctypes.c_ulong), ("gamepad", _XInputGamepad)]


def _open_xinput():
    for name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
        try:
            return getattr(ctypes.windll, name)
        except OSError:
            continue
    return None


def _pad_state(xinput, names: Iterable[str]) -> set[str]:
    if xinput is None:
        return set()
    state = _XInputState()
    if xinput.XInputGetState(0, ctypes.byref(state)) != 0:
        return set()
    return {n for n in names if n in PAD and state.gamepad.buttons & PAD[n]}


class InputWatcher:
    """別の流れで入力を見回り、命令を on_command に渡す。"""

    def __init__(self, mapper: KeyMapper, on_command: Callable[[Command], None],
                 is_active: Callable[[], bool] = lambda: True, interval_s: float = 0.01):
        self.mapper = mapper
        self.on_command = on_command
        self.is_active = is_active
        self.interval_s = interval_s
        self.edges = EdgeDetector()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="input")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(1.0)

    def _loop(self) -> None:
        names = self.mapper.names()
        xinput = _open_xinput()
        while self._running:
            pressed = _key_state(names) | _pad_state(xinput, names)
            for name in self.edges.update(pressed):
                if self.is_active():
                    command = self.mapper.on_press(name, time.monotonic())
                    if command:
                        self.on_command(command)
            for name in self.edges.released:
                # 離したのは窓が前になくても知らせる。押したままの見積もりが残らないように
                command = self.mapper.on_release(name, time.monotonic())
                if command:
                    self.on_command(command)
            time.sleep(self.interval_s)
