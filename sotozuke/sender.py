"""ゲームへ入力を送る、本体の中のただ1か所。

決まり（2026-11-06 に改めた）：本体がゲームへ入力を送るのは、利用者が対応表のキーを押した直後だけ。
1回押して1つの動きだけ。本体が自分から文を進めたり、選んだりしない。対応表が空のゲームでは何も送らない。

送るのはマウスの移動、押し、ホイールの3つ。キーボードの入力は送らない。
MouseSender.send を呼ぶのは判断の中心が利用者のキーを受けたときだけで、テストで確かめている。
"""
from __future__ import annotations

import ctypes
import queue
import threading
import time
from ctypes import wintypes
from typing import Callable

from .game_config import MouseAction
from .inputs import MOUSE_BUTTONS

user32 = ctypes.windll.user32

CLICK_HOLD_S = 0.05   # 押してから離すまで
OWN_CLICK_S = 0.4     # 本体が押してからこの間のボタンの押しは、利用者の押しと数えない
WHEEL_DELTA = 120

INPUT_MOUSE = 0
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
BUTTON_FLAGS = {"left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
                "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP)}
ULONG_PTR = ctypes.c_size_t


class _MouseInput(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _InputUnion)]


def _mouse_event(flags: int, data: int = 0) -> None:
    event = _Input(INPUT_MOUSE, _InputUnion(mi=_MouseInput(0, 0, data & 0xFFFFFFFF, flags, 0, 0)))
    user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_Input))


def move_to(x: int, y: int) -> None:
    """画面の上の座標へマウスを動かす。"""
    user32.SetCursorPos(int(x), int(y))


def click_at(x: int, y: int, button: str = "left", hold_s: float = CLICK_HOLD_S) -> None:
    down, up = BUTTON_FLAGS[button]
    move_to(x, y)
    _mouse_event(down)
    time.sleep(hold_s)
    _mouse_event(up)


def wheel_at(x: int, y: int, amount: int) -> None:
    """ホイールを amount 刻み回す。上が正。"""
    move_to(x, y)
    _mouse_event(MOUSEEVENTF_WHEEL, amount * WHEEL_DELTA)


class MouseSender:
    """窓の中の座標の動きを、別の流れで送る。押している間、判断の中心を待たせない。"""

    def __init__(self, origin: Callable[[], tuple[int, int] | None]):
        self.origin = origin  # 窓の中身の左上の、画面の上の座標
        self.last_own = -100.0
        self.queue: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="send")
        self.thread.start()

    def send(self, action: MouseAction) -> None:
        self.queue.put(action)

    def is_own(self, name: str) -> bool:
        return name in MOUSE_BUTTONS and time.monotonic() - self.last_own < OWN_CLICK_S

    def close(self) -> None:
        self.queue.put(None)

    def _loop(self) -> None:
        while True:
            action = self.queue.get()
            if action is None:
                return
            self._perform(action)

    def _perform(self, action: MouseAction) -> None:
        origin = self.origin()
        if origin is None:
            return  # 窓が無い
        x, y = origin[0] + action.at[0], origin[1] + action.at[1]
        if action.do == "move":
            move_to(x, y)
        elif action.do == "click":
            self.last_own = time.monotonic()
            click_at(x, y, action.button)
            self.last_own = time.monotonic()
        elif action.do == "wheel":
            wheel_at(x, y, action.amount)
