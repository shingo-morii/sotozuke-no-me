"""NVDA のブラウズモードの案内。

NVDA が Web の作りの窓（Chromium で作ったゲームの窓など）をブラウズモードで開くと、字や矢印のキーを
NVDA が受け取り、ゲームにも本体にも届かない（11-07、3本目で見つけた）。ブラウズモードかどうかを本体から
NVDA に聞く口は無いので、NVDA が動いていて、窓が Web の作りなら、始めに1回だけ案内する。
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Callable

from .capture import process_name

NVDA_PROCESS = "nvda.exe"
# ブラウズモードになりやすい窓の種類の名前の始まり。Chromium（NW.js、Electron）と Firefox
WEB_WINDOW_CLASSES = ("Chrome_WidgetWin", "Chrome_RenderWidgetHostHWND", "MozillaWindowClass")
NOTICE = ("NVDA のブラウズモードでは、字や矢印のキーがゲームに届きません。"
          "ゲームの窓で NVDA キーとスペースを押して、フォーカスモードにしてください。")


def nvda_running() -> bool:
    psapi = ctypes.windll.psapi
    pids = (wintypes.DWORD * 4096)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcesses(pids, ctypes.sizeof(pids), ctypes.byref(needed)):
        return False
    count = needed.value // ctypes.sizeof(wintypes.DWORD)
    return any(process_name(pids[i]).lower() == NVDA_PROCESS for i in range(count) if pids[i])


def window_classes(hwnd: int) -> list[str]:
    """窓とその子の窓の種類の名前。"""
    user32 = ctypes.windll.user32
    names = []

    def name_of(handle) -> str:
        buffer = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(handle, buffer, 256)
        return buffer.value

    names.append(name_of(hwnd))
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(child, _):
        if len(names) < 50:
            names.append(name_of(child))
        return True

    user32.EnumChildWindows(hwnd, proc(visit), 0)
    return names


def is_web_window(classes: list[str]) -> bool:
    return any(name.startswith(WEB_WINDOW_CLASSES) for name in classes)


class BrowseModeNotice:
    """ゲームの窓が見つかったら1回だけ見て、案内が要れば案内の文を返す。"""

    def __init__(self, is_running: Callable[[], bool] = nvda_running,
                 classes_of: Callable[[int], list[str]] = window_classes):
        self.is_running = is_running
        self.classes_of = classes_of
        self.checked = False

    def check(self, hwnd: int | None, log: Callable[..., None]) -> str | None:
        if self.checked or hwnd is None:
            return None
        self.checked = True
        running = self.is_running()
        classes = self.classes_of(hwnd)
        web = is_web_window(classes)
        log("nvda_check", running=running, web=web, classes=sorted(set(classes))[:10])
        return NOTICE if running and web else None
