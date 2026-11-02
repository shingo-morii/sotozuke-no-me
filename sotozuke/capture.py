"""ゲームの窓を撮る。ゲームが前にないときは撮らない。"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def make_dpi_aware() -> None:
    """画面の拡大率に左右されない本当の画素で座標を受け取る。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        user32.SetProcessDPIAware()


def process_name(pid: int) -> str:
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(2048)
        size = wintypes.DWORD(2048)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return buffer.value.rsplit("\\", 1)[-1]
    finally:
        kernel32.CloseHandle(handle)
    return ""


def find_window(process: str) -> int | None:
    """プロセスの名前から、見えているいちばん大きい窓を探す。"""
    found: list[tuple[int, int]] = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if process_name(pid.value).lower() == process.lower():
            rect = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            found.append((rect.right * rect.bottom, hwnd))
        return True

    user32.EnumWindows(enum_proc(visit), 0)
    return max(found)[1] if found else None


def client_box(hwnd: int) -> tuple[int, int, int, int]:
    """窓の中身の画面の上の位置。left, top, width, height。"""
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return origin.x, origin.y, rect.right, rect.bottom


def is_foreground(hwnd: int) -> bool:
    return user32.GetForegroundWindow() == hwnd


@dataclass
class WindowCapture:
    """窓の位置を画面から撮る。"""
    process: str

    def __post_init__(self):
        import mss
        self._mss = mss.mss()
        self.hwnd: int | None = None

    def refresh(self) -> bool:
        if self.hwnd is None or not user32.IsWindow(self.hwnd):
            self.hwnd = find_window(self.process)
        return self.hwnd is not None

    def in_front(self) -> bool:
        return self.refresh() and is_foreground(self.hwnd)

    def size(self) -> tuple[int, int] | None:
        if not self.refresh():
            return None
        _, _, width, height = client_box(self.hwnd)
        return width, height

    def grab(self) -> np.ndarray | None:
        """RGB の配列を返す。窓が無ければ None。"""
        if not self.refresh():
            return None
        left, top, width, height = client_box(self.hwnd)
        if width <= 0 or height <= 0:
            return None
        shot = self._mss.grab({"left": left, "top": top, "width": width, "height": height})
        bgra = np.frombuffer(shot.bgra, np.uint8).reshape(shot.height, shot.width, 4)
        return bgra[..., [2, 1, 0]].copy()
