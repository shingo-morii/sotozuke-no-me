"""探索のときにゲームを操作する道具。

使い方（ゲームごとに work フォルダを決める）:
    python -m ingest.gamectl --process Game.exe --work work shot 名前
    python -m ingest.gamectl --process Game.exe --work work key enter
    python -m ingest.gamectl --process Game.exe --work work keys down down enter
    python -m ingest.gamectl --process Game.exe --work work hold shift 5
    python -m ingest.gamectl --process Game.exe --work work burst 名前 30 0.1
    python -m ingest.gamectl --process Game.exe --work work mark 目標の名前
    python -m ingest.gamectl --process Game.exe --work work size

動作は全部 work/actions.jsonl に時刻つきで残る。探索の時間とキーの回数を、あとで機械で数える。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path

from PIL import Image

from sotozuke.capture import WindowCapture, make_dpi_aware
from sotozuke.inputs import VK

user32 = ctypes.windll.user32
HOLD_S = 0.12         # 0.06秒では取りこぼすゲームがあった。0.1秒以上押す
KEY_GAP_S = 0.15
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_EXTENDEDKEY = 0x0001
EXTENDED = {"up", "down", "left", "right", "delete", "page_up", "page_down"}
ULONG_PTR = ctypes.c_size_t


class KeyboardInput(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class InputUnion(ctypes.Union):
    _fields_ = [("ki", KeyboardInput), ("mi", MouseInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", InputUnion)]


def send_key(name: str, up: bool) -> None:
    code = VK[name]
    scan = user32.MapVirtualKeyW(code, 0)
    flags = KEYEVENTF_KEYUP if up else 0
    if name in EXTENDED:
        flags |= KEYEVENTF_EXTENDEDKEY
    event = Input(INPUT_KEYBOARD, InputUnion(ki=KeyboardInput(code, scan, flags, 0, 0)))
    user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(Input))


def press(name: str, hold_s: float = HOLD_S) -> None:
    send_key(name, up=False)
    time.sleep(hold_s)
    send_key(name, up=True)


def bring_to_front(hwnd: int) -> None:
    # Alt を一度押すと SetForegroundWindow が通りやすい
    send_key_alt = 0x12
    event = Input(INPUT_KEYBOARD, InputUnion(ki=KeyboardInput(send_key_alt, 0, 0, 0, 0)))
    user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(Input))
    event = Input(INPUT_KEYBOARD, InputUnion(ki=KeyboardInput(send_key_alt, 0, KEYEVENTF_KEYUP, 0, 0)))
    user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(Input))
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.2)


class Control:
    def __init__(self, process: str, work: Path):
        make_dpi_aware()
        self.capture = WindowCapture(process)
        self.work = Path(work)
        (self.work / "shots").mkdir(parents=True, exist_ok=True)

    def log(self, action: str, **fields) -> None:
        row = {"time": round(time.time(), 3), "action": action, **fields}
        with (self.work / "actions.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def front(self) -> None:
        if not self.capture.in_front() and self.capture.hwnd:
            bring_to_front(self.capture.hwnd)

    def shot(self, name: str) -> Path | None:
        self.front()
        frame = self.capture.grab()
        if frame is None:
            print("窓が見つかりません")
            return None
        path = self.work / "shots" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(path)
        self.log("shot", name=name)
        print(path)
        return path

    def key(self, name: str, hold_s: float = HOLD_S) -> None:
        self.front()
        press(name, hold_s)
        self.log("key", name=name, hold=hold_s)

    def hold(self, name: str, seconds: float) -> None:
        self.front()
        send_key(name, up=False)
        time.sleep(seconds)
        send_key(name, up=True)
        self.log("hold", name=name, seconds=seconds)

    def burst(self, name: str, count: int, period: float) -> None:
        """連続で撮る。場面の切り替わりや文字の速さを調べるときに使う。時刻を添えて残す。"""
        self.front()
        folder = self.work / "shots" / name
        folder.mkdir(parents=True, exist_ok=True)
        times = []
        start = time.perf_counter()
        for i in range(count):
            frame = self.capture.grab()
            if frame is not None:
                Image.fromarray(frame).save(folder / f"{i:04d}.png", compress_level=1)
                times.append(round(time.perf_counter() - start, 3))
            wait = start + (i + 1) * period - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        (folder / "times.json").write_text(json.dumps(times), encoding="utf-8")
        self.log("burst", name=name, count=count, period=period)
        print(folder)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gamectl")
    parser.add_argument("--process", required=True)
    parser.add_argument("--work", default="work")
    parser.add_argument("command", choices=["shot", "key", "keys", "hold", "burst", "mark", "size"])
    parser.add_argument("args", nargs="*")
    options = parser.parse_args(argv)
    control = Control(options.process, Path(options.work))
    args = options.args
    if options.command == "shot":
        control.shot(args[0])
    elif options.command == "key":
        control.key(args[0], float(args[1]) if len(args) > 1 else HOLD_S)
    elif options.command == "keys":
        for name in args:
            control.key(name)
            time.sleep(KEY_GAP_S)
    elif options.command == "hold":
        control.hold(args[0], float(args[1]))
    elif options.command == "burst":
        control.burst(args[0], int(args[1]), float(args[2]))
    elif options.command == "mark":
        control.log("mark", name=" ".join(args))
    elif options.command == "size":
        print(control.capture.size())
    return 0


if __name__ == "__main__":
    sys.exit(main())
