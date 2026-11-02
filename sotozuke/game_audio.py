"""ゲームの音の大きさを見る。音声そのものは録らず、音量の針だけを読む。

声の見分けで「音で見る」ゲームに使う。プロセスごとの針は試した PC で値が0のままだったので、
出力の装置全体の針を読む。自分の読み上げが鳴っている間は数えない。
NVDA など別のアプリの音は混ざる。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from ctypes import POINTER, c_float, c_int, c_void_p
from ctypes.wintypes import DWORD
from typing import Callable

import comtypes
from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

CLSID_DEVICE_ENUMERATOR = GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
CLSCTX_ALL = 23
E_RENDER = 0
E_MULTIMEDIA = 1


class IAudioMeterInformation(IUnknown):
    _iid_ = GUID("{C02216F6-8C67-4B5B-9D00-D008E73E0064}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetPeakValue", (["out"], POINTER(c_float), "peak")),
    ]


class IMMDevice(IUnknown):
    _iid_ = GUID("{D666063F-1587-4E43-81F1-B948E807363F}")
    _methods_ = [
        COMMETHOD([], HRESULT, "Activate", (["in"], POINTER(GUID), "iid"), (["in"], DWORD, "ctx"),
                  (["in"], c_void_p, "params"), (["out"], POINTER(POINTER(IUnknown)), "interface")),
    ]


class IMMDeviceEnumerator(IUnknown):
    _iid_ = GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
    _methods_ = [
        COMMETHOD([], HRESULT, "EnumAudioEndpoints", (["in"], c_int, "flow"), (["in"], DWORD, "mask"),
                  (["out"], POINTER(c_void_p), "devices")),
        COMMETHOD([], HRESULT, "GetDefaultAudioEndpoint", (["in"], c_int, "flow"), (["in"], c_int, "role"),
                  (["out"], POINTER(POINTER(IMMDevice)), "device")),
    ]


def open_output_meter():
    """既定の出力の装置の音量の針を開く。呼ぶ流れで CoInitialize しておく。"""
    enumerator = comtypes.CoCreateInstance(CLSID_DEVICE_ENUMERATOR, IMMDeviceEnumerator, CLSCTX_ALL)
    device = enumerator.GetDefaultAudioEndpoint(E_RENDER, E_MULTIMEDIA)
    unknown = device.Activate(IAudioMeterInformation._iid_, CLSCTX_ALL, None)
    return unknown.QueryInterface(IAudioMeterInformation)


class GameAudioMonitor:
    """別の流れで針を読み続ける。文が出たときに、声が鳴っているかを答える。

    背景の音楽より大きい音が、続けて鳴っていれば声と見なす。効果音で誤らないよう、
    短い音だけでは声と見なさない。
    """

    def __init__(self, threshold: float = 0.05, min_active_s: float = 0.3, window_s: float = 1.0,
                 interval_s: float = 0.03, is_own_sound: Callable[[], bool] = lambda: False):
        self.threshold = threshold
        self.min_active_s = min_active_s
        self.window_s = window_s
        self.interval_s = interval_s
        self.is_own_sound = is_own_sound
        self.samples: deque[tuple[float, float]] = deque(maxlen=400)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="game-audio")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(1.0)

    def _loop(self) -> None:
        comtypes.CoInitialize()
        meter = open_output_meter()
        while self._running:
            level = 0.0 if self.is_own_sound() else float(meter.GetPeakValue())
            self.samples.append((time.monotonic(), level))
            time.sleep(self.interval_s)

    def recent(self, seconds: float) -> list[tuple[float, float]]:
        limit = time.monotonic() - seconds
        return [s for s in list(self.samples) if s[0] >= limit]

    def voice_playing(self) -> bool:
        active = [t for t, level in self.recent(self.window_s) if level >= self.threshold]
        if len(active) < 2:
            return False
        return active[-1] - active[0] >= self.min_active_s
