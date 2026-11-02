"""ゲームの音の大きさを見る。音声そのものは録らず、音量の針だけを読む。

Windows の音声のセッションごとの針（IAudioMeterInformation）を使う。ゲームのプロセスの音だけが分かるので、
読み上げの声が混ざらない。声の見分けで「音で見る」ゲームに使う。
"""
from __future__ import annotations

import threading
import time
from collections import deque

import comtypes
from comtypes import COMMETHOD, GUID, HRESULT, IUnknown
from ctypes import POINTER, c_float, c_int, c_void_p
from ctypes.wintypes import DWORD, LPCWSTR

from .capture import process_name

CLSID_DEVICE_ENUMERATOR = GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")


class IAudioMeterInformation(IUnknown):
    _iid_ = GUID("{C02216F6-8C67-4B5B-9D00-D008E73E0064}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetPeakValue", (["out"], POINTER(c_float), "peak")),
    ]


class IAudioSessionControl2(IUnknown):
    _iid_ = GUID("{BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetState", (["out"], POINTER(c_int), "state")),
        COMMETHOD([], HRESULT, "GetDisplayName", (["out"], POINTER(c_void_p), "name")),
        COMMETHOD([], HRESULT, "SetDisplayName", (["in"], LPCWSTR, "name"), (["in"], c_void_p, "ctx")),
        COMMETHOD([], HRESULT, "GetIconPath", (["out"], POINTER(c_void_p), "path")),
        COMMETHOD([], HRESULT, "SetIconPath", (["in"], LPCWSTR, "path"), (["in"], c_void_p, "ctx")),
        COMMETHOD([], HRESULT, "GetGroupingParam", (["out"], POINTER(GUID), "group")),
        COMMETHOD([], HRESULT, "SetGroupingParam", (["in"], POINTER(GUID), "group"), (["in"], c_void_p, "ctx")),
        COMMETHOD([], HRESULT, "RegisterAudioSessionNotification", (["in"], c_void_p, "client")),
        COMMETHOD([], HRESULT, "UnregisterAudioSessionNotification", (["in"], c_void_p, "client")),
        COMMETHOD([], HRESULT, "GetSessionIdentifier", (["out"], POINTER(c_void_p), "value")),
        COMMETHOD([], HRESULT, "GetSessionInstanceIdentifier", (["out"], POINTER(c_void_p), "value")),
        COMMETHOD([], HRESULT, "GetProcessId", (["out"], POINTER(DWORD), "pid")),
        COMMETHOD([], HRESULT, "IsSystemSoundsSession"),
        COMMETHOD([], HRESULT, "SetDuckingPreference", (["in"], c_int, "opt_out")),
    ]


class IAudioSessionEnumerator(IUnknown):
    _iid_ = GUID("{E2F5BB11-0570-40CA-ACDD-3AA01277DEE8}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetCount", (["out"], POINTER(c_int), "count")),
        COMMETHOD([], HRESULT, "GetSession", (["in"], c_int, "index"),
                  (["out"], POINTER(POINTER(IUnknown)), "session")),
    ]


class IAudioSessionManager2(IUnknown):
    _iid_ = GUID("{77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetAudioSessionControl", (["in"], c_void_p, "guid"),
                  (["in"], DWORD, "flags"), (["out"], POINTER(c_void_p), "control")),
        COMMETHOD([], HRESULT, "GetSimpleAudioVolume", (["in"], c_void_p, "guid"),
                  (["in"], DWORD, "flags"), (["out"], POINTER(c_void_p), "volume")),
        COMMETHOD([], HRESULT, "GetSessionEnumerator",
                  (["out"], POINTER(POINTER(IAudioSessionEnumerator)), "enumerator")),
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


CLSCTX_ALL = 23
E_RENDER = 0
E_MULTIMEDIA = 1


def open_game_meters(process: str) -> list:
    """ゲームのプロセスの音の針の一覧を返す。ゲームが音を出す前は空のことがある。"""
    enumerator = comtypes.CoCreateInstance(CLSID_DEVICE_ENUMERATOR, IMMDeviceEnumerator, CLSCTX_ALL)
    device = enumerator.GetDefaultAudioEndpoint(E_RENDER, E_MULTIMEDIA)
    manager = device.Activate(IAudioSessionManager2._iid_, CLSCTX_ALL, None).QueryInterface(IAudioSessionManager2)
    sessions = manager.GetSessionEnumerator()
    meters = []
    for i in range(sessions.GetCount()):
        session = sessions.GetSession(i)
        control = session.QueryInterface(IAudioSessionControl2)
        if process_name(control.GetProcessId()).lower() == process.lower():
            meters.append(session.QueryInterface(IAudioMeterInformation))
    return meters


class GameAudioMonitor:
    """別の流れで針を読み続ける。文が出たときに、声が鳴っているかを答える。

    背景の音楽より大きい音が、続けて鳴っていれば声と見なす。効果音で誤らないよう、
    短い音だけでは声と見なさない。
    """

    def __init__(self, process: str, threshold: float = 0.05, min_active_s: float = 0.3,
                 window_s: float = 1.0, interval_s: float = 0.03):
        self.process = process
        self.threshold = threshold
        self.min_active_s = min_active_s
        self.window_s = window_s
        self.interval_s = interval_s
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
        meters: list = []
        next_open = 0.0
        while self._running:
            now = time.monotonic()
            if not meters and now >= next_open:
                try:
                    meters = open_game_meters(self.process)
                except OSError:
                    meters = []
                next_open = now + 2.0
            try:
                level = max((m.GetPeakValue() for m in meters), default=0.0)
                self.samples.append((now, float(level)))
            except OSError:
                meters = []
            time.sleep(self.interval_s)

    def recent(self, seconds: float) -> list[tuple[float, float]]:
        limit = time.monotonic() - seconds
        return [s for s in list(self.samples) if s[0] >= limit]

    def voice_playing(self) -> bool:
        active = [t for t, level in self.recent(self.window_s) if level >= self.threshold]
        if len(active) < 2:
            return False
        return active[-1] - active[0] >= self.min_active_s

