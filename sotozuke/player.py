"""再生。出力は開いたままにして、差し替えと停止をすぐ効かせる。"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile

SAMPLE_RATE = 44100
BLOCK = 512


def load_audio(path: Path) -> np.ndarray:
    """音声のファイルを出力の形（44.1kHz、1チャンネル、float32）にする。"""
    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate == SAMPLE_RATE or len(mono) == 0:
        return mono
    count = int(len(mono) * SAMPLE_RATE / rate)
    positions = np.linspace(0, len(mono) - 1, count)
    return np.interp(positions, np.arange(len(mono)), mono).astype(np.float32)


def tone(frequency: float, seconds: float, volume: float = 0.25) -> np.ndarray:
    """短い合図の音。端をなめらかにして、ぶつっと鳴らないようにする。"""
    count = int(SAMPLE_RATE * seconds)
    wave = np.sin(2 * np.pi * frequency * np.arange(count) / SAMPLE_RATE) * volume
    edge = min(count // 4, int(SAMPLE_RATE * 0.01))
    if edge:
        ramp = np.linspace(0, 1, edge)
        wave[:edge] *= ramp
        wave[-edge:] *= ramp[::-1]
    return wave.astype(np.float32)


class Player:
    """音を1つだけ鳴らす。新しい音は今の音を置き換える。順番待ちの列は作らない。"""

    def __init__(self):
        import sounddevice
        self._lock = threading.Lock()
        self._buffer: np.ndarray | None = None
        self._position = 0
        self._on_end: Callable[[], None] | None = None
        self._stop_requested_at: float | None = None
        self.last_stop_latency: float | None = None
        self._stream = sounddevice.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK,
            latency="low", callback=self._callback)
        self._stream.start()

    def _callback(self, out, frames, time_info, status) -> None:
        finished = None
        with self._lock:
            if self._stop_requested_at is not None:
                self.last_stop_latency = time.perf_counter() - self._stop_requested_at
                self._stop_requested_at = None
            buffer = self._buffer
            if buffer is None:
                out.fill(0)
                return
            start = self._position
            chunk = buffer[start:start + frames]
            out[:len(chunk), 0] = chunk
            if len(chunk) < frames:
                out[len(chunk):, 0] = 0
                self._buffer = None
                finished, self._on_end = self._on_end, None
            self._position = start + len(chunk)
        if finished:
            finished()

    def play(self, audio: np.ndarray, on_end: Callable[[], None] | None = None) -> None:
        with self._lock:
            self._buffer = audio
            self._position = 0
            self._on_end = on_end

    def stop(self) -> None:
        """今の音を止める。終わりの知らせは呼ばない。"""
        with self._lock:
            if self._buffer is not None:
                self._stop_requested_at = time.perf_counter()
            self._buffer = None
            self._on_end = None

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._buffer is not None

    def close(self) -> None:
        self.stop()
        self._stream.stop()
        self._stream.close()
