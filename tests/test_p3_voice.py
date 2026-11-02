"""P3 の関門。音声ファイルの長さが字数に合う。止める命令で0.2秒以内に止まる。"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile

from sotozuke.player import SAMPLE_RATE, Player, load_audio, tone
from sotozuke.voice import SapiVoice, VoiceCache

SENTENCES = [
    "はい。",
    "朝の光が窓から差しこんでいた。",
    "駅までの道はいつもより静かで、遠くの踏切の音だけが聞こえていた。",
    "そんなこと、急に言われても困る、と彼女は小さな声でつぶやいて、窓の外へ目をそらした。",
]


class CountingVoice:
    name = "count"

    def __init__(self):
        self.calls = 0

    def synthesize(self, text: str, path: Path) -> None:
        self.calls += 1
        soundfile.write(str(path), np.zeros(1000, np.float32), 22050, format="WAV")


def test_cache_does_not_make_twice(tmp_path):
    voice = CountingVoice()
    cache = VoiceCache(voice, tmp_path)
    first = cache.get("同じ文")
    second = cache.get("同じ文")
    third = cache.get("違う文")
    assert first == second
    assert first != third
    assert voice.calls == 2
    assert cache.reused == 1


def test_tone_has_expected_length():
    wave = tone(880, 0.2)
    assert len(wave) == int(SAMPLE_RATE * 0.2)
    assert abs(wave).max() <= 0.26


def test_load_audio_resamples(tmp_path):
    path = tmp_path / "a.wav"
    soundfile.write(str(path), np.ones(22050, np.float32) * 0.1, 22050)
    audio = load_audio(path)
    assert abs(len(audio) - SAMPLE_RATE) < 3
    assert audio.dtype == np.float32


@pytest.fixture(scope="module")
def sapi_cache(tmp_path_factory):
    return VoiceCache(SapiVoice(), tmp_path_factory.mktemp("voice"))


@pytest.mark.audio
@pytest.mark.parametrize("text", SENTENCES)
def test_gate_audio_length(sapi_cache, text):
    start = time.perf_counter()
    path = sapi_cache.get(text)
    made = time.perf_counter() - start
    data, rate = soundfile.read(str(path), dtype="float32")
    seconds = len(data) / rate
    n = len(text)
    print(f"{n}字 長さ {seconds:.2f}秒 作る時間 {made:.3f}秒")
    assert 0.05 * n <= seconds <= 0.35 * n + 1.0
    assert np.abs(data).max() > 0.01


@pytest.mark.audio
def test_voice_from_worker_thread(sapi_cache):
    """読み手の結果は別の流れで声にするので、別の流れからも作れること。"""
    import threading
    box = {}

    def work():
        box["path"] = sapi_cache.get("別の流れの文です。")

    thread = threading.Thread(target=work)
    thread.start()
    thread.join(20)
    assert box["path"].exists()


@pytest.mark.audio
def test_gate_stop_within_limit():
    player = Player()
    try:
        latencies = []
        wave = tone(440, 1.0, volume=0.02)
        for _ in range(20):
            player.play(wave)
            time.sleep(0.05)
            player.stop()
            deadline = time.perf_counter() + 1.0
            while player.last_stop_latency is None and time.perf_counter() < deadline:
                time.sleep(0.002)
            assert player.last_stop_latency is not None
            latencies.append(player.last_stop_latency)
            player.last_stop_latency = None
            assert not player.playing
        print(f"止まるまで 中央値 {statistics.median(latencies)*1000:.1f}ms 最大 {max(latencies)*1000:.1f}ms")
        assert max(latencies) <= 0.2
    finally:
        player.close()


def test_voice_needs_a_lasting_sound():
    """短い音は声と見なさない。続けて大きい音が鳴れば声と見なす。"""
    from sotozuke.game_audio import GameAudioMonitor
    monitor = GameAudioMonitor(threshold=0.05, min_active_s=0.3, window_s=1.0)
    now = time.monotonic()
    monitor.samples.extend((now - 0.5 + i * 0.03, 0.1) for i in range(3))
    assert not monitor.voice_playing()
    monitor.samples.clear()
    monitor.samples.extend((now - 0.6 + i * 0.03, 0.1) for i in range(20))
    assert monitor.voice_playing()
    monitor.samples.clear()
    monitor.samples.extend((now - 0.6 + i * 0.03, 0.02) for i in range(20))
    assert not monitor.voice_playing()  # 背景の音楽の大きさ


@pytest.mark.audio
def test_finish_callback_and_replace():
    player = Player()
    try:
        ended = []
        player.play(tone(440, 0.1, volume=0.02), lambda: ended.append("a"))
        player.play(tone(660, 0.1, volume=0.02), lambda: ended.append("b"))
        time.sleep(0.5)
        assert ended == ["b"]  # 置き換えられた音の知らせは来ない
        player.play(tone(440, 0.5, volume=0.02), lambda: ended.append("c"))
        time.sleep(0.05)
        player.stop()
        time.sleep(0.3)
        assert ended == ["b"]  # 止めた音の知らせも来ない
    finally:
        player.close()
