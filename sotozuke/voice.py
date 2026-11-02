"""声を作る。読み手の結果を音声のファイルにする。声は差し替えられる作りにする。"""
from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Protocol

SAFT_22KHZ_16BIT_MONO = 22
SSFM_CREATE_FOR_WRITE = 3
SVSF_IS_NOT_XML = 16


class Voice(Protocol):
    name: str

    def synthesize(self, text: str, path: Path) -> None:
        """text を読み上げた音声を path に書く。"""


class SapiVoice:
    """Windows 標準の声。追加の準備なしに必ず鳴らせる。"""

    def __init__(self, voice_name: str = "Haruka", rate: int = 2):
        self.voice_name = voice_name
        self.rate = rate
        self.name = f"sapi-{voice_name}-{rate}"
        self._local = threading.local()

    def _speaker(self):
        speaker = getattr(self._local, "speaker", None)
        if speaker is None:
            import comtypes
            import comtypes.client
            comtypes.CoInitialize()
            speaker = comtypes.client.CreateObject("SAPI.SpVoice")
            voices = speaker.GetVoices()
            for i in range(voices.Count):
                if self.voice_name in voices.Item(i).GetDescription():
                    speaker.Voice = voices.Item(i)
                    break
            else:
                raise RuntimeError(f"声が見つかりません: {self.voice_name}")
            speaker.Rate = self.rate
            self._local.speaker = speaker
        return speaker

    def synthesize(self, text: str, path: Path) -> None:
        import comtypes.client
        speaker = self._speaker()
        stream = comtypes.client.CreateObject("SAPI.SpFileStream")
        stream.Format.Type = SAFT_22KHZ_16BIT_MONO
        stream.Open(str(path), SSFM_CREATE_FOR_WRITE, False)
        try:
            speaker.AudioOutputStream = stream
            speaker.Speak(text, SVSF_IS_NOT_XML)
        finally:
            stream.Close()
            speaker.AudioOutputStream = None


class CommandVoice:
    """外のコマンドで声を作る。彩など、任意の声をつなぐ口。

    command は {text} と {out} を含むコマンドの文字列。
    例: python make_voice.py --text {text} --out {out}
    """

    def __init__(self, command: str, name: str = "command", timeout_s: float = 60.0):
        self.command = command
        self.name = name
        self.timeout_s = timeout_s

    def synthesize(self, text: str, path: Path) -> None:
        args = [part.replace("{text}", text).replace("{out}", str(path))
                for part in shlex.split(self.command, posix=os.name != "nt")]
        subprocess.run(args, check=True, timeout=self.timeout_s, capture_output=True)


class VoiceCache:
    """一度作った文の音声を保存して、同じ文なら作り直さない。"""

    def __init__(self, voice: Voice, folder: Path):
        self.voice = voice
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.made = 0
        self.reused = 0

    def path_for(self, text: str) -> Path:
        digest = hashlib.sha1(f"{self.voice.name}\n{text}".encode("utf-8")).hexdigest()[:24]
        return self.folder / f"{digest}.wav"

    def get(self, text: str) -> Path:
        path = self.path_for(text)
        if path.exists() and path.stat().st_size > 0:
            self.reused += 1
            return path
        temporary = path.with_suffix(f".{threading.get_ident()}.tmp")
        self.voice.synthesize(text, temporary)
        os.replace(temporary, path)
        self.made += 1
        return path
