"""判断の中心を、時計を差し替えて動かす試験の土台。実時間にも音にも API にも頼らない。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import synth
from sotozuke.director import Director, Services
from sotozuke.game_config import GameConfig
from sotozuke.inputs import Command
from sotozuke.watcher import Watcher

FPS = 10


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeTasks:
    """仕事は頼まれた時点で行い、結果は遅れの分だけあとで届ける。"""

    def __init__(self, clock: FakeClock, delay: Callable[[], float]):
        self.clock = clock
        self.delay = delay
        self.queue: list[tuple[float, int, Callable[[], None]]] = []
        self.count = 0

    def _push(self, due: float, call: Callable[[], None]) -> None:
        self.count += 1
        self.queue.append((due, self.count, call))
        self.queue.sort(key=lambda item: item[:2])

    def submit(self, work, done) -> None:
        try:
            value, error = work(), None
        except Exception as exc:  # noqa: BLE001
            value, error = None, exc
        self._push(self.clock.now + self.delay(), lambda: done(value, error))

    def post(self, call) -> None:
        self._push(self.clock.now, call)

    def run_due(self) -> None:
        while self.queue and self.queue[0][0] <= self.clock.now:
            _, _, call = self.queue.pop(0)
            call()


class FakeSound:
    """鳴らした記録だけを取る。長さは配列の要素数を秒と見なす。"""

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.current: tuple[float, Callable | None] | None = None
        self.stops = 0

    def play(self, audio, on_end=None) -> None:
        self.current = (self.clock.now + float(len(audio)), on_end)

    def stop(self) -> None:
        if self.current:
            self.stops += 1
        self.current = None

    @property
    def playing(self) -> bool:
        return self.current is not None

    def run_due(self) -> None:
        if self.current and self.current[0] <= self.clock.now:
            _, on_end = self.current
            self.current = None
            if on_end:
                on_end()


@dataclass
class FakeResult:
    text: str
    seconds: float = 0.0
    usd: float = 0.0


class FakeReader:
    """読み手の代わり。正解は試験の土台が教える。"""

    def __init__(self):
        self.text_calls = 0
        self.describe_calls = 0
        self.items_calls = 0
        self.other_calls = 0
        self.truth_text: Callable[[], str] = lambda: ""
        self.truth_items: Callable[[], list[str]] = lambda: []
        self.fail_next = 0

    def _check(self):
        if self.fail_next:
            self.fail_next -= 1
            raise RuntimeError("読み手の失敗の模擬")

    def read_text(self, band) -> FakeResult:
        self.text_calls += 1
        self._check()
        return FakeResult(self.truth_text())

    def read_items(self, area):
        self.items_calls += 1
        self._check()
        items = self.truth_items()
        return items, FakeResult("")

    def describe(self, frame) -> FakeResult:
        self.describe_calls += 1
        self._check()
        return FakeResult(f"説明{self.describe_calls}")

    def read_menu(self, frame):
        self.other_calls += 1
        self._check()
        return {"title": "メニュー", "selected": f"項目{self.other_calls}"}, FakeResult("")

    def read_confirm(self, frame):
        self.other_calls += 1
        self._check()
        return {"message": "確認の文", "selected": "はい"}, FakeResult("")

    def read_map(self, frame, moved):
        self.other_calls += 1
        return FakeResult("地図の説明")

    def read_plain(self, frame, prompt, purpose):
        self.other_calls += 1
        return FakeResult(f"{purpose}の字")


@dataclass
class Sim:
    style: synth.Style
    config: GameConfig
    read_delay: float = 1.0
    clock: FakeClock = field(default_factory=FakeClock)
    records: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.tasks = FakeTasks(self.clock, lambda: self.read_delay)
        self.sound = FakeSound(self.clock)
        self.reader = FakeReader()
        self.view = synth.View(scene=1)
        self.reader.truth_text = lambda: self.view.text
        self.reader.truth_items = lambda: list(self.view.choices)

        def speech(text: str) -> np.ndarray:
            # 1字0.1秒の長さの音と見なす
            return np.zeros(int(len(text) * 0.1) + 1, np.float32)

        services = Services(
            reader=self.reader, speech=speech, sound=self.sound, tasks=self.tasks,
            clock=self.clock, log=self._log,
            chime=np.zeros(1, np.float32), fail_tone=np.zeros(1, np.float32),
        )
        self.watcher = Watcher(self.config)
        self.director = Director(self.config, self.watcher, services)

    def _log(self, event: str, **fields) -> None:
        self.records.append({"t": round(self.clock.now, 2), "event": event, **fields})

    def run(self, seconds: float) -> None:
        end = self.clock.now + seconds
        while self.clock.now < end - 1e-9:
            self.tasks.run_due()
            self.sound.run_due()
            self.tasks.run_due()
            frame = synth.render(self.style, self.view, self.clock.now)
            self.director.step(frame, self.clock.now)
            self.clock.now += 1 / FPS

    def press(self, kind: str = "advance", source: str = "enter") -> None:
        self.director.on_command(Command(kind, source, self.clock.now))

    def said(self, what: str | None = None) -> list[str]:
        return [r["text"] for r in self.records
                if r["event"] == "speak" and (what is None or r["what"] == what)]

    def events(self, name: str) -> list[dict]:
        return [r for r in self.records if r["event"] == name]
