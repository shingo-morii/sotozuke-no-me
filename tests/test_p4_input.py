"""P4 の関門。模擬の入力が命令に変わる。本体にキーを送る関数が無いことを機械で確かめる。"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from sotozuke.inputs import EdgeDetector, InputWatcher, KeyMapper

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_WORDS = [
    "SendInput", "keybd_event", "mouse_event", "PostMessage", "SendMessage",
    "pyautogui", "pynput",
]


def mapper():
    return KeyMapper(advance=["enter", "space"], cursor=["up", "down"])


def test_keys_become_commands():
    m = mapper()
    assert m.on_press("enter", 1.0).kind == "advance"
    assert m.on_press("space", 1.0).kind == "advance"
    assert m.on_press("up", 1.0).kind == "cursor"
    assert m.on_press("f6", 1.0).kind == "repeat"
    assert m.on_press("f7", 1.0).kind == "describe"
    assert m.on_press("f8", 1.0).kind == "stop"
    assert m.on_press("z", 1.0) is None


def test_controller_becomes_commands():
    m = mapper()
    assert m.on_press("pad_a", 1.0).kind == "advance"
    assert m.on_press("pad_down", 1.0).kind == "cursor"
    assert m.on_press("pad_l3", 1.0).kind == "repeat"
    assert m.on_press("pad_r3", 1.0).kind == "describe"


def test_game_keys_cannot_steal_the_fixed_commands_silently():
    """ゲームの次へのキーは、読み直しのキーより後に割り当たる。設定で上書きしたことが分かる形にする。"""
    m = KeyMapper(advance=["f6"], cursor=[])
    assert m.on_press("f6", 1.0).kind == "advance"


def test_nvda_key_cannot_be_bound():
    with pytest.raises(ValueError):
        KeyMapper(advance=["nonconvert"], cursor=[])
    with pytest.raises(ValueError):
        KeyMapper(advance=[], cursor=[], bindings={"repeat": ("nonconvert",)})


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError):
        KeyMapper(advance=["no_such_key"], cursor=[])


def test_edge_is_only_at_press():
    edges = EdgeDetector()
    assert edges.update({"enter"}) == ["enter"]
    assert edges.update({"enter"}) == []
    assert edges.update(set()) == []
    assert edges.update({"enter", "up"}) == ["enter", "up"]


def test_watcher_passes_commands(monkeypatch):
    import sotozuke.inputs as inputs
    states = [{"enter"}, {"enter"}, set(), {"f6"}, set()]
    got = []

    def fake_keys(names):
        return states.pop(0) if states else set()

    monkeypatch.setattr(inputs, "_key_state", fake_keys)
    monkeypatch.setattr(inputs, "_open_xinput", lambda: None)
    watcher = InputWatcher(mapper(), got.append, interval_s=0.001)
    watcher.start()
    import time
    deadline = time.time() + 1.0
    while len(got) < 2 and time.time() < deadline:
        time.sleep(0.005)
    watcher.stop()
    assert [c.kind for c in got] == ["advance", "repeat"]


def test_watcher_ignores_input_when_game_not_in_front(monkeypatch):
    import sotozuke.inputs as inputs
    states = [{"enter"}, set()]
    got = []
    monkeypatch.setattr(inputs, "_key_state", lambda names: states.pop(0) if states else set())
    monkeypatch.setattr(inputs, "_open_xinput", lambda: None)
    watcher = InputWatcher(mapper(), got.append, is_active=lambda: False, interval_s=0.001)
    watcher.start()
    import time
    time.sleep(0.05)
    watcher.stop()
    assert got == []


def test_gate_body_has_no_function_that_sends_keys():
    """本体の中に、ゲームへキーや入力を送る語が1つも無いこと。"""
    pattern = re.compile("|".join(FORBIDDEN_WORDS))
    found = []
    for path in sorted((ROOT / "sotozuke").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                found.append(f"{path.name}:{number}: {line.strip()}")
    assert not found, found


def test_body_config_has_no_game_specific_values():
    """本体に画面の大きさや座標の決め打ちが無いこと。"""
    pattern = re.compile(r"\b(1024|768|1360|1280)\b")
    found = []
    for path in sorted((ROOT / "sotozuke").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                found.append(f"{path.name}:{number}: {line.strip()}")
    assert not found, found
