"""G1b フェイズ D の関門。取り込みの道具のマウスの命令と、スキルに足したこと。本体とスキルに1本目の値が無いこと。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / ".claude" / "skills" / "add-game" / "SKILL.md"


class FakeCapture:
    hwnd = 1

    def in_front(self):
        return True

    def origin(self):
        return (100, 200)


@pytest.fixture
def control(tmp_path, monkeypatch):
    """送る部分を差し替えた道具。実際のマウスは動かさない。"""
    from ingest import gamectl
    done = []
    monkeypatch.setattr(gamectl.sender, "click_at", lambda x, y, button, hold_s: done.append(("click", x, y, button)))
    monkeypatch.setattr(gamectl.sender, "move_to", lambda x, y: done.append(("move", x, y)))
    monkeypatch.setattr(gamectl.sender, "wheel_at", lambda x, y, amount: done.append(("wheel", x, y, amount)))
    monkeypatch.setattr(gamectl, "press", lambda name, hold_s=0: done.append(("key", name)))
    monkeypatch.setattr(gamectl, "WindowCapture", lambda process: FakeCapture())
    monkeypatch.setattr(gamectl, "make_dpi_aware", lambda: None)
    return gamectl, done, tmp_path


def test_gate_gamectl_mouse_commands_are_sent_and_logged(control):
    gamectl, done, work = control
    for argv in (["click", "640", "600"], ["click", "10", "20", "right"], ["move", "5", "6"],
                 ["wheel", "640", "300", "-1"]):
        assert gamectl.main(["--process", "game.exe", "--work", str(work), *argv]) == 0
    assert done == [("click", 740, 800, "left"), ("click", 110, 220, "right"), ("move", 105, 206),
                    ("wheel", 740, 500, -1)]
    rows = [json.loads(line) for line in (work / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["action"] for r in rows] == ["click", "click", "move", "wheel"]
    assert rows[0]["x"] == 640 and rows[1]["button"] == "right" and rows[3]["amount"] == -1


def test_action_time_counts_mouse_actions(control):
    gamectl, done, work = control
    from ingest import action_time
    for argv in (["key", "enter"], ["click", "1", "2"], ["click", "3", "4"], ["move", "1", "1"],
                 ["wheel", "1", "1", "2"], ["mark", "本文"]):
        gamectl.main(["--process", "game.exe", "--work", str(work), *argv])
    summary = action_time.summarize(work / "actions.jsonl")
    assert (summary["keys"], summary["clicks"], summary["moves"], summary["wheels"]) == (1, 2, 1, 1)


def test_gate_skill_has_the_new_steps_and_goal():
    text = SKILL.read_text(encoding="utf-8")
    assert "キーだけで最後まで遊ぶ" in text
    assert "13" in text
    for word in ("click", "move", "wheel", "input", "rows", "read_name", "text_diff", "end_marks",
                 "dark_margin", "lbutton"):
        assert word in text, word
    assert "11/6 1本目のゲームから" in text
    assert text.count("11/6 1本目のゲームから") >= 5
    assert "ゲームにキーを送らない" not in text  # 前の決まり


def test_gate_no_coordinates_or_values_in_body_and_skill():
    """本体のコードとスキルに、座標の並びや画面の大きさが無い。1本目の値は設定にだけ書く。"""
    four = re.compile(r"[\[(]\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*[\])]")
    pair = re.compile(r"\b\d{3,4}\s*,\s*\d{3,4}\b")
    found = []
    for path in [*sorted((ROOT / "sotozuke").rglob("*.py")), SKILL]:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if four.search(line) or pair.search(line):
                found.append(f"{path.name}:{number}: {line.strip()}")
    assert found == []


def test_measured_thresholds_have_no_default_in_the_body():
    """1本目で測った値（行の暗さの差、文ごとの場面の差）は、本体の既定にしない。書かなければ切。"""
    from sotozuke.game_config import ChoiceRule, SceneRule
    assert ChoiceRule(box=(0, 0, 1, 1)).dark_margin == 0
    assert SceneRule(box=(0, 0, 1, 1)).text_diff == 0
