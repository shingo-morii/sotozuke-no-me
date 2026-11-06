"""G1b フェイズ C の関門。改めた決まり「本体がゲームへ入力を送るのは、利用者が対応表のキーを押した直後だけ。
1回押して1つの動きだけ。対応表が空のゲームでは何も送らない」を機械で確かめる。"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

import synth
import timeline
from harness import Sim
from sotozuke.game_config import ConfigError, load_config
from sotozuke.inputs import InputWatcher, KeyMapper

ROOT = Path(__file__).resolve().parent.parent
BODY = ROOT / "sotozuke"
S = timeline.SAMPLE_SENTENCES
CHOICE_HINT = "上下 選択  決定  履歴  設定"
SAVE_HINT = "上下左右 選択  決定  戻る"
TEXT_CLICK = [400, 300]
SAVE_ROWS = [[100, 150, 500, 200], [100, 220, 500, 270], [100, 290, 500, 340]]


def mouse_game(path, dark=True):
    data = json.loads(path.read_text(encoding="utf-8"))
    data["input"] = {
        "text": {"keys": False, "actions": {
            "enter": {"do": "click", "at": TEXT_CLICK},
            "page_down": {"do": "wheel", "at": TEXT_CLICK, "amount": -1, "as": "advance"}}},
        "choice": {"keys": False, "rows": "choice"},
        "セーブ": {"keys": False, "rows": SAVE_ROWS,
                 "actions": {"backspace": {"do": "click", "at": [20, 20], "button": "right"}}},
    }
    if dark:
        data["choice"]["selected"] = "dark"
        data["choice"]["dark_margin"] = 40
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return load_config(path)


def every_key(sim):
    """本体が見るキーを全部、1回ずつ押す。"""
    for kind, source in (("advance", "enter"), ("cursor", "up"), ("cursor", "down"), ("repeat", "f6"),
                         ("describe", "f7"), ("stop", "f8"), ("key", "page_down"), ("key", "backspace")):
        sim.press(kind, source)
        sim.run(0.3)


def screens_with_text_choice_and_save(sim, keys=True):
    sim.view = synth.View(scene=1, text=S[0])
    sim.run(2.0)
    if keys:
        every_key(sim)
    sim.view = synth.View(scene=1, choices=["一つ目", "二つ目"], hint=CHOICE_HINT, choice_dark=True, selected=-1)
    sim.run(3.0)
    if keys:
        every_key(sim)
    sim.view = synth.View(scene=1, hint=SAVE_HINT)
    sim.run(2.0)
    if keys:
        every_key(sim)


def test_gate_empty_table_sends_nothing(game_path, style):
    """対応表が空のゲームでは、キーを何回押しても何も送らない。"""
    sim = Sim(style, load_config(game_path))
    for _ in range(2):
        screens_with_text_choice_and_save(sim)
    assert sim.sent == []
    assert sim.events("send") == []


def test_gate_nothing_is_sent_without_a_key(game_path, style):
    """対応表があっても、キーを押さない間は、画面が変わっても読み手が答えても何も送らない。"""
    sim = Sim(style, mouse_game(game_path))
    screens_with_text_choice_and_save(sim, keys=False)
    sim.view = synth.View(scene=2, dark=1.0)
    sim.run(1.0)
    sim.view = synth.View(scene=2, text=S[1])
    sim.run(4.0)
    assert sim.said()  # 読んではいる
    assert sim.sent == []


def test_gate_one_key_one_action(game_path, style):
    sim = Sim(style, mouse_game(game_path))
    sim.view = synth.View(scene=1, text=S[0])
    sim.run(2.0)
    sim.press("advance", "enter")
    assert [(a.do, a.at) for a in sim.sent] == [("click", tuple(TEXT_CLICK))]
    sim.run(3.0)
    assert len(sim.sent) == 1
    sim.press("advance", "f6")  # 対応表に無いキー（ここでは仮に次へのキーとして押す）
    sim.press("repeat", "f6")
    assert len(sim.sent) == 1


def test_gate_advance_stops_speech_before_sending(game_path, style):
    sim = Sim(style, mouse_game(game_path))
    sim.view = synth.View(scene=1, text=S[11])
    sim.run(1.8)
    assert sim.sound.playing  # 読み上げの途中
    start = len(sim.records)
    sim.press("key", "page_down")
    events = [r["event"] for r in sim.records[start:]]
    assert events.index("stop") < events.index("sent")
    assert not sim.sound.playing
    assert sim.sent[-1].do == "wheel" and sim.sent[-1].amount == -1


def test_gate_rows_on_choice_move_and_pick(game_path, style):
    """選択肢で、下へ3回、上へ1回、決めるで、乗せた行と押した行が合う。最初の下は1行目に乗せる。
    すでに選ばれている行に乗せても、乗せるたびにその項目を読む。"""
    config = mouse_game(game_path)
    sim = Sim(style, config)
    items = ["一つ目の項目", "二つ目の項目", "三つ目の項目"]
    rows = config.choice.rows_for(3)
    centers = [((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in rows]
    sim.view = synth.View(scene=1, choices=items, hint=CHOICE_HINT, choice_dark=True, selected=-1)
    sim.run(3.0)
    # どの行も暗くない（マウスが行の外）なら選ばれていない。11-07 に「1行目を選んでいる」から改めた
    assert sim.said("choice")[0] == "選択肢。" + "".join(f"{i}。" for i in items)
    for key, hovered in (("down", 0), ("down", 1), ("down", 2), ("up", 1)):
        before = len(sim.said("choice"))
        sim.press("cursor", key)
        assert sim.sent[-1].do == "move" and sim.sent[-1].at == centers[hovered]
        assert sim.said("choice")[before:] == [items[hovered]]  # キーのすぐあとに1回だけ読む
        sim.view.selected = hovered  # ゲームは乗せた行を暗くする
        sim.run(1.0)
        assert sim.said("choice")[before:] == [items[hovered]]  # 光が移っても読み直さない
    sim.press("advance", "enter")
    assert sim.sent[-1].do == "click" and sim.sent[-1].at == centers[1]
    assert len(sim.sent) == 5


def test_rows_stop_at_the_ends_and_reset_on_new_screen(game_path, style):
    sim = Sim(style, mouse_game(game_path))
    sim.view = synth.View(scene=1, hint=SAVE_HINT)
    sim.run(2.0)
    sim.press("advance", "enter")  # まだどの行にも乗せていないので押さない
    assert sim.sent == []
    for _ in range(5):
        sim.press("cursor", "down")
    sim.press("cursor", "up")
    sim.press("key", "backspace")
    got = [(a.do, a.at, a.button) for a in sim.sent]
    centers = [((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in SAVE_ROWS]
    assert got[:6] == [("move", centers[i], "left") for i in (0, 1, 2, 2, 2, 1)]
    assert got[6] == ("click", (20, 20), "right")
    sim.view = synth.View(scene=1, text=S[2])
    sim.run(2.0)
    sim.view = synth.View(scene=1, hint=SAVE_HINT)
    sim.run(2.0)
    sim.press("cursor", "down")
    assert sim.sent[-1].at == centers[0]  # 入り直したら1行目から


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == name]


def _function_of(tree: ast.AST, target: ast.AST) -> str:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(child is target for child in ast.walk(node)):
            inner = [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n is not node
                     and any(c is target for c in ast.walk(n))]
            if not inner:
                return node.name
    return ""


def test_gate_sending_is_in_one_place_and_called_only_from_keys():
    """送る関数は sender.py にだけあり、送る口を呼ぶのは判断の中心の on_command の1か所だけ。"""
    low = re.compile(r"SendInput|SetCursorPos|mouse_event")
    for path in sorted(BODY.rglob("*.py")):
        if path.name != "sender.py":
            assert not low.search(path.read_text(encoding="utf-8")), path.name
    sends = []
    for path in sorted(BODY.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in _calls(tree, "send"):
            sends.append((path.name, _function_of(tree, call)))
    assert sends == [("director.py", "_send_for_key")]
    tree = ast.parse((BODY / "director.py").read_text(encoding="utf-8"))
    callers = [_function_of(tree, c) for c in _calls(tree, "_send_for_key")]
    assert callers == ["on_command"]


def test_gate_body_sends_no_keyboard_input():
    """本体はキーボードの入力を送らない。送る語が本体のどこにも無い。"""
    words = re.compile(r"keybd_event|KEYEVENTF|INPUT_KEYBOARD|KeyboardInput|PostMessage|SendMessage|"
                       r"pyautogui|pynput")
    found = [f"{p.name}:{n}" for p in sorted(BODY.rglob("*.py"))
             for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if words.search(line)]
    assert found == []


def test_mapped_keys_are_watched_and_become_key_commands(game_path):
    config = mouse_game(game_path)
    assert config.input_keys() == {"enter", "page_down", "up", "down", "backspace"}
    mapper = KeyMapper(config.advance_keys, config.cursor_keys, mapped=sorted(config.input_keys()))
    assert mapper.on_press("page_down", 1.0).kind == "key"
    assert mapper.on_press("enter", 1.0).kind == "advance"  # 元の役目のまま
    assert KeyMapper(["enter"], ["lbutton"]).on_press("lbutton", 1.0).kind == "cursor"


def test_own_clicks_are_not_counted_as_the_user(monkeypatch):
    import time
    import sotozuke.inputs as inputs
    states = [{"lbutton"}, set(), {"lbutton"}, set()]
    own = [True, False]  # 1回目は本体が押した、2回目は利用者が押した
    got = []
    monkeypatch.setattr(inputs, "_key_state", lambda names: states.pop(0) if states else set())
    monkeypatch.setattr(inputs, "_open_xinput", lambda: None)

    def is_own(name):
        return own.pop(0) if own else False

    watcher = InputWatcher(KeyMapper([], ["lbutton"]), got.append, interval_s=0.001, is_own=is_own)
    watcher.start()
    deadline = time.time() + 1.0
    while not got and time.time() < deadline:
        time.sleep(0.005)
    time.sleep(0.02)
    watcher.stop()
    assert [c.source for c in got] == ["lbutton"]


@pytest.mark.parametrize("change", [
    lambda d: d["input"].update({"どこにも無い画面": {"keys": False}}),
    lambda d: d["input"]["text"]["actions"].update({"f6": {"do": "click", "at": [1, 1]}}),
    lambda d: d["input"]["text"]["actions"].update({"z": {"do": "click", "at": [99999, 1]}}),
    lambda d: d["input"]["text"]["actions"].update({"z": {"do": "wheel", "at": [1, 1]}}),
    lambda d: d["input"]["text"]["actions"].update({"z": {"do": "tap", "at": [1, 1]}}),
    lambda d: d["input"]["text"]["actions"].update({"z": {"do": "click", "at": [1, 1], "as": "jump"}}),
    lambda d: d["input"]["セーブ"]["actions"].update({"up": {"do": "click", "at": [1, 1]}}),
    lambda d: d["input"]["text"]["actions"].update({"lbutton": {"do": "click", "at": [1, 1]}}),
])
def test_bad_input_tables_are_rejected(game_path, change):
    mouse_game(game_path)
    data = json.loads(game_path.read_text(encoding="utf-8"))
    change(data)
    game_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(game_path)


def test_sender_turns_window_points_into_screen_points(monkeypatch):
    """送る部品は、窓の中の座標に窓の左上を足して送る。窓が無ければ送らない。実際のマウスは動かさない。"""
    import time
    import sotozuke.sender as sender
    from sotozuke.game_config import MouseAction
    done = []
    monkeypatch.setattr(sender, "move_to", lambda x, y: done.append(("move", x, y)))
    monkeypatch.setattr(sender, "click_at", lambda x, y, button: done.append(("click", x, y, button)))
    monkeypatch.setattr(sender, "wheel_at", lambda x, y, amount: done.append(("wheel", x, y, amount)))
    origins = [(1000, 50), (1000, 50), (1000, 50), None]
    box = sender.MouseSender(lambda: origins.pop(0))
    box.send(MouseAction("move", (10, 20)))
    box.send(MouseAction("click", (30, 40), button="right"))
    box.send(MouseAction("wheel", (5, 5), amount=-2))
    box.send(MouseAction("click", (1, 1)))
    box.close()
    box.thread.join(1.0)
    assert done == [("move", 1010, 70), ("click", 1030, 90, "right"), ("wheel", 1005, 55, -2)]
    assert box.is_own("lbutton") and not box.is_own("enter")
    box.last_own = time.monotonic() - 1.0
    assert not box.is_own("lbutton")
