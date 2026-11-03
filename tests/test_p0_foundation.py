"""P0 の関門。土台、合成の画面、設定の読み込み、公開の検査。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import synth
from sotozuke.game_config import ConfigError, load_config, parse_config

ROOT = Path(__file__).resolve().parent.parent


def test_render_makes_expected_size(style):
    frame = synth.render(style, synth.View(text="合成の文です。"))
    assert frame.shape == (style.size[1], style.size[0], 3)
    assert frame.dtype == np.uint8


def test_two_styles_differ():
    a = synth.STYLE_A
    b = synth.STYLE_B
    assert a.size != b.size
    assert a.band != b.band
    assert a.tag_kind != b.tag_kind


def test_scene_backgrounds_differ(style):
    one = synth.background(style, 1)
    two = synth.background(style, 2)
    assert np.abs(one.astype(int) - two.astype(int)).mean() > 10


def test_load_config(game_path, style):
    config = load_config(game_path)
    assert config.size == style.size
    assert config.voice_rule == "tag"
    assert config.tag is not None
    assert {s.kind for s in config.screens} >= {"choice", "map", "walk", "menu"}
    assert config.text.band == style.band


def test_config_rejects_box_outside_screen():
    data = {"name": "x", "size": [100, 100], "text": {"band": [0, 0, 200, 10]}}
    with pytest.raises(ConfigError):
        parse_config(data)


def test_config_rejects_tag_rule_without_tag():
    data = {"name": "x", "size": [100, 100], "text": {"band": [0, 0, 50, 10]}, "voice_rule": "tag"}
    with pytest.raises(ConfigError):
        parse_config(data)


def test_config_rejects_unknown_screen_kind(tmp_path):
    (tmp_path / "t.png").write_bytes(b"x")
    data = {"name": "x", "size": [100, 100], "text": {"band": [0, 0, 50, 10]},
            "screens": [{"name": "n", "kind": "unknown", "box": [0, 0, 10, 10], "template": "t.png"}]}
    with pytest.raises(ConfigError):
        parse_config(data, base=tmp_path)


def _load_checker():
    sys.path.insert(0, str(ROOT))
    import check_public
    return check_public


def test_public_check_finds_media():
    checker = _load_checker()
    assert checker.check_media(["sotozuke/a.png"])
    assert checker.check_media(["voice.wav"])
    assert not checker.check_media(["tests/synth_media/a.png"])
    assert not checker.check_media(["sotozuke/a.py"])


def test_public_check_finds_prework_documents():
    checker = _load_checker()
    assert checker.check_prework(["docs/設計書_v4.md"])
    assert checker.check_prework(["11_2の指示書.md"])
    assert not checker.check_prework(["README.md"])


def test_public_check_finds_secret_and_forbidden(tmp_path, monkeypatch):
    checker = _load_checker()
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    (tmp_path / "a.txt").write_text("xx-secret-value-xx", encoding="utf-8")
    (tmp_path / "b.txt").write_text("安全な文", encoding="utf-8")
    problems = checker.check_contents(["a.txt", "b.txt"], [("API の鍵", "secret-value")])
    assert len(problems) == 1
    assert "a.txt" in problems[0]


def test_public_check_runs_clean():
    code = subprocess.run([sys.executable, "check_public.py"], cwd=ROOT).returncode
    assert code == 0


def test_no_game_files_in_repository():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8").stdout
    assert ".local" not in out
    assert "costs.jsonl" not in out


def test_games_folder_is_json_or_readme():
    for path in (ROOT / "games").glob("*"):
        assert path.suffix in {".json", ".md"} or path.is_dir()
    json.dumps({})


def test_stall_dump_writes_stacks_when_the_loop_stops(tmp_path):
    """1本の流れが止まったら、呼び出しの順がファイルに残る。止まった原因を調べるため。"""
    import time
    from sotozuke.app import StallDump
    path = tmp_path / "speech.stacks.txt"
    stall = StallDump(path, seconds=0.2)
    stall.arm()
    time.sleep(0.6)
    stall.close()
    assert "Thread" in path.read_text(encoding="utf-8")


def test_stall_dump_stays_quiet_while_the_loop_runs(tmp_path):
    import time
    from sotozuke.app import StallDump
    path = tmp_path / "speech.stacks.txt"
    stall = StallDump(path, seconds=0.3)
    for _ in range(8):
        stall.arm()
        time.sleep(0.1)
    stall.close()
    assert path.read_text(encoding="utf-8") == ""


def test_stall_dump_writes_stacks_on_request(tmp_path):
    from sotozuke.app import StallDump
    path = tmp_path / "speech.stacks.txt"
    stall = StallDump(path, seconds=30)
    path.with_suffix(".request").write_text("")
    stall.arm()
    stall.close()
    assert "most recent call first" in path.read_text(encoding="utf-8")
    assert not path.with_suffix(".request").exists()
