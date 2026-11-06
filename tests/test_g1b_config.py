"""G1b の関門。設定の新しい書き方を読めること、書いていない設定は前と同じに読むこと。

基準のゲームの設定は .local にあり、コミットしない。変える前に読んだ結果を .local/g1b/config_before.json に
残してあり、今のコードで読んだ結果と比べる（local の印）。

    python tests/test_g1b_config.py 設定の名前...   変える前の結果を書く。G1b の始めに1回だけ流した

設定の名前はゲームの題名を含むので、ここに書かず、残した結果から取る。
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from sotozuke.game_config import load_config  # noqa: E402

SNAPSHOT = ROOT / ".local" / "g1b" / "config_before.json"
BASE_CONFIGS = sorted(json.loads(SNAPSHOT.read_text(encoding="utf-8"))) if SNAPSHOT.exists() else []


def snapshot(config) -> dict:
    """読んだ結果を JSON にできる形にする。見本のパスは名前だけにする。"""
    def plain(value):
        if dataclasses.is_dataclass(value):
            return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, Path):
            return value.name
        return value
    data = plain(config)
    data.pop("source", None)
    return data


def _write_before(names: list[str]) -> None:
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    data = {name: snapshot(load_config(ROOT / ".local" / name)) for name in names}
    SNAPSHOT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(SNAPSHOT)


def _same_as_before(before, now, path="") -> list[str]:
    """前にあった項目は同じ値で、新しく足した項目は書いていないときの値（切）であること。"""
    if isinstance(before, dict):
        problems = []
        for key, value in before.items():
            if key not in now:
                problems.append(f"{path}.{key}: 無くなった")
            else:
                problems += _same_as_before(value, now[key], f"{path}.{key}")
        return problems
    return [] if before == now else [f"{path}: {before!r} が {now!r} になった"]


@pytest.mark.local
@pytest.mark.parametrize("name", BASE_CONFIGS)
def test_gate_base_config_reads_the_same(name):
    """関門：基準のゲームの設定を読んだ結果が、変える前と同じ。"""
    before = json.loads(SNAPSHOT.read_text(encoding="utf-8"))[name]
    now = snapshot(load_config(ROOT / ".local" / name))
    assert _same_as_before(before, now) == []
    assert_new_items_are_off(load_config(ROOT / ".local" / name))


def assert_new_items_are_off(config) -> None:
    """G1b で足した項目が、書いていない設定では全部切になっていること。フェイズごとに足す。"""
    assert config.text.end_marks == ""
    if config.tag is not None:
        assert config.tag.read_name is False
    assert config.input == {}
    assert config.input_keys() == set()
    if config.choice is not None:
        assert config.choice.selected == "bright"
    if config.scene is not None:
        assert config.scene.text_diff == 0


def test_new_items_are_off_in_synthetic_configs(game_path):
    assert_new_items_are_off(load_config(game_path))


if __name__ == "__main__":
    _write_before(sys.argv[1:])
