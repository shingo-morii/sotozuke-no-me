from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import synth  # noqa: E402


def pytest_collection_modifyitems(config, items):
    if synth.font_path() is None:
        skip = pytest.mark.skip(reason="日本語のフォントがありません")
        for item in items:
            item.add_marker(skip)


@pytest.fixture(params=["a", "b"])
def style(request):
    return synth.STYLES[request.param]


@pytest.fixture
def game_path(style, tmp_path):
    return synth.write_game(style, tmp_path)
