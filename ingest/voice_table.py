"""声の見分け方を調べる表を作る。

文を1つずつ進めて、キーを押す前と撮った後の時刻、ゲームの音の大きさ、札の有無を1行ずつ残す。
結論を書く前に、この表を機械で作る。効果音があるので、音の大きさだけで決めない。

使い方:
    python -m ingest.voice_table --process Game.exe --work work --config games/ゲーム.json --count 40 --key enter
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import comtypes
from PIL import Image

from ingest.gamectl import Control
from sotozuke.game_audio import open_game_meters
from sotozuke.game_config import load_config
from sotozuke.watcher import has_name_tag


def peak(meters) -> float:
    return max((m.GetPeakValue() for m in meters), default=0.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="voice_table")
    parser.add_argument("--process", required=True)
    parser.add_argument("--work", default="work")
    parser.add_argument("--config")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--key", default="enter")
    parser.add_argument("--seconds", type=float, default=2.5, help="1文を見る秒数")
    options = parser.parse_args(argv)

    comtypes.CoInitialize()
    control = Control(options.process, Path(options.work))
    config = load_config(options.config) if options.config else None
    folder = Path(options.work) / "voice"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(options.count):
        before = time.time()
        control.key(options.key)
        levels = []
        meters = open_game_meters(options.process)
        end = time.perf_counter() + options.seconds
        while time.perf_counter() < end:
            levels.append(round(peak(meters), 3))
            time.sleep(0.05)
        frame = control.capture.grab()
        after = time.time()
        if frame is None:
            print("窓が見つかりません")
            return 1
        Image.fromarray(frame).save(folder / f"{i:03d}.png", compress_level=1)
        active = sum(1 for level in levels if level >= 0.07) * 0.05
        row = {"index": i, "before": round(before, 3), "after": round(after, 3),
               "max_level": max(levels, default=0), "active_s": round(active, 2),
               "tag": bool(has_name_tag(frame, config)) if config and config.tag else None}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (folder / "table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
