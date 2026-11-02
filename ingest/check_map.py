"""設定を、撮りためた画面全部に当てて、種類ごとの当たりと外れを数える。

画面はフォルダの名前を正解にする。例: corpus/選択肢/0001.png
見本の基準は、当たりの最小とほかの画面の最大の間に余裕があることを確かめる。外れた画面は開いて理由を書く。

使い方:
    python -m ingest.check_map games/ゲーム.json corpus
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from sotozuke.game_config import load_config
from sotozuke.watcher import ScreenClassifier, has_name_tag


def check(config_path: str, corpus: str, step: int = 1) -> dict:
    config = load_config(config_path)
    classifier = ScreenClassifier(config)
    report = {}
    for folder in sorted(p for p in Path(corpus).iterdir() if p.is_dir()):
        files = sorted(folder.glob("*.png"))[::step]
        counts: dict[str, int] = {}
        best: dict[str, float] = {}
        tags = 0
        for file in files:
            frame = np.asarray(Image.open(file).convert("RGB"))
            if frame.shape[1::-1] != config.size:
                counts["大きさが違う"] = counts.get("大きさが違う", 0) + 1
                continue
            screen = classifier.classify(frame)
            name = screen.name if screen else "文"
            counts[name] = counts.get(name, 0) + 1
            tags += has_name_tag(frame, config)
            for rule, score in classifier.scores(frame):
                best[rule.name] = max(best.get(rule.name, 0.0), score)
        report[folder.name] = {"files": len(files), "counts": counts, "tags": tags,
                               "best": {k: round(v, 2) for k, v in best.items()}}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="check_map")
    parser.add_argument("config")
    parser.add_argument("corpus")
    parser.add_argument("--step", type=int, default=1)
    options = parser.parse_args(argv)
    for name, row in check(options.config, options.corpus, options.step).items():
        print(name, row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
