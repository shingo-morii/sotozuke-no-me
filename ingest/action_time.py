"""探索の記録を数える。動作時間、キーとマウスの回数、目標に届いた時刻を出す。

使い方:
    python -m ingest.action_time work/actions.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"rows": 0}
    start = rows[0]["time"]
    marks = [(r["name"], round(r["time"] - start, 1)) for r in rows if r["action"] == "mark"]
    progress = 0.0
    opened = None
    for name, seconds in marks:
        if name == "進行開始":
            opened = seconds
        elif name == "進行終了" and opened is not None:
            progress += seconds - opened
            opened = None
    return {
        "total_s": round(rows[-1]["time"] - start, 1),
        "keys": sum(1 for r in rows if r["action"] == "key"),
        "clicks": sum(1 for r in rows if r["action"] == "click"),
        "moves": sum(1 for r in rows if r["action"] == "move"),
        "wheels": sum(1 for r in rows if r["action"] == "wheel"),
        "shots": sum(1 for r in rows if r["action"] == "shot"),
        "progress_s": round(progress, 1),
        "marks": marks,
    }


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    print(json.dumps(summarize(Path(args[0])), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
