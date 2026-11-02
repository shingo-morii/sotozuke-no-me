"""回帰テスト。それまでの全フェイズの関門を1つのコマンドで流す。

使い方:
    python run_gates.py            公開してよいテストと検査
    python run_gates.py --api      API を呼ぶ関門も流す
    python run_gates.py --local    手元の実画面の関門も流す（公開しない）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(title: str, cmd: list[str]) -> bool:
    print(f"--- {title}", flush=True)
    code = subprocess.run(cmd, cwd=ROOT).returncode
    print(f"--- {title}: {'通過' if code == 0 else '失敗'}", flush=True)
    return code == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    markers = []
    if not args.api:
        markers.append("not api")
    if not args.local:
        markers.append("not local")
    cmd = [sys.executable, "-m", "pytest", "tests"]
    if markers:
        cmd += ["-m", " and ".join(markers)]

    results = [
        run("テスト", cmd),
        run("公開の検査", [sys.executable, "check_public.py"]),
    ]
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    ok = all(results)
    print("回帰テスト: " + ("全部通過" if ok else "失敗あり"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
