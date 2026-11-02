"""公開の前の検査。コミットのたびに流す。

検査の内容:
1. 鍵のファイルの中身と同じ文字列が、コミットするファイルと履歴に無いこと
2. .local/forbidden.txt の文字列が、コミットするファイルと履歴に無いこと
3. tests/ の合成の画面のほかに、画像と音声のファイルが無いこと
4. 開始前の文書がリポジトリに無いこと
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    ".wav", ".mp3", ".ogg", ".flac", ".m4a", ".mp4", ".webm",
}
ALLOWED_MEDIA_DIR = "tests/synth_media/"
# 開始前の文書の名前の一部。ファイル名で確かめる
PREWORK_NAME_PARTS = [
    "設計書", "指示書", "手順の下書き", "探索の見直し", "初見探索",
    "参加者ガイド", "応募の回答", "デモ動画の構成案", "過去の受賞作",
]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=True,
    ).stdout


def tracked_files() -> list[str]:
    """コミット済みと、これからコミットするファイルの一覧。"""
    out = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return [p for p in out.split("\0") if p]


def read_secret() -> str | None:
    path = os.environ.get("ANTHROPIC_API_KEY_FILE")
    if not path or not Path(path).exists():
        return None
    return Path(path).read_text(encoding="utf-8").strip()


def read_forbidden() -> list[str]:
    path = ROOT / ".local" / "forbidden.txt"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [s.strip() for s in lines if s.strip() and not s.startswith("#")]


def check_contents(files: list[str], needles: list[tuple[str, str]]) -> list[str]:
    problems = []
    for rel in files:
        path = ROOT / rel
        if not path.is_file():
            continue
        data = path.read_bytes()
        for label, needle in needles:
            if needle.encode("utf-8") in data:
                problems.append(f"{rel}: {label}が含まれています")
    return problems


def check_history(needles: list[tuple[str, str]]) -> list[str]:
    """コミットの文面にも同じ文字列が無いことを確かめる。"""
    try:
        log = git("log", "--all", "--format=%B")
    except subprocess.CalledProcessError:
        return []  # コミットが1つも無いとき
    return [f"コミットの文面: {label}が含まれています"
            for label, needle in needles if needle in log]


def check_media(files: list[str]) -> list[str]:
    return [
        f"{rel}: 画像か音声のファイルです"
        for rel in files
        if Path(rel).suffix.lower() in BINARY_SUFFIXES
        and not rel.replace("\\", "/").startswith(ALLOWED_MEDIA_DIR)
    ]


def check_prework(files: list[str]) -> list[str]:
    return [
        f"{rel}: 開始前の文書の名前です"
        for rel in files
        if any(part in Path(rel).name for part in PREWORK_NAME_PARTS)
    ]


def main() -> int:
    files = tracked_files()
    needles: list[tuple[str, str]] = []
    secret = read_secret()
    if secret:
        needles.append(("API の鍵", secret))
    needles += [("禁止の文字列", s) for s in read_forbidden()]

    problems = check_contents(files, needles)
    problems += check_history(needles)
    problems += check_media(files)
    problems += check_prework(files)

    if problems:
        print("公開の検査で問題が見つかりました。")
        for p in problems:
            print(" ", p)
        return 1
    print(f"公開の検査は通りました。{len(files)} ファイル、"
          f"鍵の検査 {'あり' if secret else 'なし'}、禁止の文字列 {len(needles) - bool(secret)} 件。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
