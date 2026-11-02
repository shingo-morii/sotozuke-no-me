"""画面の種類の見本を切り出す。

使い方:
    python -m ingest.template 画面.png 左 上 右 下 出力.png
"""
from __future__ import annotations

import sys

from PIL import Image


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 6:
        print(__doc__)
        return 2
    source, x0, y0, x1, y1, out = args
    Image.open(source).convert("RGB").crop((int(x0), int(y0), int(x1), int(y1))).save(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
