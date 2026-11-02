"""API の費用の記録。1回の呼び出しごとに costs.jsonl へ1行書く。

料金は 11/2 に公式の文書で確かめた値。Haiku 5.5 は「0.10 ドルから」と書いてあり、
上限が読み取れなかったので、費用を少なく見積もらないよう高めの値を使う。
"""
from __future__ import annotations

import os
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# 100万トークンあたりのドル。(入力, 出力)
PRICES = {
    "claude-haiku-5-5": (1.0, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0),
}


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CostLog:
    path: Path
    limit_usd: float = float(os.environ.get("SOTOZUKE_BUDGET_USD", "4.5"))

    def __post_init__(self):
        self.path = Path(self.path)
        self._lock = threading.Lock()

    def add(self, purpose: str, model: str, input_tokens: int, output_tokens: int,
            seconds: float) -> float:
        cost = cost_of(model, input_tokens, output_tokens)
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "purpose": purpose, "model": model,
               "input_tokens": input_tokens, "output_tokens": output_tokens,
               "seconds": round(seconds, 3), "usd": round(cost, 6)}
        with self._lock, self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        return cost

    def total(self) -> float:
        if not self.path.exists():
            return 0.0
        with self.path.open(encoding="utf-8") as file:
            return sum(json.loads(line)["usd"] for line in file if line.strip())

    def check(self) -> None:
        if self.total() >= self.limit_usd:
            raise BudgetExceeded(f"API の費用が上限 {self.limit_usd} ドルに届いた")


def main() -> None:
    import sys
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "costs.jsonl")
    log = CostLog(path)
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"呼び出し {len(rows)} 回、累計 {log.total():.4f} ドル")
    by_purpose: dict[str, tuple[int, float]] = {}
    for row in rows:
        count, usd = by_purpose.get(row["purpose"], (0, 0.0))
        by_purpose[row["purpose"]] = (count + 1, usd + row["usd"])
    for purpose, (count, usd) in sorted(by_purpose.items()):
        print(f"  {purpose}: {count} 回、{usd:.4f} ドル")


if __name__ == "__main__":
    main()
