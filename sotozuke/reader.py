"""読み手。Claude に画面を見せて、字や絵の内容を受け取る。"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .costs import CostLog

TEXT_MODEL = "claude-haiku-5-5"
PICTURE_MODEL = "claude-sonnet-5-5"
# 字を読むだけなら思考は要らない。11/2 に10枚で比べて、正しさは同じだった
NO_THINKING = {"type": "disabled"}

PROMPT_TEXT = (
    "画像はノベルゲームの本文の部分です。画像に書かれている字だけを、一字も変えずに返してください。"
    "改行はつなげてください。説明、引用符、飾りは付けないでください。"
    "字がなければ何も返さないでください。"
)
PROMPT_ITEMS = (
    "画像はノベルゲームの選択肢です。各項目の字を、上から順に JSON の文字列の配列だけで返してください。"
    "項目の先頭の記号や A、B のような印は含めないでください。ほかの文は書かないでください。"
)
PROMPT_SCENE = (
    "ノベルゲームの場面の絵を、目の見えない人に伝えます。場所、時間帯、人物と様子を、2文以内、60字以内の"
    "日本語で書いてください。画面の文字は読まないでください。説明の文だけを返してください。"
)


class ReaderError(RuntimeError):
    pass


@dataclass
class ReadResult:
    text: str
    seconds: float
    usd: float
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


def clean_text(raw: str) -> str:
    """飾りと改行を取り除く。"""
    text = raw.strip()
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"\s*\n\s*", "", text)
    return text.strip()


def parse_items(raw: str) -> list[str]:
    """JSON の配列を取り出す。余分な文が付いていても探す。"""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ReaderError(f"項目の配列が見つからない: {raw[:80]}")
    items = json.loads(match.group(0))
    if not (isinstance(items, list) and all(isinstance(i, str) for i in items)):
        raise ReaderError(f"項目の形が違う: {raw[:80]}")
    return [clean_text(i) for i in items]


def limit_description(text: str, max_sentences: int = 2, max_chars: int = 60) -> str:
    """説明を2文以内、60字以内に収める。文の途中で切らない。"""
    sentences = [s for s in re.findall(r"[^。！？]+[。！？]?", clean_text(text)) if s.strip()]
    out = ""
    for sentence in sentences[:max_sentences]:
        if len(out) + len(sentence) > max_chars:
            break
        out += sentence
    if not out and sentences:
        out = sentences[0][:max_chars]
    return out


def encode_png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, "PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def response_text(response: Any) -> str:
    """text の部分を探す。先頭が thinking のことがあるので決め打ちしない。"""
    parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
    return "".join(parts)


def load_key() -> str:
    path = os.environ.get("ANTHROPIC_API_KEY_FILE")
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    raise ReaderError("鍵がありません。ANTHROPIC_API_KEY_FILE を設定してください")


@dataclass
class ClaudeReader:
    costs: CostLog
    client: Any = None
    timeout_s: float = 20.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic(api_key=load_key(), timeout=self.timeout_s,
                                              max_retries=1)

    def ask(self, image: np.ndarray, prompt: str, model: str, purpose: str,
            max_tokens: int = 500, **options) -> ReadResult:
        self.costs.check()
        start = time.perf_counter()
        response = self.client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": encode_png(image)}},
                {"type": "text", "text": prompt},
            ]}],
            **{**self.extra, **options},
        )
        seconds = time.perf_counter() - start
        usage = response.usage
        usd = self.costs.add(purpose, model, usage.input_tokens, usage.output_tokens, seconds)
        return ReadResult(response_text(response), seconds, usd, model,
                          usage.input_tokens, usage.output_tokens)

    def read_text(self, band: np.ndarray) -> ReadResult:
        result = self.ask(band, PROMPT_TEXT, TEXT_MODEL, "text", thinking=NO_THINKING)
        result.text = clean_text(result.text)
        return result

    def read_items(self, area: np.ndarray) -> tuple[list[str], ReadResult]:
        result = self.ask(area, PROMPT_ITEMS, TEXT_MODEL, "items", thinking=NO_THINKING)
        return parse_items(result.text), result

    def describe(self, frame: np.ndarray) -> ReadResult:
        result = self.ask(frame, PROMPT_SCENE, PICTURE_MODEL, "scene")
        result.text = limit_description(result.text)
        return result
