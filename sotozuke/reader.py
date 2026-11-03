"""読み手。Claude に画面を見せて、字や絵の内容を受け取る。"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
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
# Sonnet は思考を切れないので effort を下げる。地図で約6秒が約3秒になった
LOW_EFFORT = {"output_config": {"effort": "low"}}
TEXT_SCALE = 3  # 本文を読むときの拡大の倍率。読み違えた文で6回中6回正しくなった

PROMPT_TEXT = (
    "画像はノベルゲームの本文の部分です。画像に書かれている字だけを、一字も変えずに返してください。"
    "改行はつなげてください。説明、引用符、飾りは付けないでください。"
    "点だけの文は「……」と返してください。字がなければ「なし」とだけ返してください。"
)
PROMPT_ITEMS = (
    "画像はノベルゲームの選択肢です。各項目の字を、上から順に JSON の文字列の配列だけで返してください。"
    "項目の先頭の記号や A、B のような印は含めないでください。ほかの文は書かないでください。"
)
PROMPT_SCENE = (
    "ノベルゲームの場面の絵を、目の見えない人に伝えます。場所、時間帯、人物と様子を、2文以内、60字以内の"
    "日本語で書いてください。画面の文字は読まないでください。説明の文だけを返してください。"
)


PROMPT_MENU = (
    "画像はゲームのメニュー画面です。次の形の JSON だけを返してください。"
    '{"title": "画面の題", "items": ["上から順の項目"], "selected": "いま選ばれている項目の字"}。'
    "選ばれている項目は、色や明るさが周りと違う項目です。データの枠なら、枠の中の字を1つの項目にしてください。"
)
PROMPT_CONFIRM = (
    "画像はゲームの確認の窓が重なった画面です。窓の中だけを見て、次の形の JSON だけを返してください。"
    '{"message": "窓の中の文", "selected": "いま選ばれているボタンの字"}'
)
PROMPT_MAP = (
    "画像はノベルゲームの行き先を選ぶ地図です。目の見えない人に伝えます。選べる場所と、そこにいる人の見た目、"
    "いま選ばれている行き先を、100字以内の日本語で書いてください。説明の文だけを返してください。"
)
PROMPT_MAP_MOVE = (
    "画像はノベルゲームの行き先を選ぶ地図です。いま選ばれている行き先の名前と、そこにいる人の見た目を、"
    "40字以内の日本語で書いてください。説明の文だけを返してください。"
)
PROMPT_WALK = (
    "画像はゲームの歩く画面です。画面に場所の名前が書かれていれば、その名前だけを返してください。"
    "書かれていなければ何も返さないでください。"
)
PROMPT_WALK_VIEW = (
    "画像はゲームの中を歩く画面です。目の見えない人に伝えます。画面の手前に背中を見せている人物は操作している主人公です。"
    "それ以外の人物が、正面、左、右のどこに、近いか遠いかと、髪や服の見た目、進める通路の向き、"
    "横へ入れる入口や分かれ道を、100字以内の日本語で書いてください。説明の文だけを返してください。"
)
PROMPT_BACKLOG = (
    "画像はゲームの文字の履歴の画面です。いちばん明るく表示されている文だけを、一字も変えずに返してください。"
    "説明や飾りは付けないでください。"
)
PROMPT_ALL = (
    "画像はゲームの画面です。画面に書かれている字を、読む順に、そのまま返してください。"
    "説明や飾りは付けないでください。字がなければ何も返さないでください。"
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
    # 文の途中の ？ や ！ のあとの空きは、幅の読み取りが揺れるので全角の空きにそろえる
    text = re.sub(r"([？！])[ 　]*(?=[^？！」』）\s])", "\\1　", text)
    return text.strip()


REASONING = re.compile(r"ユーザーの質問|回答案|字数約|最終的な答え|考えた過程")


def drop_reasoning(text: str) -> str:
    """考えの途中が混ざった答えから、最後の答えの文だけを残す。頼んでも考えを書くことがある。"""
    if not REASONING.search(text):
        return text
    for marks in (r"[)）」]", r"[:：]"):
        tail = re.split(marks, text)[-1].strip(" 　:：")
        if len(tail) >= 6 and not REASONING.search(tail):
            return tail
    quoted = [q for q in re.findall(r"「([^「」]{6,})」", text) if not REASONING.search(q)]
    return quoted[-1] if quoted else ""


def same_text(a: str, b: str) -> bool:
    """読み上げの音が同じなら同じ文と見なす。数字の全角半角、かぎかっこの種類、空きは問わない。"""
    def shape(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        return re.sub(r"[「」『』\s]", "", text)
    return shape(a) == shape(b)


def is_no_text(text: str) -> bool:
    """字が無いという返事か。読み手は頼んでも文で答えることがある。"""
    if text in ("", "なし", "（なし）", "無し") or "何も返" in text:
        return True
    return bool(re.search(r"(画像|文字|テキスト)", text) and re.search(r"(ません|ない)[。.]?$", text))


def parse_items(raw: str) -> list[str]:
    """JSON の配列を取り出す。余分な文が付いていても探す。"""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ReaderError(f"項目の配列が見つからない: {raw[:80]}")
    items = json.loads(match.group(0))
    if not (isinstance(items, list) and all(isinstance(i, str) for i in items)):
        raise ReaderError(f"項目の形が違う: {raw[:80]}")
    return [clean_text(i) for i in items]


def parse_object(raw: str) -> dict:
    """JSON のオブジェクトを取り出す。"""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ReaderError(f"JSON が見つからない: {raw[:80]}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ReaderError(f"JSON の形が違う: {raw[:80]}")
    return data


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


def enlarge(image: np.ndarray, scale: int = TEXT_SCALE) -> np.ndarray:
    """小さな字は拡大してから見せる。そのままだと「んじゃ」を「ンジャ」と読むことがあった。"""
    picture = Image.fromarray(image)
    return np.asarray(picture.resize((picture.width * scale, picture.height * scale), Image.BICUBIC))


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
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="read")
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

    def _read_once(self, image: np.ndarray, purpose: str) -> ReadResult:
        result = self.ask(image, PROMPT_TEXT, TEXT_MODEL, purpose, thinking=NO_THINKING)
        result.text = clean_text(result.text)
        return result

    def read_text(self, band: np.ndarray, on_check=None) -> ReadResult:
        """本文を読む。そのままと拡大の2通りで同時に読み、割れたら Sonnet に決めてもらう。

        Haiku は文の2%ほどで1字読み違えるが、どちらで間違えるかは文によって違う。
        """
        plain = self.pool.submit(self._read_once, band, "text")
        large = self.pool.submit(self._read_once, enlarge(band), "text_large")
        first, second = plain.result(), large.result()
        usd = first.usd + second.usd
        seconds = max(first.seconds, second.seconds)
        if same_text(first.text, second.text):
            result = first
        else:
            if on_check:
                on_check()  # 読みが割れたので時間がかかると知らせる
            result = self.ask(band, PROMPT_TEXT, PICTURE_MODEL, "text_check", max_tokens=1000,
                              extra_body=LOW_EFFORT)
            result.text = clean_text(result.text)
            usd += result.usd
            seconds += result.seconds
        result.usd, result.seconds = usd, seconds
        if is_no_text(result.text):
            result.text = ""
        return result

    def read_items(self, area: np.ndarray) -> tuple[list[str], ReadResult]:
        result = self.ask(area, PROMPT_ITEMS, TEXT_MODEL, "items", thinking=NO_THINKING)
        return parse_items(result.text), result

    def describe(self, frame: np.ndarray) -> ReadResult:
        result = self.ask(frame, PROMPT_SCENE, PICTURE_MODEL, "scene", max_tokens=1500,
                          extra_body=LOW_EFFORT)
        result.text = limit_description(result.text)
        return result

    def read_menu(self, frame: np.ndarray) -> tuple[dict, ReadResult]:
        result = self.ask(frame, PROMPT_MENU, TEXT_MODEL, "menu", thinking=NO_THINKING, max_tokens=800)
        return parse_object(result.text), result

    def read_confirm(self, frame: np.ndarray) -> tuple[dict, ReadResult]:
        result = self.ask(frame, PROMPT_CONFIRM, TEXT_MODEL, "confirm", thinking=NO_THINKING)
        return parse_object(result.text), result

    def read_map(self, frame: np.ndarray, moved: bool, prompt: str = "") -> ReadResult:
        prompt = prompt or (PROMPT_MAP_MOVE if moved else PROMPT_MAP)
        result = self.ask(frame, prompt, PICTURE_MODEL, "map", max_tokens=1500, extra_body=LOW_EFFORT)
        result.text = drop_reasoning(clean_text(result.text))
        return result

    def read_plain(self, frame: np.ndarray, prompt: str, purpose: str) -> ReadResult:
        result = self.ask(frame, prompt, TEXT_MODEL, purpose, thinking=NO_THINKING)
        result.text = drop_reasoning(clean_text(result.text))
        if is_no_text(result.text):
            result.text = ""
        return result
