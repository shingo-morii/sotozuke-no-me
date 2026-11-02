"""合成のゲームの時間の流れ。時刻を渡すと、その時刻の画面を返す。"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

import synth

SAMPLE_SENTENCES = [
    "朝の光が窓から差しこんでいた。", "駅までの道はいつもより静かだ。",
    "今日は少し早く家を出ることにした。", "冷たい風が頬をなでていく。",
    "校門の前で友達が手を振っている。", "昨日の宿題、もう終わった？",
    "図書室の窓から桜の木が見える。", "雨の音だけが部屋に響いていた。",
    "そんなこと、急に言われても困る。", "時計の針は八時を指していた。",
    "商店街の角を曲がると海が見えた。", "彼女は小さくうなずいて歩き出した。",
    "もう少しだけ、ここにいてもいいかな。", "空の色が少しずつ変わっていく。",
    "階段を上がる足音が近づいてくる。", "机の上に一通の手紙が置かれていた。",
    "お昼は屋上で食べようよ。", "教室の中はざわざわしている。",
    "ふと見上げると、月が雲に隠れていた。", "明日の約束、忘れないでね。",
]


@dataclass
class Segment:
    start: float
    end: float
    view: synth.View
    typing_cps: float | None = None   # 1秒の字数。None なら一度に出る
    fade: str = ""                    # "down" か "up" か "hold"
    label: str = ""                   # 正解の印


@dataclass
class Timeline:
    style: synth.Style
    segments: list[Segment] = field(default_factory=list)
    sentences: list[Segment] = field(default_factory=list)
    dawns: list[float] = field(default_factory=list)

    @property
    def end(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    def at(self, t: float) -> synth.View:
        for seg in self.segments:
            if seg.start <= t < seg.end:
                return self._view(seg, t)
        return self.segments[-1].view

    def frame(self, t: float) -> np.ndarray:
        return synth.render(self.style, self.at(t), t)

    def _view(self, seg: Segment, t: float) -> synth.View:
        view = seg.view
        kwargs = dict(view.__dict__)
        if seg.typing_cps:
            kwargs["visible_chars"] = int((t - seg.start) * seg.typing_cps) + 1
        progress = (t - seg.start) / max(seg.end - seg.start, 1e-6)
        if seg.fade == "down":
            kwargs["dark"] = min(1.0, progress)
        elif seg.fade == "hold":
            kwargs["dark"] = 1.0
        elif seg.fade == "up":
            kwargs["dark"] = max(0.0, 1.0 - progress)
        return synth.View(**kwargs)


def build(style: synth.Style, count: int = 40, seed: int = 1, typing_cps: float | None = None,
          blackouts: int = 5, gap_s: float = 0.25) -> Timeline:
    """文を count 個と暗転を blackouts 回並べる。"""
    rng = random.Random(seed)
    line = Timeline(style)
    t = 0.0
    scene = 1
    every = max(1, count // (blackouts + 1)) if blackouts else count + 1
    for i in range(count):
        if blackouts and i and i % every == 0:
            for fade, dur in (("down", 1.0), ("hold", rng.uniform(2.0, 4.0))):
                line.segments.append(Segment(t, t + dur, synth.View(scene=scene), fade=fade))
                t += dur
            scene += 1
            line.segments.append(Segment(t, t + 1.0, synth.View(scene=scene), fade="up"))
            t += 1.0
            line.dawns.append(t)
            hold = rng.uniform(0.3, 1.0)
            line.segments.append(Segment(t, t + hold, synth.View(scene=scene)))
            t += hold
        text = SAMPLE_SENTENCES[rng.randrange(len(SAMPLE_SENTENCES))]
        name = "名前" if rng.random() < 0.4 else ""
        length = len(text) / typing_cps if typing_cps else 0.0
        duration = length + rng.uniform(0.7, 3.0)
        seg = Segment(t, t + duration, synth.View(scene=scene, text=text, name=name),
                      typing_cps=typing_cps, label="sentence")
        line.segments.append(seg)
        line.sentences.append(seg)
        t += duration
        if gap_s:
            line.segments.append(Segment(t, t + gap_s, synth.View(scene=scene)))
            t += gap_s
    return line
