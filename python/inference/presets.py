"""
presets.py

Activation Steering系プリセット(Sycophancy Reducer・Refusal Behavior Tuner)の読み込み(2026-09-22追加)。

文例・定型語彙は python/presets/steering_presets.json を唯一の定義元とし、Python側
(sycophancy_utils.py・refusal_utils.py)とReact側(src/app/presets.ts)の両方がこのJSONを読む。
以前はPythonとReact(App.tsx)に同じ文例を二重に書いていたため、片方だけ直して食い違う危険があった。
"""

from __future__ import annotations

import json
from pathlib import Path

PRESETS_PATH = Path(__file__).resolve().parent.parent / "presets" / "steering_presets.json"


def load_presets() -> dict:
    with open(PRESETS_PATH, encoding="utf-8") as f:
        return json.load(f)


_PRESETS = load_presets()


def preset(name: str, lang: str = "ja") -> dict:
    """"sycophancy" / "refusal" のプリセット({positive_prompts, negative_prompts, phrases})を返す。

    lang="en" で英語UI用の文例(2026-10-13追加、JSONの "en")を返す。
    """
    return _PRESETS["en"][name] if lang == "en" else _PRESETS[name]


def phrases(name: str) -> list[str]:
    """語彙カウント用の定型語彙。画面の言語に関係なく、日本語・英語の両方の語彙を重複なしで返す
    (英語で質問しても日本語で答えるモデル、その逆もあるため)。"""
    out: list[str] = []
    for p in list(_PRESETS[name]["phrases"]) + list(_PRESETS["en"][name]["phrases"]):
        if p not in out:
            out.append(p)
    return out
