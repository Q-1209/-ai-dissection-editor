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


def preset(name: str) -> dict:
    """"sycophancy" / "refusal" のプリセット({positive_prompts, negative_prompts, phrases})を返す。"""
    return _PRESETS[name]
