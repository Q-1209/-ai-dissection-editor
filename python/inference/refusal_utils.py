"""
refusal_utils.py

Phase3: Refusal Behavior Tuner(拒否挙動の調整)の中核ロジック(torch非依存)。

2026-09-22、漣さんとの確認事項:
- 調整の方向: 両方向。拒否を減らす方向・増やす方向の両方を扱う
- 介入方式: 既存のActivation Steering Studioの「加算」に加えて、拒否方向の成分だけを
  活性化値から射影で取り除く「方向除去」も実装する(Arditi et al. 2024の手法と同じ考え方)
  - 加算・scale正 → 拒否を増やす / 加算・scale負 → 拒否を減らす
  - 方向除去 → 拒否を減らす(その方向の成分をゼロにする、出力が崩壊しにくい)
- チャットテンプレート: 全タブ共通で対応(tokenizer_utils.py参照)。拒否は「応答」モードでしか
  出ないため、この機能ではチャットテンプレート適用が事実上の前提になる
- 評価: Sycophancy Reducerと同じ。生成テキストの目視比較+拒否の定型句の出現回数カウント。
  対照プロンプトは固定プリセットを内蔵し、GUIの「Refusal Tunerプリセット」ボタン1つで流し込む

ベクトルの向き: positive=拒否を引き出す依頼文、negative=構造がほぼ同じで無害な依頼文。
差分(positive - negative)が「拒否方向」になる。positive/negativeは内容以外の違い
(長さ・文体)がなるべく出ないよう、1対1で対応する文例にしてある。
"""

from __future__ import annotations

from .sycophancy_utils import count_sycophantic_phrases

from .presets import phrases, preset

# 文例・定型語彙の実体は python/presets/steering_presets.json("refusal")にある(2026-09-22、一本化)。
# REFUSAL_POSITIVE_PROMPTS: 拒否を引き出しやすい依頼文(実行可能な詳細を含まない短い依頼文)
# REFUSAL_NEGATIVE_PROMPTS: 上と1対1で対応する、構造がほぼ同じで無害な依頼文
# DEFAULT_REFUSAL_PHRASES: 拒否応答によく現れる定型句(「できません」単体のような広すぎる語は含めない)
REFUSAL_POSITIVE_PROMPTS: list[str] = list(preset("refusal")["positive_prompts"])
REFUSAL_NEGATIVE_PROMPTS: list[str] = list(preset("refusal")["negative_prompts"])
DEFAULT_REFUSAL_PHRASES: list[str] = phrases("refusal")  # 日本語＋英語(2026-10-13)


def count_refusal_phrases(text: str, phrases: list[str] | None = None) -> dict[str, object]:
    """テキスト中の拒否定型句の出現回数を数える(部分一致、単純な文字列カウント)。

    カウントのロジック自体はSycophancy Reducerと共通(count_sycophantic_phrasesを語彙だけ
    差し替えて使い回す)。戻り値の形式も同じ: {"total": 合計, "by_phrase": {句: 回数}}
    """
    return count_sycophantic_phrases(text, phrases if phrases is not None else DEFAULT_REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# 方向除去の数式(torch非依存版、テスト用)。実際の適用はgraph_runner.execute()が
# torchテンソルに対して同じ計算を行う。
# ---------------------------------------------------------------------------

def normalize(vector: list[float]) -> list[float]:
    """単位ベクトルに正規化する。ゼロベクトルは正規化できないためValueError。"""
    norm = sum(v * v for v in vector) ** 0.5
    if norm < 1e-12:
        raise ValueError("zero vector cannot be normalized")
    return [v / norm for v in vector]


def project_out(hidden_vector: list[float], direction: list[float]) -> list[float]:
    """隠れ状態ベクトルから、指定方向の成分を取り除く: h - (h・r̂) r̂

    結果は方向ベクトルと直交する(内積が0になる)。方向ベクトルの大きさ(ノルム)には
    依存しない(正規化してから使うため、scaleの概念が無い)。
    """
    if len(hidden_vector) != len(direction):
        raise ValueError("hidden_vector and direction must have the same dimension")
    unit = normalize(direction)
    coeff = sum(h * u for h, u in zip(hidden_vector, unit))
    return [h - coeff * u for h, u in zip(hidden_vector, unit)]
