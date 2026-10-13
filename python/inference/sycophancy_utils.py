"""
sycophancy_utils.py

Phase3: Sycophancy Reducer(迎合的な応答傾向の抑制)の中核ロジック(torch非依存)。

2026-09-15、漣さんとの確認事項:
- 実装方法: 独立した新機能ではなく、既存のActivation Steering Studioの拡張として実現する。
  「迎合的な応答例(positive)」と「中立的・事実重視の応答例(negative)」の対照プロンプト対から
  計算した差分ベクトルを、負のscale(逆方向)で適用することで「迎合傾向を減らす」方向へ誘導する
  (ベクトルの向き自体はActivation Steering Studioと全く同じ仕組みを使い回す)
- 対照プロンプトの用意のしかた: ユーザーが毎回入力するのではなく、固定のプリセット文例を
  ここに内蔵し、GUI側の「Sycophancy Reducerプリセット」ボタンで対照プロンプト欄に流し込む
- 迎合的な応答傾向の検出・評価: 生成テキストの目視比較に加えて、機械的な指標も併用する。
  「素晴らしいですね」「おっしゃる通りです」のような定型的な迎合語彙の出現回数を数える、
  という単純なヒューリスティックで定量化する(厳密な意味解析ではなく、あくまで簡易指標)
"""

from __future__ import annotations

from .presets import phrases, preset

# 文例・定型語彙の実体は python/presets/steering_presets.json("sycophancy")にある(2026-09-22、
# PythonとReactで二重定義していたのを一本化)。
# SYCOPHANCY_POSITIVE_PROMPTS: 迎合的な応答の例文(ユーザーの意見に理由なく強く同意・称賛する言い回し)
# SYCOPHANCY_NEGATIVE_PROMPTS: 中立的・事実重視の応答の例文(根拠を示し、懸念点や別の見方にも触れる)
# DEFAULT_SYCOPHANTIC_PHRASES: 迎合的な応答によく現れる定型語彙(部分一致でカウントする簡易ヒューリスティック)
SYCOPHANCY_POSITIVE_PROMPTS: list[str] = list(preset("sycophancy")["positive_prompts"])
SYCOPHANCY_NEGATIVE_PROMPTS: list[str] = list(preset("sycophancy")["negative_prompts"])
DEFAULT_SYCOPHANTIC_PHRASES: list[str] = phrases("sycophancy")  # 日本語＋英語(2026-10-13)


def count_sycophantic_phrases(
    text: str, phrases: list[str] | None = None
) -> dict[str, object]:
    """テキスト中に迎合的な定型語彙が何回出現するかを数える(部分一致、単純な文字列カウント)。

    phrases: Noneの場合はDEFAULT_SYCOPHANTIC_PHRASESを使う。カスタムのリストを渡すことも可能。
    戻り値: {"total": 合計出現回数, "by_phrase": {語彙: 出現回数, ...}(0回のものは含めない)}
    空文字列・出現なしの場合はtotal=0、by_phrase={}を返す。
    """
    target_phrases = phrases if phrases is not None else DEFAULT_SYCOPHANTIC_PHRASES
    by_phrase: dict[str, int] = {}
    for phrase in target_phrases:
        if not phrase:
            continue
        count = text.count(phrase)
        if count > 0:
            by_phrase[phrase] = count
    total = sum(by_phrase.values())
    return {"total": total, "by_phrase": by_phrase}
