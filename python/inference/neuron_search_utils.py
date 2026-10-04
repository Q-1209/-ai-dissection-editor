"""
neuron_search_utils.py

Neuron Search: torch非依存の純粋ロジック(層番号の検証・スコアのランキング)。
実際のテンソルからの値抽出・集約はrun_neuron_search.py(torch依存)側で行い、
このモジュールはPythonのlist/dict操作のみを担当する(この開発環境でも
torch無しでテストできるようにするため、他の*_numpy.pyテストと同じ狙い)。
"""

from __future__ import annotations


def resolve_target_layers(num_layers: int, target_layers: list[int] | None) -> list[int]:
    """target_layersが指定されていれば範囲内かを検証し、Noneなら全層を対象にする。

    重複や順序はここで正規化する(呼び出し側で毎回集合化・ソートしなくて済むように)。
    """
    if target_layers is None:
        return list(range(num_layers))
    invalid = [i for i in target_layers if i < 0 or i >= num_layers]
    if invalid:
        raise ValueError(f"存在しない層番号が指定されました: {invalid} (このモデルは{num_layers}層)")
    return sorted(set(target_layers))


def rank_scored_neurons(scored: list[dict], top_k: int) -> list[dict]:
    """score(降順)で並べ替え、上位top_k件を返す。

    scoredの各要素は最低限 {"score": float, ...} を持つ辞書を想定(layer_index/kind/
    neuron_indexなど他のフィールドはそのまま保持される)。
    """
    return sorted(scored, key=lambda e: e["score"], reverse=True)[:top_k]
