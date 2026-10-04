"""
test_neuron_search_utils.py

neuron_search_utils.py の純粋ロジック(層番号検証・ランキング)を検証する。
torch非依存なので、この開発環境でも実行できる。

実行方法: python3 python/tests/test_neuron_search_utils.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.neuron_search_utils import rank_scored_neurons, resolve_target_layers  # noqa: E402


def test_resolve_target_layers_none_returns_all():
    assert resolve_target_layers(5, None) == [0, 1, 2, 3, 4]
    print("PASS: test_resolve_target_layers_none_returns_all")


def test_resolve_target_layers_sorts_and_dedupes():
    assert resolve_target_layers(10, [3, 1, 3, 7]) == [1, 3, 7]
    print("PASS: test_resolve_target_layers_sorts_and_dedupes")


def test_resolve_target_layers_rejects_out_of_range():
    try:
        resolve_target_layers(5, [2, 10, -1])
    except ValueError as e:
        assert "10" in str(e) and "-1" in str(e)
        print("PASS: test_resolve_target_layers_rejects_out_of_range")
        return
    raise AssertionError("expected ValueError for out-of-range layer indices")


def test_rank_scored_neurons_sorts_descending():
    scored = [
        {"layer_index": 0, "kind": "mlp", "neuron_index": 0, "score": 1.0},
        {"layer_index": 0, "kind": "mlp", "neuron_index": 1, "score": 5.0},
        {"layer_index": 1, "kind": "attention_head", "neuron_index": 0, "score": 3.0},
    ]
    ranked = rank_scored_neurons(scored, top_k=10)
    assert [e["score"] for e in ranked] == [5.0, 3.0, 1.0]
    print("PASS: test_rank_scored_neurons_sorts_descending")


def test_rank_scored_neurons_respects_top_k():
    scored = [{"score": float(i)} for i in range(100)]
    ranked = rank_scored_neurons(scored, top_k=3)
    assert len(ranked) == 3
    assert [e["score"] for e in ranked] == [99.0, 98.0, 97.0]
    print("PASS: test_rank_scored_neurons_respects_top_k")


def test_rank_scored_neurons_top_k_larger_than_list():
    scored = [{"score": 1.0}, {"score": 2.0}]
    ranked = rank_scored_neurons(scored, top_k=100)
    assert len(ranked) == 2
    print("PASS: test_rank_scored_neurons_top_k_larger_than_list")


if __name__ == "__main__":
    test_resolve_target_layers_none_returns_all()
    test_resolve_target_layers_sorts_and_dedupes()
    test_resolve_target_layers_rejects_out_of_range()
    test_rank_scored_neurons_sorts_descending()
    test_rank_scored_neurons_respects_top_k()
    test_rank_scored_neurons_top_k_larger_than_list()
    print("\nAll neuron_search_utils logic-verification tests passed.")
