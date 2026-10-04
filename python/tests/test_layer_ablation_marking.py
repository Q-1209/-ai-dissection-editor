"""
test_layer_ablation_marking.py

run_layer_ablation.py の mark_ablated_nodes() (純粋なPythonロジック、torch非依存)を検証する。
GraphNode.ablated フィールドの読み書きだけなので、この開発環境でも実行できる。

実行方法: python3 python/tests/test_layer_ablation_marking.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.graph_validation import GraphNode, GraphValidationError  # noqa: E402
from run_layer_ablation import mark_ablated_nodes  # noqa: E402


def _make_nodes():
    return [
        GraphNode(id="embedding", role="embedding", layer_index=None),
        GraphNode(id="model.layers.0", role="layer_block", layer_index=0),
        GraphNode(id="model.layers.1", role="layer_block", layer_index=1),
        GraphNode(id="lm_head", role="lm_head", layer_index=None),
    ]


def test_mark_ablated_nodes_sets_only_specified():
    nodes = _make_nodes()
    mark_ablated_nodes(nodes, ["model.layers.1"])
    ablated_ids = {n.id for n in nodes if n.ablated}
    assert ablated_ids == {"model.layers.1"}
    print("PASS: test_mark_ablated_nodes_sets_only_specified")


def test_mark_ablated_nodes_resets_previous_flags():
    nodes = _make_nodes()
    mark_ablated_nodes(nodes, ["model.layers.0", "model.layers.1"])
    mark_ablated_nodes(nodes, ["model.layers.0"])  # 再設定で1つ減らす
    ablated_ids = {n.id for n in nodes if n.ablated}
    assert ablated_ids == {"model.layers.0"}
    print("PASS: test_mark_ablated_nodes_resets_previous_flags")


def test_mark_ablated_nodes_empty_clears_all():
    nodes = _make_nodes()
    mark_ablated_nodes(nodes, ["model.layers.0"])
    mark_ablated_nodes(nodes, [])  # baseline計算用: 全てFalseに戻る
    assert all(not n.ablated for n in nodes)
    print("PASS: test_mark_ablated_nodes_empty_clears_all")


def test_mark_ablated_nodes_rejects_unknown_id():
    nodes = _make_nodes()
    try:
        mark_ablated_nodes(nodes, ["model.layers.99"])
    except GraphValidationError as e:
        assert "model.layers.99" in str(e)
        print("PASS: test_mark_ablated_nodes_rejects_unknown_id")
        return
    raise AssertionError("expected GraphValidationError for unknown node id")


if __name__ == "__main__":
    test_mark_ablated_nodes_sets_only_specified()
    test_mark_ablated_nodes_resets_previous_flags()
    test_mark_ablated_nodes_empty_clears_all()
    test_mark_ablated_nodes_rejects_unknown_id()
    print("\nAll layer-ablation marking logic-verification tests passed.")
