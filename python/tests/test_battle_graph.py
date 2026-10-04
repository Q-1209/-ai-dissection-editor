"""
test_battle_graph.py

battle_graph.py の build_default_execution_graph() を検証する。
このロジックはtorch非依存の純粋なPython(list/dataclass操作のみ)なので、
torch環境がなくても実行できる(他のtorch非依存テストと同じ位置づけ)。

実行方法: python3 python/tests/test_battle_graph.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.battle_graph import build_default_execution_graph  # noqa: E402
from inference.graph_validation import (  # noqa: E402
    GraphValidationError,
    build_input_map_multi,
    topological_sort,
)


def test_basic_structure_has_embedding_layers_lm_head():
    """embedding -> layer_block * N -> lm_head の順に並んでいることを確認する。"""
    nodes, edges = build_default_execution_graph(num_layers=3)
    ids = [n.id for n in nodes]
    assert ids == [
        "model.embed_tokens",
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
        "lm_head",
    ], f"unexpected node order: {ids}"
    print("PASS: test_basic_structure_has_embedding_layers_lm_head")


def test_roles_are_correct():
    """各ノードのroleが期待通りであることを確認する(layer_blockのまま展開しない)。"""
    nodes, _ = build_default_execution_graph(num_layers=2)
    roles = {n.id: n.role for n in nodes}
    assert roles["model.embed_tokens"] == "embedding"
    assert roles["model.layers.0"] == "layer_block"
    assert roles["model.layers.1"] == "layer_block"
    assert roles["lm_head"] == "lm_head"
    print("PASS: test_roles_are_correct")


def test_layer_index_is_set_for_layer_blocks_only():
    """layer_indexはlayer_blockのみに設定され、embedding/lm_headはNoneのままである。"""
    nodes, _ = build_default_execution_graph(num_layers=2)
    by_id = {n.id: n for n in nodes}
    assert by_id["model.embed_tokens"].layer_index is None
    assert by_id["lm_head"].layer_index is None
    assert by_id["model.layers.0"].layer_index == 0
    assert by_id["model.layers.1"].layer_index == 1
    print("PASS: test_layer_index_is_set_for_layer_blocks_only")


def test_edges_form_a_single_serial_chain():
    """エッジが全ノードを一直線に繋ぐ、分岐も合流もないチェーンになっていることを確認する。"""
    nodes, edges = build_default_execution_graph(num_layers=4)
    assert len(edges) == len(nodes) - 1
    for i in range(len(nodes) - 1):
        assert edges[i] == (nodes[i].id, nodes[i + 1].id)
    print("PASS: test_edges_form_a_single_serial_chain")


def test_zero_layers_still_connects_embedding_to_lm_head():
    """層数0でも、embeddingからlm_headへ直接繋がる(実行可能な)グラフになる。"""
    nodes, edges = build_default_execution_graph(num_layers=0)
    assert [n.id for n in nodes] == ["model.embed_tokens", "lm_head"]
    assert edges == [("model.embed_tokens", "lm_head")]
    print("PASS: test_zero_layers_still_connects_embedding_to_lm_head")


def test_negative_layers_raises_value_error():
    """負の層数は明確なエラーとして拒否する(モデルconfigの異常値混入を検出する)。"""
    try:
        build_default_execution_graph(num_layers=-1)
    except ValueError:
        print("PASS: test_negative_layers_raises_value_error")
        return
    raise AssertionError("expected ValueError for num_layers=-1")


def test_generated_graph_passes_existing_validation_logic():
    """生成したグラフが、既存のgraph_validation.py(トポロジカルソート・
    孤立ノード検出)を問題なく通過することを確認する。異なるアーキテクチャの
    モデル同士(層数が大きく異なる)を想定し、大きめの層数でも確認する。
    """
    for num_layers in (0, 1, 24, 32):
        nodes, edges = build_default_execution_graph(num_layers=num_layers)
        order = topological_sort(nodes, edges)
        assert len(order) == len(nodes)
        input_map = build_input_map_multi(nodes, edges, entry_node_id="model.embed_tokens")
        # embedding以外の全ノードに入力元が1つずつ解決されているはず(合流なし)
        for n in nodes:
            if n.id == "model.embed_tokens":
                continue
            assert n.id in input_map and len(input_map[n.id]) == 1
    print("PASS: test_generated_graph_passes_existing_validation_logic")


def test_different_layer_counts_do_not_interfere():
    """2つの異なる層数から生成したグラフが、互いに独立していることを確認する
    (Battle Viewでは異なるアーキテクチャのモデルを同時に扱うため、
    グラフ生成がモデルごとに独立していることが重要)。
    """
    nodes_a, edges_a = build_default_execution_graph(num_layers=24)  # 例: Qwen2.5-0.5B
    nodes_b, edges_b = build_default_execution_graph(num_layers=32)  # 例: 別アーキテクチャの大きめモデル
    assert len(nodes_a) == 26  # embed + 24 layers + lm_head
    assert len(nodes_b) == 34  # embed + 32 layers + lm_head
    assert nodes_a is not nodes_b
    assert edges_a is not edges_b
    print("PASS: test_different_layer_counts_do_not_interfere")


if __name__ == "__main__":
    test_basic_structure_has_embedding_layers_lm_head()
    test_roles_are_correct()
    test_layer_index_is_set_for_layer_blocks_only()
    test_edges_form_a_single_serial_chain()
    test_zero_layers_still_connects_embedding_to_lm_head()
    test_negative_layers_raises_value_error()
    test_generated_graph_passes_existing_validation_logic()
    test_different_layer_counts_do_not_interfere()
    print("\nAll battle_graph logic-verification tests passed.")
