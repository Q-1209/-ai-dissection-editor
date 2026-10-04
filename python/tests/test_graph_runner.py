"""
test_graph_runner.py

graph_runner.py のうち、PyTorchに依存しない部分(トポロジカルソート、
グラフ検証ロジック)は、この環境でも直接実行して検証できる。

実行方法: python3 python/tests/test_graph_runner.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.graph_validation import (
    GraphNode,
    GraphValidationError,
    build_input_map,
    build_input_map_multi,
    topological_sort,
)


def test_simple_linear_order():
    """A -> B -> C という単純な直列グラフは、その順序通りにソートされる。"""
    nodes = [GraphNode("A", "norm", 0), GraphNode("B", "norm", 0), GraphNode("C", "norm", 0)]
    edges = [("A", "B"), ("B", "C")]
    order = topological_sort(nodes, edges)
    assert order == ["A", "B", "C"], f"unexpected order: {order}"
    print("PASS: test_simple_linear_order")


def test_user_reordering_is_respected():
    """Phase1のコア価値: ユーザーがオリジナルと違う順序でエッジを組んだ場合、
    その通りの順序が導出される(例: Layer1のAttentionをLayer0の前に持ってくる)。
    """
    nodes = [
        GraphNode("layer0.attn", "attention_block", 0),
        GraphNode("layer1.attn", "attention_block", 1),
    ]
    # ユーザーがlayer1を先に、layer0を後に繋いだ場合
    edges = [("layer1.attn", "layer0.attn")]
    order = topological_sort(nodes, edges)
    assert order == ["layer1.attn", "layer0.attn"], f"unexpected order: {order}"
    print("PASS: test_user_reordering_is_respected")


def test_skip_connection_no_error():
    """スキップ接続(Layer0 -> Layer2、Layer1を経由しない)がエラーにならず実行できる。"""
    nodes = [
        GraphNode("layer0", "attention_block", 0),
        GraphNode("layer1", "attention_block", 1),
        GraphNode("layer2", "attention_block", 2),
    ]
    # layer1をスキップしてlayer0 -> layer2を直結
    edges = [("layer0", "layer2")]
    order = topological_sort(nodes, edges)
    # layer1は孤立ノードとして順序のどこかに入る(接続はないので実行結果には影響しない想定)
    assert set(order) == {"layer0", "layer1", "layer2"}
    assert order.index("layer0") < order.index("layer2")
    print("PASS: test_skip_connection_no_error")


def test_cycle_detection():
    """循環接続(A->B->A)はGraphValidationErrorとして検出される。"""
    nodes = [GraphNode("A", "norm", 0), GraphNode("B", "norm", 0)]
    edges = [("A", "B"), ("B", "A")]
    try:
        topological_sort(nodes, edges)
        assert False, "expected GraphValidationError for cyclic graph"
    except GraphValidationError:
        print("PASS: test_cycle_detection")


def test_unknown_node_reference_detection():
    """存在しないノードIDへのエッジはGraphValidationErrorになる。"""
    nodes = [GraphNode("A", "norm", 0)]
    edges = [("A", "NONEXISTENT")]
    try:
        topological_sort(nodes, edges)
        assert False, "expected GraphValidationError for unknown node reference"
    except GraphValidationError:
        print("PASS: test_unknown_node_reference_detection")


def test_deterministic_order_for_parallel_branches():
    """複数の独立したブランチがある場合も、実行順序は決定的(再現可能)である。"""
    nodes = [GraphNode(nid, "norm", 0) for nid in ["Z", "Y", "X"]]
    edges: list[tuple[str, str]] = []  # 全ノードが独立(依存関係なし)
    order1 = topological_sort(nodes, edges)
    order2 = topological_sort(nodes, edges)
    assert order1 == order2, "topological_sort should be deterministic"
    assert order1 == ["X", "Y", "Z"], f"expected alphabetical order for independent nodes, got {order1}"
    print("PASS: test_deterministic_order_for_parallel_branches")


def test_disconnected_edge_creates_unreachable_node():
    """回帰テスト: エッジを削除してグラフを分断した場合、切り離された側のノードは
    「孤立(到達不可能)」として検出されなければならない。

    このテストが検証するのは、実際に発生したバグ: topological_sort() だけでは
    ノードの並び順しか分からず、エッジが実際に繋がっているかは見ていないため、
    分断されたグラフでも「実行成功」してしまっていた。build_input_map() で
    この問題が検出できることを確認する。
    """
    nodes = [
        GraphNode("embedding", "embedding", None),
        GraphNode("layer0.attn", "attention_block", 0),
        GraphNode("layer0.mlp", "mlp_block", 0),
        GraphNode("layer1.attn", "attention_block", 1),
    ]
    # layer0.attn -> layer0.mlp の接続を「削除した」状態を模す
    edges = [
        ("embedding", "layer0.attn"),
        # ("layer0.attn", "layer0.mlp"),  <- ここが削除された接続
        ("layer0.mlp", "layer1.attn"),
    ]
    try:
        build_input_map(nodes, edges, entry_node_id="embedding")
        assert False, "expected GraphValidationError for disconnected node"
    except GraphValidationError as e:
        assert "layer0.mlp" in str(e), f"error should mention the unreachable node: {e}"
        print("PASS: test_disconnected_edge_creates_unreachable_node")


def test_valid_linear_graph_builds_correct_input_map():
    """正常な直列グラフでは、各ノードの入力元が正しく1対1で解決される。"""
    nodes = [
        GraphNode("embedding", "embedding", None),
        GraphNode("layer0.attn", "attention_block", 0),
        GraphNode("layer0.mlp", "mlp_block", 0),
    ]
    edges = [("embedding", "layer0.attn"), ("layer0.attn", "layer0.mlp")]
    input_map = build_input_map(nodes, edges, entry_node_id="embedding")
    assert input_map == {"layer0.attn": "embedding", "layer0.mlp": "layer0.attn"}
    print("PASS: test_valid_linear_graph_builds_correct_input_map")


def test_reordered_graph_builds_correct_input_map():
    """Phase1のコア機能: ユーザーが層を入れ替えた場合も、入力元が正しく解決される。"""
    nodes = [
        GraphNode("embedding", "embedding", None),
        GraphNode("layer0.attn", "attention_block", 0),
        GraphNode("layer1.attn", "attention_block", 1),
    ]
    # layer1.attnをlayer0.attnより先に実行する繋ぎ替え
    edges = [("embedding", "layer1.attn"), ("layer1.attn", "layer0.attn")]
    input_map = build_input_map(nodes, edges, entry_node_id="embedding")
    assert input_map == {"layer1.attn": "embedding", "layer0.attn": "layer1.attn"}
    print("PASS: test_reordered_graph_builds_correct_input_map")


def test_merge_multiple_inputs_rejected():
    """1つのノードに複数の入力エッジがある場合(合流)はPhase1スコープ外としてエラーになる。"""
    nodes = [
        GraphNode("embedding", "embedding", None),
        GraphNode("layer0.attn", "attention_block", 0),
        GraphNode("layer1.attn", "attention_block", 1),
        GraphNode("layer2.attn", "attention_block", 2),
    ]
    # layer0.attnとlayer1.attnの両方がlayer2.attnに接続(合流)
    edges = [
        ("embedding", "layer0.attn"),
        ("embedding", "layer1.attn"),
        ("layer0.attn", "layer2.attn"),
        ("layer1.attn", "layer2.attn"),
    ]
    try:
        build_input_map(nodes, edges, entry_node_id="embedding")
        assert False, "expected GraphValidationError for merged inputs"
    except GraphValidationError as e:
        assert "layer2.attn" in str(e)
        print("PASS: test_merge_multiple_inputs_rejected")


def test_multi_allows_merge_with_multiple_sources():
    """build_input_map_multi()は合流(複数入力)を許可し、全ての入力元をリストで返す。"""
    nodes = [
        GraphNode("embedding", "embedding", None),
        GraphNode("layer0.attn", "attention_block", 0),
        GraphNode("layer1.attn", "attention_block", 1),
        GraphNode("layer2.attn", "attention_block", 2),
    ]
    edges = [
        ("embedding", "layer0.attn"),
        ("embedding", "layer1.attn"),
        ("layer0.attn", "layer2.attn"),
        ("layer1.attn", "layer2.attn"),
    ]
    input_map = build_input_map_multi(nodes, edges, entry_node_id="embedding")
    assert set(input_map["layer2.attn"]) == {"layer0.attn", "layer1.attn"}
    assert input_map["layer0.attn"] == ["embedding"]
    assert input_map["layer1.attn"] == ["embedding"]
    print("PASS: test_multi_allows_merge_with_multiple_sources")


def test_multi_still_detects_unreachable_nodes():
    """build_input_map_multi()でも、孤立ノードの検出は引き続き機能する
    (合流を許可しても、到達不可能なノードの検出という安全性は失われない)。
    """
    nodes = [
        GraphNode("embedding", "embedding", None),
        GraphNode("layer0.attn", "attention_block", 0),
        GraphNode("layer0.mlp", "mlp_block", 0),
    ]
    edges = [("embedding", "layer0.attn")]  # layer0.mlpへの接続が欠落
    try:
        build_input_map_multi(nodes, edges, entry_node_id="embedding")
        assert False, "expected GraphValidationError for unreachable node"
    except GraphValidationError as e:
        assert "layer0.mlp" in str(e)
        print("PASS: test_multi_still_detects_unreachable_nodes")


def test_multi_single_input_returns_single_element_list():
    """通常の(合流でない)単一入力ノードも、build_input_map_multi()では
    要素数1のリストとして一貫した形式で返る。
    """
    nodes = [
        GraphNode("embedding", "embedding", None),
        GraphNode("layer0.attn", "attention_block", 0),
    ]
    edges = [("embedding", "layer0.attn")]
    input_map = build_input_map_multi(nodes, edges, entry_node_id="embedding")
    assert input_map["layer0.attn"] == ["embedding"]
    print("PASS: test_multi_single_input_returns_single_element_list")


if __name__ == "__main__":
    test_simple_linear_order()
    test_user_reordering_is_respected()
    test_skip_connection_no_error()
    test_cycle_detection()
    test_unknown_node_reference_detection()
    test_deterministic_order_for_parallel_branches()
    test_disconnected_edge_creates_unreachable_node()
    test_valid_linear_graph_builds_correct_input_map()
    test_reordered_graph_builds_correct_input_map()
    test_merge_multiple_inputs_rejected()
    test_multi_allows_merge_with_multiple_sources()
    test_multi_still_detects_unreachable_nodes()
    test_multi_single_input_returns_single_element_list()
    print("\nAll graph_runner logic tests passed.")
