"""
battle_graph.py

Multi-Model Battle View: 比較対象の各モデルについて、実行/観察タブで
ユーザーが編集した実行グラフとは独立に、「そのモデルのオリジナル構造通りの
デフォルト実行グラフ(embedding -> layer_block * num_layers -> lm_head)」を
モデルのconfig(num_hidden_layers)だけから機械的に組み立てる。

Battle Viewでは異なるアーキテクチャ・異なる層数のモデル同士を比較することも
想定しているため(2026-09-10、漣さんとの確認事項)、ユーザーがモデルごとに
いちいちキャンバスでグラフを組む必要をなくし、常に「そのモデル本来の構造」で
比較する。フロントエンド側のtoExecutionGraph.ts(GUIキャンバスの初期グラフ生成)と
同じ考え方をPython側にも用意したもの。

torch非依存の純粋なロジックのみを含む(GraphNodeの組み立てとエッジの生成のみ)。
"""

from __future__ import annotations

from .graph_validation import GraphNode


def build_default_execution_graph(
    num_layers: int,
) -> tuple[list[GraphNode], list[tuple[str, str]]]:
    """モデルの層数だけから、オリジナル構造通りの直列実行グラフを組み立てる。

    Layerは折りたたんだ状態(layer_blockロール)のまま扱う。Battle Viewの
    比較粒度(モデル単位での生成結果・活性化値統計)にはブロック単位の展開は
    不要なため、ノード数を抑える目的で折りたたみのままにしている
    (layer_blockはgraph_runner.py内部でAttention/MLPを自動実行し、
    capture_internals=True時は両方の内部状態が記録される)。

    num_layersが0の場合、embeddingとlm_headの2ノードだけの空グラフになる
    (実行自体は可能。実質的に埋め込み直後にlm_headを適用するだけの経路)。
    """
    if num_layers < 0:
        raise ValueError(f"num_layers must be >= 0, got {num_layers}")

    nodes: list[GraphNode] = [
        GraphNode(id="model.embed_tokens", role="embedding", layer_index=None)
    ]
    for i in range(num_layers):
        nodes.append(GraphNode(id=f"model.layers.{i}", role="layer_block", layer_index=i))
    nodes.append(GraphNode(id="lm_head", role="lm_head", layer_index=None))

    edges: list[tuple[str, str]] = [
        (nodes[i].id, nodes[i + 1].id) for i in range(len(nodes) - 1)
    ]
    return nodes, edges
