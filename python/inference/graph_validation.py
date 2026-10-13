"""
graph_validation.py

ノード+エッジのグラフ定義に対する検証・トポロジカルソートロジック。
このモジュールはPyTorchに依存しない(純粋なPythonロジックのみ)ため、
軽量な環境でも単体テストが可能。実際のテンソル計算を伴う実行は
graph_runner.py (torch依存)が担当する。
"""

from __future__ import annotations

from dataclasses import dataclass
from .i18n import tr


class GraphValidationError(Exception):
    """ユーザーが組んだグラフが実行不可能な場合(shape不一致・サイクル等)に送出する。"""


@dataclass
class GraphNode:
    id: str
    role: str  # "embedding" | "attention_block" | "mlp_block" | "norm" | "lm_head" 等
    layer_index: int | None
    # このノードが複数の入力(合流)を受け取る場合、どう1つのテンソルに組み合わせるか。
    # "add": 要素ごとの加算(残差接続と同じ発想)。"average": 要素ごとの平均。
    # 入力が1つだけの場合は無視される(合成の必要がないため)。
    merge_mode: str = "add"
    # Layer Ablation Wizard(2026-09-12追加)用: Trueの場合、このノードの実際の計算
    # (Attention/MLP等)を一切行わず、出力を単純にゼロテンソルにする(残差接続も含めて
    # このノードの寄与を完全に断つ、という漣さんとの確認事項に基づく定義)。
    # デフォルトFalseなので、既存の全エントリポイント(execute_graph.py等)の挙動には
    # 一切影響しない。
    ablated: bool = False


def topological_sort(nodes: list[GraphNode], edges: list[tuple[str, str]]) -> list[str]:
    """エッジ定義からノードの実行順序を導出する(Kahnのアルゴリズム)。

    サイクルがある場合はGraphValidationErrorを送出する
    (「ノードAの出力がノードBの入力で、ノードBの出力がノードAの入力」のような
    循環接続は、この設計では実行不可能なため)。

    独立したノード(依存関係のないノード)が複数ある場合は、id順にソートして
    取り出すことで実行順序を決定的(再現可能)にしている。

    注意: この関数はノードの「実行可能な順序」を返すのみで、各ノードが実際に
    何のノードから入力を受け取るか(=グラフ上のエッジで実際に繋がっているか)は
    保証しない。孤立ノード(どこにも接続されていないノード)も順序には含まれる。
    実際の推論実行では、これとは別に build_input_map() で各ノードの入力元を
    明示的に解決し、孤立ノードは実行前にエラーとして検出する必要がある。
    """
    node_ids = {n.id for n in nodes}
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for source, target in edges:
        if source not in node_ids or target not in node_ids:
            raise GraphValidationError(
                f"Edge references unknown node: {source} -> {target}"
            )
        adjacency[source].append(target)
        in_degree[target] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    order: list[str] = []

    while queue:
        queue.sort()
        current = queue.pop(0)
        order.append(current)
        for neighbor in adjacency[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(node_ids):
        remaining = node_ids - set(order)
        raise GraphValidationError(
            f"Graph contains a cycle involving nodes: {remaining}"
        )

    return order


def build_input_map_multi(
    nodes: list[GraphNode], edges: list[tuple[str, str]], entry_node_id: str
) -> dict[str, list[str]]:
    """各ノードが「どのノードから入力を受け取るか」を解決する(複数入力=合流に対応)。

    戻り値: {target_node_id: [source_node_id, ...]} の辞書。
    入力が1つだけのノードもリスト(要素数1)として返す。

    重要な検証:
    - entry_node_id(通常はembeddingノード)以外のノードで、入力エッジが
      1つもないノードは「グラフから孤立している(到達不可能)」としてエラーにする。
      これがないと、エッジを削除してグラフを分断しても実行時にそれが無視され、
      分断前と同じ結果が出てしまうバグになる。
    - 複数の入力エッジを持つノード(合流)は許可する。実際にどう合成するか
      (加算/平均等)はgraph_runner.py側のmerge_modeが決める。このモジュールは
      「どのノードから入力を受け取るか」の解決のみを担当する。
    """
    node_ids = {n.id for n in nodes}
    incoming: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for source, target in edges:
        incoming[target].append(source)

    input_map: dict[str, list[str]] = {}
    unreachable: list[str] = []

    for nid in node_ids:
        if nid == entry_node_id:
            continue  # entry nodeは外部入力(トークンID)を使うため、グラフ上の入力元は不要
        sources = incoming[nid]
        if len(sources) == 0:
            unreachable.append(nid)
        else:
            input_map[nid] = sources

    if unreachable:
        raise GraphValidationError(
            tr(f"以下のノードはどこからも接続されておらず、実行できません(グラフから孤立しています): "
            f"{sorted(unreachable)}。ノード同士を接続してください。", f"These nodes are not connected to anything and cannot run (isolated from the graph): " f"{sorted(unreachable)}. Connect them to the graph.")
        )

    return input_map


def build_input_map(
    nodes: list[GraphNode], edges: list[tuple[str, str]], entry_node_id: str
) -> dict[str, str]:
    """各ノードが「どのノードから入力を受け取るか」を解決する(単一入力のみ許可)。

    build_input_map_multi() の単一入力専用ラッパー。合流(複数入力)が
    許可されていない文脈(例: 単純な逐次実行のみサポートする経路)で使う。
    合流を検出した場合はGraphValidationErrorを送出する。

    戻り値: {target_node_id: source_node_id} の辞書。
    """
    multi_map = build_input_map_multi(nodes, edges, entry_node_id)

    merged = [nid for nid, sources in multi_map.items() if len(sources) > 1]
    if merged:
        raise GraphValidationError(
            tr(f"以下のノードは複数の入力を受け取っています: {sorted(merged)}。"
            f"このグラフでは単純な直列接続のみサポートしており、複数ノードの出力を"
            f"1つのノードに合流させることはできません。", f"These nodes receive more than one input: {sorted(merged)}. " f"This graph only supports simple chains; outputs of several nodes " f"cannot be merged into one node here.")
        )

    return {nid: sources[0] for nid, sources in multi_map.items()}


def validate_shape_compatibility(
    source_last_dim: int, target_expected_dim: int, source_id: str, target_id: str
) -> None:
    """接続元の出力次元数が、接続先が期待する入力次元数と一致するか検証する。

    Phase1合意事項: 不一致の場合は自動変換(アダプター挿入)をせず、エラーで拒否する。
    """
    if source_last_dim != target_expected_dim:
        raise GraphValidationError(
            tr(f"Shape mismatch: '{source_id}' outputs dim {source_last_dim}, "
            f"but '{target_id}' expects dim {target_expected_dim}. "
            f"Phase1では自動アダプター変換をサポートしていません。"
            f"同じhidden_sizeを持つノード同士のみ接続できます。", f"Shape mismatch: '{source_id}' outputs dim {source_last_dim}, " f"but '{target_id}' expects dim {target_expected_dim}. " f"No adapter is inserted automatically; " f"only nodes with the same hidden_size can be connected.")
        )
