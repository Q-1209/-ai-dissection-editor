#!/usr/bin/env python3
"""
run_neuron_search.py

Phase2: Neuron Search — 特定ニューロンの探索。

2026-09-12、漣さんとの確認事項:
- 「ニューロン」の定義: MLPの中間層次元(gate/up projection後、down_proj前の1次元)、
  Attentionヘッド単位の両方に対応する
- 探索の方向性: (a) 入力に対して強く反応するニューロンをランキングする、(b) 指定した
  ニューロンが複数の入力に対してどう反応するかを調べる、の両方に対応する
- 既存の観察タブ「内部状態」(層単位の活性化値統計)を、ニューロン単位まで細分化した
  もの、という位置づけ

「ニューロンの活性化値」の定義:
- MLPニューロン: SwiGLU MLPの中間活性化値 silu(gate(x)) * up(x) の該当次元(down_proj前)
- Attentionヘッド: そのヘッドの出力ベクトル(head_dim次元、o_proj前)のL2ノルム。
  ヘッド単位で「何か強い信号を出しているか」の代理指標として使う
  (attention_weights自体は「どこに注目しているか」であり、「どれだけ強く反応しているか」
  とは別の情報のため)

いずれも、入力トークン列の各位置ごとに値を持つ([seq_len]次元)。「反応の強さ」は
位置方向に集約(--aggregation: "max_abs"=系列内の最大絶対値、"mean_abs"=平均絶対値)
した1つのスカラー値として扱う。

入力フォーマット(stdin、mode="rank"の場合):
{
  "model_dir": "...", "config": {...}, "nodes": [...], "edges": [...],
  // nodes/edgesは実行/観察タブで組んだ現在のグラフ構造をそのまま使う
  // (Battle Viewとは異なり、モデル自身のデフォルトグラフではなくGUI側のグラフを使う。
  //  既存の内部状態可視化(inspect_internals.py)と同じ考え方)
  "mode": "rank",
  "input_text": "こんにちは",       // または input_ids
  "neuron_kind": "mlp" | "attention_head" | "both",  // 省略時"both"
  "target_layers": [0, 1, 2] | null,  // 省略時(null)は全層が対象
  "top_k": 20,                        // 省略時20
  "aggregation": "max_abs" | "mean_abs"  // 省略時"max_abs"
}

出力フォーマット(mode="rank"、成功時):
{
  "success": true, "mode": "rank",
  "input_token_ids": [...], "input_tokens_text": [...] | null,
  "ranked_neurons": [
    {"layer_index": 5, "kind": "mlp", "neuron_index": 812, "score": 12.4},
    {"layer_index": 3, "kind": "attention_head", "neuron_index": 2, "score": 8.1},
    ...
  ]
}

入力フォーマット(stdin、mode="probe"の場合):
{
  "model_dir": "...", "config": {...}, "nodes": [...], "edges": [...],
  "mode": "probe",
  "layer_index": 5, "kind": "mlp" | "attention_head", "neuron_index": 812,
  "input_texts": ["こんにちは", "さようなら", ...],
  "aggregation": "max_abs" | "mean_abs"  // 省略時"max_abs"
}

出力フォーマット(mode="probe"、成功時):
{
  "success": true, "mode": "probe",
  "layer_index": 5, "kind": "mlp", "neuron_index": 812,
  "results": [{"input_text": "こんにちは", "score": 3.2}, ...]
}

出力フォーマット(エラー時、両モード共通):
{"success": false, "error_type": "GraphValidationError" | "UnexpectedError" | "InvalidInput" | "TokenizerLoadError", "message": "..."}

使い方: python run_neuron_search.py < neuron_search_input.json
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.graph_validation import (  # noqa: E402
    GraphNode,
    GraphValidationError,
    build_input_map_multi,
    topological_sort,
)
from inference.neuron_search_utils import rank_scored_neurons, resolve_target_layers  # noqa: E402
from inference.tokenizer_utils import TokenizerLoadError  # noqa: E402


def main() -> None:
    raw_input = sys.stdin.read()
    try:
        request = json.loads(raw_input)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error_type": "InvalidInput", "message": str(e)}))
        sys.exit(1)

    try:
        result = run_neuron_search(request)
        print(json.dumps(result, ensure_ascii=False))
    except GraphValidationError as e:
        print(json.dumps({"success": False, "error_type": "GraphValidationError", "message": str(e)}, ensure_ascii=False))
        sys.exit(0)
    except TokenizerLoadError as e:
        print(json.dumps({"success": False, "error_type": "TokenizerLoadError", "message": str(e)}, ensure_ascii=False))
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"success": False, "error_type": "UnexpectedError", "message": str(e)}, ensure_ascii=False))
        sys.exit(1)


def _build_nodes_edges(request: dict) -> tuple[list[GraphNode], list[tuple[str, str]], str, str]:
    nodes = [
        GraphNode(
            id=n["id"], role=n["role"], layer_index=n.get("layer_index"),
            merge_mode=n.get("merge_mode", "add"),
        )
        for n in request["nodes"]
    ]
    edges = [(e["source"], e["target"]) for e in request["edges"]]

    if not nodes:
        raise GraphValidationError("グラフにノードが1つもありません。")

    entry_candidates = [n.id for n in nodes if n.role == "embedding"]
    exit_candidates = [n.id for n in nodes if n.role == "lm_head"]
    if len(entry_candidates) != 1:
        raise GraphValidationError(f"embeddingノードは必ず1つ必要です(現在: {len(entry_candidates)}個)。")
    if len(exit_candidates) != 1:
        raise GraphValidationError(f"lm_headノードは必ず1つ必要です(現在: {len(exit_candidates)}個)。")
    entry_node_id = entry_candidates[0]
    exit_node_id = exit_candidates[0]

    topological_sort(nodes, edges)
    build_input_map_multi(nodes, edges, entry_node_id=entry_node_id)

    return nodes, edges, entry_node_id, exit_node_id


def _create_executor(model_dir: Path, config: dict, nodes: list[GraphNode]):
    import torch  # noqa: PLC0415,F401

    from inference.graph_runner import LlamaGraphExecutor  # noqa: PLC0415

    executor = LlamaGraphExecutor(model_dir, config)
    executor.load_weights_for_nodes([n.id for n in nodes])
    return executor


def _tokenize(model_dir: Path, request: dict):
    import torch  # noqa: PLC0415

    input_text = request.get("input_text")
    input_tokens_text: list[str] | None = None
    if input_text is not None:
        from inference.tokenizer_utils import decode_tokens, encode_text, load_tokenizer  # noqa: PLC0415

        tokenizer = load_tokenizer(model_dir)
        input_ids_list = encode_text(
            tokenizer, input_text, model_dir=model_dir,
            use_chat_template=bool(request.get("use_chat_template", False)),
        )
        if not input_ids_list:
            raise GraphValidationError("入力テキストのトークナイズ結果が空でした。")
        input_tokens_text = [decode_tokens(tokenizer, [tid]) for tid in input_ids_list]
    else:
        input_ids_list = request.get("input_ids", [1, 2, 3, 4, 5])

    input_ids = torch.tensor([input_ids_list])
    return input_ids, input_ids_list, input_tokens_text


def _capture_keys_for_node(node: GraphNode) -> tuple[str | None, str | None]:
    """nodeのroleに応じて、(attention head_norms用キー, mlp neurons用キー)を返す。
    該当しない方はNone。graph_runner.execute()が内部で使う命名規則と一致させている。
    """
    if node.role == "attention_block":
        return f"{node.id}.head_norms", None
    if node.role == "mlp_block":
        return None, f"{node.id}.neurons"
    if node.role == "layer_block":
        return f"{node.id}.self_attn.head_norms", f"{node.id}.mlp.neurons"
    return None, None


def _aggregate(tensor, aggregation: str):
    """tensor: [..., seq_len]。最後の次元(トークン位置)方向に集約して1つ次元を落とす。"""
    if aggregation == "mean_abs":
        return tensor.abs().mean(dim=-1)
    return tensor.abs().max(dim=-1).values  # "max_abs"(デフォルト)


def run_neuron_search(request: dict) -> dict:
    mode = request.get("mode", "rank")
    if mode == "rank":
        return _run_rank(request)
    if mode == "probe":
        return _run_probe(request)
    raise GraphValidationError(f"未知のmodeです: {mode!r} ('rank'または'probe'を指定してください)")


def _run_rank(request: dict) -> dict:
    model_dir = Path(request["model_dir"])
    config = request["config"]
    nodes, edges, entry_node_id, exit_node_id = _build_nodes_edges(request)
    num_layers = int(config.get("num_hidden_layers", 0))

    neuron_kind = request.get("neuron_kind", "both")
    target_layers = set(resolve_target_layers(num_layers, request.get("target_layers")))
    top_k = int(request.get("top_k", 20))
    aggregation = request.get("aggregation", "max_abs")

    executor = _create_executor(model_dir, config, nodes)
    input_ids, input_ids_list, input_tokens_text = _tokenize(model_dir, request)

    neuron_capture: dict = {}
    executor.execute(
        input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
        kv_cache={}, position_offset=0, neuron_capture=neuron_capture,
    )

    scored: list[dict] = []
    for node in nodes:
        if node.layer_index is None or node.layer_index not in target_layers:
            continue
        attn_key, mlp_key = _capture_keys_for_node(node)

        if attn_key and attn_key in neuron_capture and neuron_kind in ("attention_head", "both"):
            head_norms = neuron_capture[attn_key][0]  # [num_heads, seq_len] (batch=1を除去)
            scores = _aggregate(head_norms, aggregation)  # [num_heads]
            # ヘッド数は多くても数十程度なので、全ヘッド分そのままスコアリング対象にする
            for head_idx in range(scores.shape[0]):
                scored.append({
                    "layer_index": node.layer_index, "kind": "attention_head",
                    "neuron_index": head_idx, "score": float(scores[head_idx]),
                })

        if mlp_key and mlp_key in neuron_capture and neuron_kind in ("mlp", "both"):
            mlp_intermediate = neuron_capture[mlp_key][0].transpose(0, 1)  # [intermediate_size, seq_len]
            scores = _aggregate(mlp_intermediate, aggregation)  # [intermediate_size]
            # MLP中間次元は数千に及ぶため、この1ノード内での上位top_k件だけを候補に残す。
            # (大域top_kは各グループの局所top_k候補の和集合に必ず含まれるため、
            #  正しさを保ったまま候補数を大幅に減らせる: あるニューロンが1つのノード内で
            #  自身のtop_k圏外なら、そのノードだけで既にtop_k件以上に負けているので、
            #  大域top_kにも入り得ない)
            import torch  # noqa: PLC0415

            local_k = min(top_k, scores.shape[0])
            top_scores, top_indices = torch.topk(scores, local_k)
            for score_val, neuron_idx in zip(top_scores.tolist(), top_indices.tolist()):
                scored.append({
                    "layer_index": node.layer_index, "kind": "mlp",
                    "neuron_index": neuron_idx, "score": float(score_val),
                })

    ranked = rank_scored_neurons(scored, top_k)

    return {
        "success": True,
        "mode": "rank",
        "input_token_ids": input_ids_list,
        "input_tokens_text": input_tokens_text,
        "ranked_neurons": ranked,
    }


def _run_probe(request: dict) -> dict:
    model_dir = Path(request["model_dir"])
    config = request["config"]
    nodes, edges, entry_node_id, exit_node_id = _build_nodes_edges(request)
    num_layers = int(config.get("num_hidden_layers", 0))

    layer_index = request.get("layer_index")
    kind = request.get("kind")
    neuron_index = request.get("neuron_index")
    if layer_index is None or kind not in ("mlp", "attention_head") or neuron_index is None:
        raise GraphValidationError(
            "probeモードには layer_index, kind('mlp'または'attention_head'), neuron_index の指定が必要です。"
        )
    if layer_index < 0 or layer_index >= num_layers:
        raise GraphValidationError(f"存在しない層番号です: {layer_index} (このモデルは{num_layers}層)")

    input_texts = request.get("input_texts")
    if not input_texts:
        raise GraphValidationError("probeモードには input_texts に1つ以上の入力文を指定してください。")

    aggregation = request.get("aggregation", "max_abs")

    target_node = next(
        (
            n for n in nodes
            if n.layer_index == layer_index
            and (
                (kind == "attention_head" and n.role in ("attention_block", "layer_block"))
                or (kind == "mlp" and n.role in ("mlp_block", "layer_block"))
            )
        ),
        None,
    )
    if target_node is None:
        raise GraphValidationError(f"層{layer_index}の{kind}ノードがグラフ内に見つかりません。")

    attn_key, mlp_key = _capture_keys_for_node(target_node)
    capture_key = attn_key if kind == "attention_head" else mlp_key

    executor = _create_executor(model_dir, config, nodes)

    results: list[dict] = []
    for text in input_texts:
        input_ids, _input_ids_list, _tokens_text = _tokenize(model_dir, {**request, "input_text": text})
        neuron_capture: dict = {}
        executor.execute(
            input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
            kv_cache={}, position_offset=0, neuron_capture=neuron_capture,
        )
        if capture_key not in neuron_capture:
            raise GraphValidationError(f"指定ノードの内部状態が記録されませんでした({capture_key})。")

        tensor = neuron_capture[capture_key][0]  # batch=1を除去
        if kind == "mlp":
            tensor = tensor.transpose(0, 1)  # [intermediate_size, seq_len]
        # kind == "attention_head" の場合は既に [num_heads, seq_len]

        if neuron_index < 0 or neuron_index >= tensor.shape[0]:
            raise GraphValidationError(
                f"neuron_indexが範囲外です: {neuron_index} (0〜{tensor.shape[0] - 1})"
            )

        neuron_series = tensor[neuron_index]  # [seq_len]
        score = float(_aggregate(neuron_series.unsqueeze(0), aggregation)[0])
        results.append({"input_text": text, "score": score})

    return {
        "success": True,
        "mode": "probe",
        "layer_index": layer_index,
        "kind": kind,
        "neuron_index": neuron_index,
        "results": results,
    }


if __name__ == "__main__":
    main()
