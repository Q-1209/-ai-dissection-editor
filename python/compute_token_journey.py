#!/usr/bin/env python3
"""
compute_token_journey.py

Phase2: Token Journey のエントリポイント。標準入力からグラフ定義とテキスト入力を
受け取り、1回のforward passを実行しながら各ブロック(Attention直後・MLP直後)の
出力を記録し、各トークン位置についてその時点で埋め込み空間上最も近い単語
(nearest neighbor)を計算して、JSON化して標準出力に返す。

入力フォーマット(stdin):
{
  "model_dir": "/path/to/model",
  "config": { ... config.jsonの中身 ... },
  "nodes": [...],   // execute_graph.pyと同じ形式
  "edges": [...],
  "input_text": "こんにちは",       // または input_ids
  "input_ids": [1, 2, 3],
  "top_k": 1                       // 各(トークン位置,ブロック)ごとに何件の近傍単語を返すか。省略時1
}

出力フォーマット(stdout, 成功時):
{
  "success": true,
  "input_token_ids": [1, 2, 3],
  "input_tokens_text": ["こん", "にち", "は"],
  "block_order": ["model.layers.0.self_attn", "model.layers.0.mlp", ...],
  "journey": {
    "model.layers.0.self_attn": [
      [{"token_text": "...", "similarity": 0.8}, ...],
      [...],
      ...
    ],
    ...
  }
}

出力フォーマット(stdout, エラー時):
{
  "success": false,
  "error_type": "GraphValidationError" | "UnexpectedError" | "TokenizerLoadError",
  "message": "..."
}

使い方: python compute_token_journey.py < journey_input.json
"""

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
from inference.tokenizer_utils import TokenizerLoadError  # noqa: E402
from inference.i18n import tr  # noqa: E402


def main() -> None:
    raw_input = sys.stdin.read()
    try:
        request = json.loads(raw_input)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error_type": "InvalidInput", "message": str(e)}))
        sys.exit(1)

    try:
        result = run_token_journey(request)
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


def run_token_journey(request: dict, apply_final_norm: bool = True) -> dict:
    """apply_final_norm: Falseにすると、logit lens補正(最終RMSNormの適用)を行わず、
    生の隠れ状態で最近傍探索する(補正の効果をA/B比較するためのデバッグ用引数。
    通常のGUI経由の呼び出しではrequestに含まれないため常にTrueのまま=補正あり)。
    """
    model_dir = Path(request["model_dir"])
    config = request["config"]

    nodes = [
        GraphNode(
            id=n["id"],
            role=n["role"],
            layer_index=n.get("layer_index"),
            merge_mode=n.get("merge_mode", "add"),
        )
        for n in request["nodes"]
    ]
    edges = [(e["source"], e["target"]) for e in request["edges"]]

    if not nodes:
        raise GraphValidationError(tr("グラフにノードが1つもありません。", "The graph has no nodes."))

    entry_candidates = [n.id for n in nodes if n.role == "embedding"]
    exit_candidates = [n.id for n in nodes if n.role == "lm_head"]
    if len(entry_candidates) != 1:
        raise GraphValidationError(
            tr(f"embeddingノードは必ず1つ必要です(現在: {len(entry_candidates)}個)。", f"Exactly one embedding node is required (found {len(entry_candidates)}).")
        )
    if len(exit_candidates) != 1:
        raise GraphValidationError(
            tr(f"lm_headノードは必ず1つ必要です(現在: {len(exit_candidates)}個)。", f"Exactly one lm_head node is required (found {len(exit_candidates)}).")
        )
    entry_node_id = entry_candidates[0]
    exit_node_id = exit_candidates[0]

    topological_sort(nodes, edges)
    build_input_map_multi(nodes, edges, entry_node_id=entry_node_id)

    input_text = request.get("input_text")
    tokenizer = None
    input_tokens_text: list[str] | None = None
    if input_text is not None:
        from inference.tokenizer_utils import decode_tokens, encode_text, load_tokenizer  # noqa: PLC0415

        tokenizer = load_tokenizer(model_dir)
        input_ids_list = encode_text(
            tokenizer, input_text, model_dir=model_dir,
            use_chat_template=bool(request.get("use_chat_template", False)),
        )
        if not input_ids_list:
            raise GraphValidationError(tr("入力テキストのトークナイズ結果が空でした。", "The input text tokenized to nothing."))
        input_tokens_text = [decode_tokens(tokenizer, [tid]) for tid in input_ids_list]
    else:
        input_ids_list = request.get("input_ids", [1, 2, 3, 4, 5])

    top_k = int(request.get("top_k", 1))

    import torch  # noqa: PLC0415

    from inference.graph_runner import LlamaGraphExecutor  # noqa: PLC0415
    from inference.token_journey import compute_token_journey  # noqa: PLC0415

    executor = LlamaGraphExecutor(model_dir, config)
    executor.load_weights_for_nodes([n.id for n in nodes])

    input_ids = torch.tensor([input_ids_list])

    result = executor.execute(
        input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
        kv_cache={}, position_offset=0, capture_internals=True,
    )

    block_order: list[str] = []
    for oid in result.node_order:
        for suffix in (".self_attn", ".mlp"):
            candidate = f"{oid}{suffix}"
            if f"{candidate}.output" in result.internals and candidate not in block_order:
                block_order.append(candidate)
        if f"{oid}.output" in result.internals and oid not in block_order:
            block_order.append(oid)

    embedding_matrix = executor._w("model.embed_tokens.weight")  # noqa: SLF001
    # logit lens補正: lm_headが実際に適用するのと同じ最終RMSNormを中間層の
    # 隠れ状態にも適用してから埋め込み空間で最近傍探索する。これがないと、
    # Qwen2.5系で知られるMassive Activations(層2以降で一部次元が異常に肥大化する
    # 現象)により、中間層以降のほぼ全てのブロック・トークン位置が無関係な同一の
    # 単語に類似度最大として吸着してしまう(token_journey.pyのcompute_token_journey
    # docstring参照)。
    final_norm_weight = executor._w("model.norm.weight") if apply_final_norm else None  # noqa: SLF001

    journey = compute_token_journey(
        result.internals, block_order, embedding_matrix, seq_len=len(input_ids_list), top_k=top_k,
        final_norm_weight=final_norm_weight, norm_eps=executor.norm_eps,
    )

    journey_json: dict[str, list[list[dict]]] = {}
    for block_id, per_position in journey.items():
        journey_json[block_id] = []
        for matches in per_position:
            entry_list = []
            for m in matches:
                item = {"token_id": m.token_id, "similarity": m.similarity}
                if tokenizer is not None:
                    from inference.tokenizer_utils import decode_tokens  # noqa: PLC0415

                    item["token_text"] = decode_tokens(tokenizer, [m.token_id])
                entry_list.append(item)
            journey_json[block_id].append(entry_list)

    return {
        "success": True,
        "input_token_ids": input_ids_list,
        "input_tokens_text": input_tokens_text,
        "block_order": block_order,
        "journey": journey_json,
        "applied_final_norm": apply_final_norm,
    }


if __name__ == "__main__":
    main()
