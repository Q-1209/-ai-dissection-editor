#!/usr/bin/env python3
"""
inspect_internals.py

Phase2: 内部状態可視化のエントリポイント。標準入力からグラフ定義とテキスト入力を
受け取り、1回のforward passを実行しながら各Attention/MLPブロックの内部状態
(Attention重み・活性化値の統計)を記録して、JSON化して標準出力に返す。

Phase1のexecute_graph.py(テキスト生成用)とは目的が異なるため、別スクリプトとして
分離している。複数トークンを生成しながら全ステップの内部状態を溜め込むと
データ量が膨大になるため、こちらは常に「1回のforward pass」のみを対象にする
(生成ではなく、既存の入力に対する内部状態の観察が目的)。

入力フォーマット(stdin):
{
  "model_dir": "/path/to/model",
  "config": { ... config.jsonの中身 ... },
  "nodes": [...],   // execute_graph.pyと同じ形式
  "edges": [...],
  "input_text": "こんにちは",       // または input_ids
  "input_ids": [1, 2, 3],
  "aggregate_heads": true          // true: ヘッド平均(トークン×トークン)を返す。
                                    // false: 全ヘッド分(ヘッド×トークン×トークン)を返す。
                                    // 省略時はtrue(データ量を抑える方をデフォルトにする)。
}

出力フォーマット(stdout, 成功時):
{
  "success": true,
  "input_token_ids": [1, 2, 3],
  "input_tokens_text": ["こん", "にち", "は"],  // tokenizer.jsonがあれば各トークンの文字列表現も返す
  "executed_order": [...],
  "attention_weights": {
    "model.layers.0.self_attn": {
      "shape": [3, 3],              // aggregate_heads=trueなら[seq_len, seq_len]
      "data": [[...], [...], [...]] // ネストしたリスト(JSON化されたテンソル)
    },
    ...
  },
  "activation_stats": {
    "model.layers.0.self_attn": {"mean": 0.01, "std": 1.2, "min": -5.0, "max": 5.0},
    "model.layers.0.mlp": {...},
    ...
  }
}

出力フォーマット(stdout, エラー時):
{
  "success": false,
  "error_type": "GraphValidationError" | "UnexpectedError" | "TokenizerLoadError",
  "message": "..."
}

使い方: python inspect_internals.py < internals_input.json
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
        result = run_inspection(request)
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


def run_inspection(request: dict) -> dict:
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

    import torch  # noqa: PLC0415

    from inference.graph_runner import LlamaGraphExecutor  # noqa: PLC0415

    executor = LlamaGraphExecutor(model_dir, config)
    executor.load_weights_for_nodes([n.id for n in nodes])

    input_text = request.get("input_text")
    tokenizer = None
    input_tokens_text: list[str] | None = None
    if input_text is not None:
        from inference.tokenizer_utils import encode_text, load_tokenizer  # noqa: PLC0415

        tokenizer = load_tokenizer(model_dir)
        input_ids_list = encode_text(
            tokenizer, input_text, model_dir=model_dir,
            use_chat_template=bool(request.get("use_chat_template", False)),
        )
        if not input_ids_list:
            raise GraphValidationError(tr("入力テキストのトークナイズ結果が空でした。", "The input text tokenized to nothing."))
        from inference.tokenizer_utils import decode_tokens  # noqa: PLC0415

        input_tokens_text = [decode_tokens(tokenizer, [tid]) for tid in input_ids_list]
    else:
        input_ids_list = request.get("input_ids", [1, 2, 3, 4, 5])

    input_ids = torch.tensor([input_ids_list])
    aggregate_heads = bool(request.get("aggregate_heads", True))

    result = executor.execute(
        input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
        kv_cache={}, position_offset=0, capture_internals=True,
    )

    attention_weights: dict[str, dict] = {}
    activation_stats: dict[str, dict] = {}

    for key, tensor in result.internals.items():
        node_id, _, kind = key.rpartition(".")
        if kind == "attn_weights":
            w = tensor[0].float()  # [num_heads, seq_len, total_key_len] (batch=1を除去)
            if aggregate_heads:
                w = w.mean(dim=0)  # [seq_len, total_key_len]
            attention_weights[node_id] = {
                "shape": list(w.shape),
                "data": w.tolist(),
            }
        elif kind == "output":
            t = tensor.float()
            activation_stats[node_id] = {
                "mean": float(t.mean()),
                "std": float(t.std()),
                "min": float(t.min()),
                "max": float(t.max()),
            }

    # layer_blockロール(GUIで層を折りたたんだままにしている場合)は、内部で
    # "{node_id}.self_attn" / "{node_id}.mlp" という2つのサブIDに分けて実行される
    # (run_attention_block/run_mlp_blockのcapture呼び出し時の命名規則、graph_runner.py参照)。
    # そのため、attention_weights/activation_statsのキーがexecuted_order(元のnode_id)と
    # 一致せず、GUI側で照合できないという不具合があった。
    # ここで、executed_orderに実際に登場するnode_idを基準に、対応するサブIDの結果を
    # 付け替える(実質的にはキーのリネーム)。
    # - Attention重みは "{node_id}.self_attn" のものをそのままnode_idに割り当てる
    #   (layer_block内のAttention部分の注目パターンを代表値として使う)。
    # - 活性化値統計は "{node_id}.mlp" (ブロック全体の最終出力)を優先し、無ければ
    #   "{node_id}.self_attn" にフォールバックする。
    for oid in result.node_order:
        if oid in attention_weights or oid in activation_stats:
            continue  # 既に(展開済みノードとして)直接キーが存在する場合はそのまま
        attn_sub_key = f"{oid}.self_attn"
        mlp_sub_key = f"{oid}.mlp"
        if attn_sub_key in attention_weights:
            attention_weights[oid] = attention_weights.pop(attn_sub_key)
        if mlp_sub_key in activation_stats:
            activation_stats[oid] = activation_stats.pop(mlp_sub_key)
        elif attn_sub_key in activation_stats:
            activation_stats[oid] = activation_stats.pop(attn_sub_key)

    return {
        "success": True,
        "input_token_ids": input_ids_list,
        "input_tokens_text": input_tokens_text,
        "executed_order": result.node_order,
        "attention_weights": attention_weights,
        "activation_stats": activation_stats,
    }


if __name__ == "__main__":
    main()
