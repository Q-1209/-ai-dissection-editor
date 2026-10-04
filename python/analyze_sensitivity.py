#!/usr/bin/env python3
"""
analyze_sensitivity.py

Phase2: Prompt Sensitivity Map のエントリポイント。標準入力からグラフ定義と
テキスト入力を受け取り、入力文の各トークンを1つずつ削除/置換しながら、
出力(次トークンのlogits分布)への影響度(KLダイバージェンス)を測定して、
JSON化して標準出力に返す。

入力文の長さ(トークン数)ぶんだけforward passを追加実行する(オリジナル1回 +
各位置1回ずつ)ため、他のエントリポイントより実行時間がかかる。

入力フォーマット(stdin):
{
  "model_dir": "/path/to/model",
  "config": { ... config.jsonの中身 ... },
  "nodes": [...],   // execute_graph.pyと同じ形式
  "edges": [...],
  "input_text": "こんにちは",       // または input_ids
  "input_ids": [1, 2, 3],
  "replacement_mode": "pad",       // "delete" | "pad" | "random" | "custom"。省略時は"pad"
  "custom_token_id": null,         // replacement_mode="custom"のとき必須
  "pad_token_id": null             // 省略時は0
}

出力フォーマット(stdout, 成功時):
{
  "success": true,
  "input_token_ids": [1, 2, 3],
  "input_tokens_text": ["こん", "にち", "は"],
  "replacement_mode": "pad",
  "results": [
    {
      "position": 0,
      "token_text": "こん",
      "kl_divergence": 0.42,
      "top_token_changed": false,
      "original_top_token_text": "。",
      "modified_top_token_text": "。"
    },
    ...
  ]
}

出力フォーマット(stdout, エラー時):
{
  "success": false,
  "error_type": "GraphValidationError" | "UnexpectedError" | "TokenizerLoadError",
  "message": "..."
}

使い方: python analyze_sensitivity.py < sensitivity_input.json
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


def main() -> None:
    raw_input = sys.stdin.read()
    try:
        request = json.loads(raw_input)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error_type": "InvalidInput", "message": str(e)}))
        sys.exit(1)

    try:
        result = run_sensitivity(request)
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


def run_sensitivity(request: dict) -> dict:
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
        raise GraphValidationError("グラフにノードが1つもありません。")

    entry_candidates = [n.id for n in nodes if n.role == "embedding"]
    exit_candidates = [n.id for n in nodes if n.role == "lm_head"]
    if len(entry_candidates) != 1:
        raise GraphValidationError(
            f"embeddingノードは必ず1つ必要です(現在: {len(entry_candidates)}個)。"
        )
    if len(exit_candidates) != 1:
        raise GraphValidationError(
            f"lm_headノードは必ず1つ必要です(現在: {len(exit_candidates)}個)。"
        )
    entry_node_id = entry_candidates[0]
    exit_node_id = exit_candidates[0]

    topological_sort(nodes, edges)
    build_input_map_multi(nodes, edges, entry_node_id=entry_node_id)

    # 入力の決定・トークナイズ・件数検証は、いずれもtorch非依存の処理
    # (tokenizersライブラリとPython標準機能のみ)なので、torch importより前に行う。
    # これにより、torch未インストール環境でも「入力が短すぎる」等の検証エラーを返せる。
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
        if len(input_ids_list) < 2:
            raise GraphValidationError(
                "感度分析には2トークン以上の入力が必要です(1トークンでは削除/置換の比較ができません)。"
            )
        input_tokens_text = [decode_tokens(tokenizer, [tid]) for tid in input_ids_list]
    else:
        input_ids_list = request.get("input_ids", [1, 2, 3, 4, 5])
        if len(input_ids_list) < 2:
            raise GraphValidationError("感度分析には2トークン以上の入力が必要です。")

    replacement_mode = request.get("replacement_mode", "pad")
    custom_token_id = request.get("custom_token_id")
    pad_token_id = request.get("pad_token_id")
    vocab_size = config.get("vocab_size")

    if replacement_mode == "custom" and custom_token_id is None:
        raise GraphValidationError("replacement_mode='custom'の場合、custom_token_idが必須です。")

    # torchはここで初めてimportする(実際の重み読み込み・計算に入る直前)
    import torch  # noqa: PLC0415

    from inference.graph_runner import LlamaGraphExecutor  # noqa: PLC0415
    from inference.sensitivity import run_sensitivity_analysis  # noqa: PLC0415

    executor = LlamaGraphExecutor(model_dir, config)
    executor.load_weights_for_nodes([n.id for n in nodes])

    input_ids = torch.tensor([input_ids_list])

    def forward_fn(ids: torch.Tensor) -> torch.Tensor:
        # 感度分析はKVキャッシュを使わない単発のforward pass。毎回新しいkv_cacheで実行する
        # (削除/置換のたびに系列長・内容が変わるため、キャッシュの使い回しはできない)。
        result = executor.execute(
            ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
            kv_cache={}, position_offset=0,
        )
        return result.logits

    sensitivity_results = run_sensitivity_analysis(
        forward_fn, input_ids, replacement_mode,
        pad_token_id=pad_token_id, custom_token_id=custom_token_id, vocab_size=vocab_size,
    )

    results_json = []
    for r in sensitivity_results:
        entry = {
            "position": r.position,
            "kl_divergence": r.kl_divergence,
            "top_token_changed": r.top_token_changed,
        }
        if tokenizer is not None:
            from inference.tokenizer_utils import decode_tokens  # noqa: PLC0415

            entry["token_text"] = decode_tokens(tokenizer, [r.original_token_id])
            entry["original_top_token_text"] = decode_tokens(tokenizer, [r.original_top_token_id])
            entry["modified_top_token_text"] = decode_tokens(tokenizer, [r.modified_top_token_id])
        results_json.append(entry)

    return {
        "success": True,
        "input_token_ids": input_ids_list,
        "input_tokens_text": input_tokens_text,
        "replacement_mode": replacement_mode,
        "results": results_json,
    }


if __name__ == "__main__":
    main()
