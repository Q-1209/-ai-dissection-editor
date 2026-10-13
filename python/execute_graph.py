#!/usr/bin/env python3
"""
execute_graph.py

Electron側から呼び出されるPhase1実行のエントリポイント。
標準入力からグラフ定義(ノード+エッジ)のJSONを受け取り、実際に推論(複数トークンの
逐次生成、KVキャッシュ使用)を実行して、結果(またはエラー)をJSONで標準出力に返す。

入力フォーマット(stdin):
{
  "model_dir": "/path/to/model",
  "config": { ... config.jsonの中身 ... },
  "nodes": [
    {"id": "model.layers.0.self_attn", "role": "attention_block", "layer_index": 0},
    {"id": "model.layers.0.mlp", "role": "mlp_block", "layer_index": 0},
    ...
  ],
  "edges": [
    {"source": "model.layers.0.self_attn", "target": "model.layers.0.mlp"},
    ...
  ],
  "input_text": "こんにちは",          // テキスト入力(推奨)。tokenizer.jsonが必要
  "input_ids": [1, 2, 3, 4, 5],     // input_textの代わりに直接トークンIDを指定することも可能
  "max_new_tokens": 20,             // 省略時は20
  "temperature": 0.0,               // 省略時は0.0(貪欲デコード)。0より大きいとサンプリング
  "top_k": null,                    // 省略時は無効
  "top_p": null,                    // 省略時は無効
  "repetition_penalty": 1.0,        // 省略時は1.0(無効)
  "no_repeat_ngram_size": null      // 省略時は無効。例えば3を指定すると同じ3-gramの繰り返しを禁止
}

出力フォーマット(stdout, 成功時):
{
  "success": true,
  "input_token_ids": [1, 2, 3],
  "generated_token_ids": [45, 12, ...],
  "generated_text": "...",
  "executed_order": ["model.layers.0.self_attn", ...]
}

出力フォーマット(stdout, エラー時):
{
  "success": false,
  "error_type": "GraphValidationError" | "UnexpectedError" | "TokenizerLoadError",
  "message": "..."
}

使い方: python execute_graph.py < graph_input.json
"""

import io
import json
import sys
from pathlib import Path

# Windows環境でのCP932エンコーディング問題を回避(inspect_model.pyと同じ対処)
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
from inference.tokenizer_utils import TokenizerLoadError, resolve_stop_token_ids  # noqa: E402
from inference.i18n import tr  # noqa: E402


def main() -> None:
    raw_input = sys.stdin.read()
    try:
        request = json.loads(raw_input)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error_type": "InvalidInput", "message": str(e)}))
        sys.exit(1)

    try:
        result = run_execution(request)
        print(json.dumps(result, ensure_ascii=False))
    except GraphValidationError as e:
        print(json.dumps({"success": False, "error_type": "GraphValidationError", "message": str(e)}, ensure_ascii=False))
        sys.exit(0)  # バリデーションエラーは「正常に検出できた失敗」なので0で終了
    except TokenizerLoadError as e:
        print(json.dumps({"success": False, "error_type": "TokenizerLoadError", "message": str(e)}, ensure_ascii=False))
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"success": False, "error_type": "UnexpectedError", "message": str(e)}, ensure_ascii=False))
        sys.exit(1)


def run_execution(request: dict) -> dict:
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

    # トポロジカルソート・入力元解決(孤立ノードの検出含む)はtorch非依存なので、
    # ここまではtorch未インストール環境でもエラー内容を返せる。
    # build_input_map_multi()は合流(複数入力)を許可する(旧build_input_mapとは異なり、
    # 単一入力のみに制限しない)。
    topological_sort(nodes, edges)  # 循環検出のためだけに呼ぶ(結果はexecute()内で再計算)
    build_input_map_multi(nodes, edges, entry_node_id=entry_node_id)  # 孤立ノードの検出

    # torchはここで初めてimportする(実際の重み読み込み・計算に入る直前)
    import torch  # noqa: PLC0415

    from inference.graph_runner import LlamaGraphExecutor  # noqa: PLC0415

    executor = LlamaGraphExecutor(model_dir, config)
    executor.load_weights_for_nodes([n.id for n in nodes])

    # 入力の決定: input_text(テキスト、要tokenizer.json) > input_ids(直接指定) > デフォルト
    input_text = request.get("input_text")
    tokenizer = None
    if input_text is not None:
        from inference.tokenizer_utils import encode_text, load_tokenizer  # noqa: PLC0415

        tokenizer = load_tokenizer(model_dir)
        input_ids_list = encode_text(
            tokenizer, input_text, model_dir=model_dir,
            use_chat_template=bool(request.get("use_chat_template", False)),
        )
        if not input_ids_list:
            raise GraphValidationError(tr("入力テキストのトークナイズ結果が空でした。", "The input text tokenized to nothing."))
    else:
        input_ids_list = request.get("input_ids", [1, 2, 3, 4, 5])  # 未指定時はダミートークン

    input_ids = torch.tensor([input_ids_list])
    max_new_tokens = int(request.get("max_new_tokens", 20))

    from inference.sampling import SamplingConfig  # noqa: PLC0415

    sampling_config = SamplingConfig(
        temperature=float(request.get("temperature", 0.0)),
        top_k=request.get("top_k"),
        top_p=request.get("top_p"),
        repetition_penalty=float(request.get("repetition_penalty", 1.0)),
        no_repeat_ngram_size=request.get("no_repeat_ngram_size"),
    )

    gen_result = executor.generate(
        input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
        max_new_tokens=max_new_tokens, sampling_config=sampling_config,
        stop_token_ids=resolve_stop_token_ids(request, model_dir, tokenizer),
    )

    generated_text = None
    if tokenizer is not None:
        from inference.tokenizer_utils import decode_tokens  # noqa: PLC0415

        generated_text = decode_tokens(tokenizer, gen_result.generated_token_ids)

    return {
        "success": True,
        "input_token_ids": input_ids_list,
        "generated_token_ids": gen_result.generated_token_ids,
        "generated_text": generated_text,
        "executed_order": gen_result.executed_order,
    }


if __name__ == "__main__":
    main()
