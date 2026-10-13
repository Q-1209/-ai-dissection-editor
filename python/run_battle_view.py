#!/usr/bin/env python3
"""
run_battle_view.py

Multi-Model Battle View: 同じ入力テキスト(または同じinput_ids)に対して、
複数のモデル(2個以上、可変)の「生成結果」と「内部状態(活性化値統計・
Attention重み)」を横並びで比較するためのエントリポイント。

2026-09-10、漣さんとの確認事項:
- 比較対象: 生成結果・内部状態の両方
- モデル数: 可変(N個)
- 複数モデルを同時にメモリへロードすることは許容する
- 異なるアーキテクチャ間の比較も想定する

異なるアーキテクチャ・異なる層数のモデル同士を比較する都合上、各モデルの
実行グラフは(execute_graph.py/inspect_internals.pyのようにユーザーが
キャンバスで組んだものではなく)モデル自身のconfig(num_hidden_layers)から
inference.battle_graph.build_default_execution_graph()で自動生成した
「オリジナル構造通りのデフォルトグラフ」を使う。これにより、モデルごとに
層数・hidden_sizeが違っても、各モデルは自分自身の構造の範囲内で独立に実行され、
破綻しない(モデル間でテンソルを混ぜ合わせるような処理は一切行わない)。

hidden_sizeが異なるモデル同士の活性化値統計・Attention重みは、そのままでは
値のスケール・行列サイズが揃わないため、比較軸を揃える処理(正規化等)は
行わず、各モデルの結果をそのまま独立に返す。GUI側でモデルごとに個別のカラムに
表示し、横に並べて目視比較する形にとどめる(将来的な高度化の余地として
claude-notes.mdに記録済み)。

入力フォーマット(stdin):
{
  "models": [
    {"model_dir": "/path/to/model_a", "config": {...}, "label": "Qwen2.5-0.5B"},
    {"model_dir": "/path/to/model_b", "config": {...}, "label": "Llama-3.2-1B"},
    ...  // 2個以上を想定するが、1個以上あれば動作する
  ],
  "input_text": "こんにちは",          // モデルごとに自分のtokenizerでエンコードする
  "input_ids": [1, 2, 3, 4, 5],       // input_textの代わりに直接トークンIDを指定(全モデル共通で使う)
  "compare_generation": true,          // 省略時true
  "compare_internals": true,           // 省略時true
  "max_new_tokens": 20,
  "temperature": 0.0,
  "top_k": null,
  "top_p": null,
  "repetition_penalty": 1.0,
  "no_repeat_ngram_size": null,
  "aggregate_heads": true              // inspect_internals.pyと同じ意味
}

出力フォーマット(stdout, 常にcode=0):
{
  "success": true,
  "results": [
    {
      "label": "Qwen2.5-0.5B",
      "model_dir": "/path/to/model_a",
      "success": true,
      "input_token_ids": [...],
      "input_tokens_text": [...] | null,
      "generated_token_ids": [...] | null,      // compare_generation=falseならnull
      "generated_text": "..." | null,
      "executed_order": [...],
      "activation_stats": {...} | null,          // compare_internals=falseならnull
      "attention_weights": {...} | null
    },
    {
      "label": "Llama-3.2-1B",
      "model_dir": "/path/to/model_b",
      "success": false,
      "error_type": "GraphValidationError" | "UnexpectedError" | "TokenizerLoadError",
      "message": "..."
    },
    ...
  ]
}

設計判断: 1つのモデルの読み込み・実行に失敗しても、他のモデルの結果には
影響させない(モデルごとにtry/exceptで隔離する)。大きめのモデルを含む組み合わせで
一部だけメモリ不足になる、といったケースでも、成功した分の比較結果は見られるようにするため。
全体としての"success"は、リクエスト自体の形式が正しく処理できた場合はtrue
(個々のモデルの失敗は結果配列の各要素のsuccess:falseで表現する)。

使い方: python run_battle_view.py < battle_input.json
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

from inference.graph_validation import GraphValidationError  # noqa: E402
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
        result = run_battle(request)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:  # noqa: BLE001
        # ここに来るのはリクエスト自体が壊れている場合のみ(models未指定等)。
        # 個々のモデルの実行失敗はrun_single_model内でキャッチ済み。
        print(json.dumps({"success": False, "error_type": "UnexpectedError", "message": str(e)}, ensure_ascii=False))
        sys.exit(1)


def run_battle(request: dict) -> dict:
    model_specs = request.get("models", [])
    if not model_specs:
        raise ValueError(tr("'models' には少なくとも1つのモデルを指定してください。", "Give at least one model in 'models'."))

    compare_generation = bool(request.get("compare_generation", True))
    compare_internals = bool(request.get("compare_internals", True))

    results = [
        run_single_model(spec, request, compare_generation, compare_internals)
        for spec in model_specs
    ]

    return {"success": True, "results": results}


def run_single_model(
    spec: dict, request: dict, compare_generation: bool, compare_internals: bool
) -> dict:
    """1モデル分の実行を行い、成功/失敗いずれの場合も結果辞書を返す
    (例外を外に伝播させず、他モデルの結果に影響しないようここで必ず握りつぶす)。
    """
    label = spec.get("label") or Path(spec["model_dir"]).name
    model_dir_str = spec["model_dir"]

    try:
        return _run_single_model_inner(spec, request, compare_generation, compare_internals, label, model_dir_str)
    except GraphValidationError as e:
        return {
            "label": label, "model_dir": model_dir_str, "success": False,
            "error_type": "GraphValidationError", "message": str(e),
        }
    except TokenizerLoadError as e:
        return {
            "label": label, "model_dir": model_dir_str, "success": False,
            "error_type": "TokenizerLoadError", "message": str(e),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "label": label, "model_dir": model_dir_str, "success": False,
            "error_type": "UnexpectedError", "message": str(e),
        }


def _run_single_model_inner(
    spec: dict,
    request: dict,
    compare_generation: bool,
    compare_internals: bool,
    label: str,
    model_dir_str: str,
) -> dict:
    model_dir = Path(model_dir_str)
    config = spec["config"]
    num_layers = int(config.get("num_hidden_layers", 0))

    from inference.battle_graph import build_default_execution_graph  # noqa: PLC0415

    nodes, edges = build_default_execution_graph(num_layers)

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
    entry_node_id = "model.embed_tokens"
    exit_node_id = "lm_head"

    generated_token_ids: list[int] | None = None
    generated_text: str | None = None
    executed_order: list[str] = []
    activation_stats: dict[str, dict] | None = None
    attention_weights: dict[str, dict] | None = None

    if compare_generation:
        from inference.sampling import SamplingConfig  # noqa: PLC0415

        sampling_config = SamplingConfig(
            temperature=float(request.get("temperature", 0.0)),
            top_k=request.get("top_k"),
            top_p=request.get("top_p"),
            repetition_penalty=float(request.get("repetition_penalty", 1.0)),
            no_repeat_ngram_size=request.get("no_repeat_ngram_size"),
        )
        max_new_tokens = int(request.get("max_new_tokens", 20))
        gen_result = executor.generate(
            input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
            max_new_tokens=max_new_tokens, sampling_config=sampling_config,
            stop_token_ids=resolve_stop_token_ids(request, model_dir, tokenizer),
        )
        generated_token_ids = gen_result.generated_token_ids
        executed_order = gen_result.executed_order
        if tokenizer is not None:
            from inference.tokenizer_utils import decode_tokens  # noqa: PLC0415

            generated_text = decode_tokens(tokenizer, generated_token_ids)

    if compare_internals:
        aggregate_heads = bool(request.get("aggregate_heads", True))
        exec_result = executor.execute(
            input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
            kv_cache={}, position_offset=0, capture_internals=True,
        )
        if not executed_order:
            executed_order = exec_result.node_order

        attention_weights = {}
        activation_stats = {}
        for key, tensor in exec_result.internals.items():
            node_id, _, kind = key.rpartition(".")
            if kind == "attn_weights":
                w = tensor[0].float()
                if aggregate_heads:
                    w = w.mean(dim=0)
                attention_weights[node_id] = {"shape": list(w.shape), "data": w.tolist()}
            elif kind == "output":
                t = tensor.float()
                activation_stats[node_id] = {
                    "mean": float(t.mean()), "std": float(t.std()),
                    "min": float(t.min()), "max": float(t.max()),
                }

        # デフォルトグラフはlayer_blockロールのまま(展開しない)ため、
        # inspect_internals.pyと同じ「{node_id}.self_attn / .mlp」からの
        # リネーム処理が常に必要になる。
        for oid in exec_result.node_order:
            if oid in attention_weights or oid in activation_stats:
                continue
            attn_sub_key = f"{oid}.self_attn"
            mlp_sub_key = f"{oid}.mlp"
            if attn_sub_key in attention_weights:
                attention_weights[oid] = attention_weights.pop(attn_sub_key)
            if mlp_sub_key in activation_stats:
                activation_stats[oid] = activation_stats.pop(mlp_sub_key)
            elif attn_sub_key in activation_stats:
                activation_stats[oid] = activation_stats.pop(attn_sub_key)

    if not executed_order:
        executed_order = [n.id for n in nodes]

    return {
        "label": label,
        "model_dir": model_dir_str,
        "success": True,
        "input_token_ids": input_ids_list,
        "input_tokens_text": input_tokens_text,
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "executed_order": executed_order,
        "activation_stats": activation_stats,
        "attention_weights": attention_weights,
    }


if __name__ == "__main__":
    main()
