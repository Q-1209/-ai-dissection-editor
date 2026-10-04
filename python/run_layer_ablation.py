#!/usr/bin/env python3
"""
run_layer_ablation.py

Phase2: Layer Ablation Wizard — 層を無効化して影響を見るツール。

2026-09-12、漣さんとの確認事項:
- 「無効化」の定義: そのノード(Attentionブロック/MLPブロック/Layer全体)の出力を
  ゼロにする(残差接続も含めて、このノードの寄与を完全に断つ)
- 既存の「実行」タブでのエッジ手動繋ぎ替え(層をバイパスする接続を自分で組む)とは別の、
  独立した対話的ツールとして位置づける: ユーザーが観察タブでノードを選んで無効化した
  「その場」で、通常時(無効化なし)との違いを見る
- 影響度の指標: KLダイバージェンス(Prompt Sensitivity Mapと同じ、既存の
  compute_kl_divergenceをそのまま流用)に加えて、実際に生成されるテキストの変化も
  比較対象に含める

「全層を自動で1層ずつ無効化してランキングする」バッチ処理ではなく、ユーザーが
GUI上でチェックボックス等で選んだ1つ以上のノードを無効化した状態で、通常時との
差分(最後の位置の次トークン分布のKLダイバージェンス、生成テキストの変化)を
1回の実行で比較する。既存のグラフ実行エンジン(graph_runner.LlamaGraphExecutor)を
そのまま使い回し、GraphNode.ablated フラグ(2026-09-12追加)を立てるだけで実現している。

入力フォーマット(stdin):
{
  "model_dir": "...", "config": {...},
  "nodes": [...], "edges": [...],
  // nodes/edgesは実行/観察タブで組んだ現在のグラフ構造をそのまま使う
  "input_text": "こんにちは",       // または input_ids
  "ablate_node_ids": ["model.layers.3", "model.layers.7.self_attn"],  // 無効化するノードID(1つ以上)
  "compare_generation": true,       // 省略時true。生成テキストの比較も行うか
  "max_new_tokens": 20,
  "temperature": 0.0, "top_k": null, "top_p": null,
  "repetition_penalty": 1.0, "no_repeat_ngram_size": null
}

出力フォーマット(成功時):
{
  "success": true,
  "input_token_ids": [...], "input_tokens_text": [...] | null,
  "ablated_node_ids": [...],
  "kl_divergence": 1.234,               // 最後の位置の次トークン分布、通常時→無効化時のKL
  "top_token_changed": true,
  "baseline_top_token_id": 45, "ablated_top_token_id": 12,
  "baseline_top_token_text": "..." | null, "ablated_top_token_text": "..." | null,
  "generated_text_baseline": "..." | null,   // compare_generation=falseならnull
  "generated_text_ablated": "..." | null,
  "executed_order": [...]
}

出力フォーマット(エラー時):
{"success": false, "error_type": "GraphValidationError" | "UnexpectedError" | "InvalidInput" | "TokenizerLoadError", "message": "..."}

使い方: python run_layer_ablation.py < ablation_input.json
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# 標準出力をUTF-8に揃える。スクリプトとして実行されたときだけ行う(2026-09-22修正: テスト
# (test_layer_ablation_marking.py)がこのモジュールをimportした際にpytestの出力捕捉を壊していたため)
if __name__ == "__main__":
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


def main() -> None:
    raw_input = sys.stdin.read()
    try:
        request = json.loads(raw_input)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error_type": "InvalidInput", "message": str(e)}))
        sys.exit(1)

    try:
        result = run_ablation(request)
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


def mark_ablated_nodes(nodes: list[GraphNode], ablate_ids) -> None:
    """nodesの各GraphNode.ablatedを、idがablate_idsに含まれるかどうかで(再)設定する。

    全ノードに対してTrue/Falseを明示的に設定し直すため、以前の呼び出しで立てた
    フラグが残らない(baseline計算用に空集合を渡せば、全て確実にFalseへ戻せる)。
    ablate_idsに、グラフ内に存在しないノードIDが含まれる場合はGraphValidationError。
    """
    ablate_set = set(ablate_ids)
    node_ids = {n.id for n in nodes}
    unmatched = ablate_set - node_ids
    if unmatched:
        raise GraphValidationError(
            f"無効化対象として指定されたノードIDがグラフ内に存在しません: {sorted(unmatched)}"
        )
    for n in nodes:
        n.ablated = n.id in ablate_set


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


def run_ablation(request: dict) -> dict:
    model_dir = Path(request["model_dir"])
    config = request["config"]
    nodes, edges, entry_node_id, exit_node_id = _build_nodes_edges(request)

    ablate_node_ids = request.get("ablate_node_ids", [])
    if not ablate_node_ids:
        raise GraphValidationError("無効化するノードを1つ以上指定してください。")
    # 事前に妥当性だけ確認しておく(実際に立てるのは後段、baseline計算の後)
    mark_ablated_nodes(nodes, [])
    unmatched = set(ablate_node_ids) - {n.id for n in nodes}
    if unmatched:
        raise GraphValidationError(
            f"無効化対象として指定されたノードIDがグラフ内に存在しません: {sorted(unmatched)}"
        )

    import torch  # noqa: PLC0415

    from inference.graph_runner import LlamaGraphExecutor  # noqa: PLC0415
    from inference.sensitivity import compute_kl_divergence  # noqa: PLC0415

    executor = LlamaGraphExecutor(model_dir, config)
    executor.load_weights_for_nodes([n.id for n in nodes])

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
            raise GraphValidationError("入力テキストのトークナイズ結果が空でした。")
        input_tokens_text = [decode_tokens(tokenizer, [tid]) for tid in input_ids_list]
    else:
        input_ids_list = request.get("input_ids", [1, 2, 3, 4, 5])

    input_ids = torch.tensor([input_ids_list])

    # --- 1回のforward passでKLダイバージェンスを測定(通常時 vs 無効化時) ---
    mark_ablated_nodes(nodes, [])  # baseline: 全ノード無効化なし
    baseline_result = executor.execute(
        input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
        kv_cache={}, position_offset=0,
    )
    baseline_next_logits = baseline_result.logits[0, -1, :]

    mark_ablated_nodes(nodes, ablate_node_ids)
    ablated_result = executor.execute(
        input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
        kv_cache={}, position_offset=0,
    )
    ablated_next_logits = ablated_result.logits[0, -1, :]

    kl = compute_kl_divergence(baseline_next_logits, ablated_next_logits)
    baseline_top_id = int(torch.argmax(baseline_next_logits).item())
    ablated_top_id = int(torch.argmax(ablated_next_logits).item())

    baseline_top_text = None
    ablated_top_text = None
    if tokenizer is not None:
        from inference.tokenizer_utils import decode_tokens  # noqa: PLC0415

        baseline_top_text = decode_tokens(tokenizer, [baseline_top_id])
        ablated_top_text = decode_tokens(tokenizer, [ablated_top_id])

    result: dict = {
        "success": True,
        "input_token_ids": input_ids_list,
        "input_tokens_text": input_tokens_text,
        "ablated_node_ids": ablate_node_ids,
        "kl_divergence": kl,
        "top_token_changed": ablated_top_id != baseline_top_id,
        "baseline_top_token_id": baseline_top_id,
        "ablated_top_token_id": ablated_top_id,
        "baseline_top_token_text": baseline_top_text,
        "ablated_top_token_text": ablated_top_text,
        "generated_text_baseline": None,
        "generated_text_ablated": None,
        "executed_order": baseline_result.node_order,
    }

    # --- 生成テキストの比較(compare_generation=trueの場合のみ、コスト高のため) ---
    if request.get("compare_generation", True):
        from inference.sampling import SamplingConfig  # noqa: PLC0415

        sampling_config = SamplingConfig(
            temperature=float(request.get("temperature", 0.0)),
            top_k=request.get("top_k"),
            top_p=request.get("top_p"),
            repetition_penalty=float(request.get("repetition_penalty", 1.0)),
            no_repeat_ngram_size=request.get("no_repeat_ngram_size"),
        )
        max_new_tokens = int(request.get("max_new_tokens", 20))

        mark_ablated_nodes(nodes, [])
        gen_baseline = executor.generate(
            input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
            max_new_tokens=max_new_tokens, sampling_config=sampling_config,
            stop_token_ids=resolve_stop_token_ids(request, model_dir, tokenizer),
        )
        mark_ablated_nodes(nodes, ablate_node_ids)
        gen_ablated = executor.generate(
            input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
            max_new_tokens=max_new_tokens, sampling_config=sampling_config,
            stop_token_ids=resolve_stop_token_ids(request, model_dir, tokenizer),
        )

        if tokenizer is not None:
            from inference.tokenizer_utils import decode_tokens  # noqa: PLC0415

            result["generated_text_baseline"] = decode_tokens(tokenizer, gen_baseline.generated_token_ids)
            result["generated_text_ablated"] = decode_tokens(tokenizer, gen_ablated.generated_token_ids)

    return result


if __name__ == "__main__":
    main()
