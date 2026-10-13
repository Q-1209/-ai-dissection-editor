#!/usr/bin/env python3
"""
run_activation_steering.py

Phase3: Activation Steering Studio — 活性化操作による挙動誘導。
(2026-09-22追加: Refusal Behavior Tunerの「方向除去」モード・拒否定型句カウント、全タブ共通の
チャットテンプレート対応もこのエントリポイントで扱う。詳細はinference/refusal_utils.py・
inference/tokenizer_utils.py参照)

2026-09-15、漣さんとの確認事項:
- ステアリングベクトルの作り方: 対照的な2つのプロンプト群(positive/negative)の
  最後のトークン位置の活性化値の平均の差分("Contrastive Activation Addition"系)
- 適用対象: ユーザーが選んだ1つ以上のノード(GUI上のチェックボックス選択、
  Layer Ablation Wizardと同じUXパターン)。各ノードにそのノード自身の差分ベクトルを、
  共通のscaleで加算する
- 効果の確認方法: 生成テキストの変化(目視)に加え、KLダイバージェンス(既存の
  compute_kl_divergenceを流用)も定量指標として提示する

Layer Ablation Wizard(run_layer_ablation.py)と同じ設計パターンを踏襲している:
既存のグラフ実行エンジン(graph_runner.LlamaGraphExecutor)をそのまま使い回し、
新しく追加したsteering_vectors/steering_scale引数を渡すだけで実現している。

入力フォーマット(stdin):
{
  "model_dir": "...", "config": {...},
  "nodes": [...], "edges": [...],
  // nodes/edgesは実行/観察タブで組んだ現在のグラフ構造をそのまま使う
  "positive_prompts": ["ありがとう、助かりました", "とても嬉しいです"],
  "negative_prompts": ["最悪だ", "もう嫌になった"],
  "target_node_ids": ["model.layers.5", "model.layers.6"],  // ベクトルを計算・適用する対象ノード(1つ以上)
  "scale": 4.0,                     // ステアリングの強さ(符号反転で逆方向)
  "input_text": "こんにちは",
  "compare_generation": true,       // 省略時true。生成テキストの比較も行うか
  "max_new_tokens": 20,
  "temperature": 0.0, "top_k": null, "top_p": null,
  "repetition_penalty": 1.0, "no_repeat_ngram_size": null,
  "measure_sycophancy": false,      // Sycophancy Reducer(2026-09-15追加)用。省略時false。
                                     // trueかつcompare_generation=trueの場合、生成テキスト中の
                                     // 迎合的定型語彙の出現回数を機械的指標として追加で返す
  "measure_refusal": false,         // Refusal Behavior Tuner(2026-09-22追加)用。省略時false。
                                     // 拒否定型句の出現回数を同様に返す
  "intervention_mode": "add",       // 2026-09-22追加。"add"(従来どおりscale倍して加算)または
                                     // "ablate_direction"(差分ベクトルの方向の成分を射影で除去。
                                     // scaleは使わない)。省略時"add"
  "use_chat_template": false        // 2026-09-22追加(全タブ共通)。trueなら入力文・対照プロンプトを
                                     // モデルのチャットテンプレートで包んでからエンコードし、
                                     // 生成は発話終了トークンで打ち切る
}

出力フォーマット(成功時):
{
  "success": true,
  "target_node_ids": [...],
  "scale": 4.0,
  "vector_norms": {"model.layers.5": 12.34, ...},   // 各対象ノードの差分ベクトルのL2ノルム(GUI表示用)
  "input_token_ids": [...], "input_tokens_text": [...] | null,
  "kl_divergence": 1.234,               // 最後の位置の次トークン分布、通常時→ステアリング適用時のKL
  "top_token_changed": true,
  "baseline_top_token_id": 45, "steered_top_token_id": 12,
  "baseline_top_token_text": "..." | null, "steered_top_token_text": "..." | null,
  "generated_text_baseline": "..." | null,   // compare_generation=falseならnull
  "generated_text_steered": "..." | null,
  "sycophancy_phrase_count_baseline": {"total": 2, "by_phrase": {"素晴らしい": 1, ...}} | null,
  "sycophancy_phrase_count_steered": {"total": 0, "by_phrase": {}} | null,
  "refusal_phrase_count_baseline": {...} | null,   // measure_refusal=trueの場合のみ(形式は上と同じ)
  "refusal_phrase_count_steered": {...} | null,
  "intervention_mode": "add" | "ablate_direction",
  "used_chat_template": false,
  "executed_order": [...]
}

出力フォーマット(エラー時):
{"success": false, "error_type": "GraphValidationError" | "UnexpectedError" | "InvalidInput" | "TokenizerLoadError", "message": "..."}

使い方: python run_activation_steering.py < steering_input.json
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
from inference.tokenizer_utils import TokenizerLoadError, resolve_stop_token_ids  # noqa: E402
from inference.i18n import tr  # noqa: E402

VALID_INTERVENTION_MODES = ("add", "ablate_direction")


def main() -> None:
    raw_input = sys.stdin.read()
    try:
        request = json.loads(raw_input)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error_type": "InvalidInput", "message": str(e)}))
        sys.exit(1)

    try:
        result = run_steering(request)
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
        raise GraphValidationError(tr("グラフにノードが1つもありません。", "The graph has no nodes."))

    entry_candidates = [n.id for n in nodes if n.role == "embedding"]
    exit_candidates = [n.id for n in nodes if n.role == "lm_head"]
    if len(entry_candidates) != 1:
        raise GraphValidationError(tr(f"embeddingノードは必ず1つ必要です(現在: {len(entry_candidates)}個)。", f"Exactly one embedding node is required (found {len(entry_candidates)})."))
    if len(exit_candidates) != 1:
        raise GraphValidationError(tr(f"lm_headノードは必ず1つ必要です(現在: {len(exit_candidates)}個)。", f"Exactly one lm_head node is required (found {len(exit_candidates)})."))
    entry_node_id = entry_candidates[0]
    exit_node_id = exit_candidates[0]

    topological_sort(nodes, edges)
    build_input_map_multi(nodes, edges, entry_node_id=entry_node_id)

    return nodes, edges, entry_node_id, exit_node_id


def run_steering(request: dict) -> dict:
    model_dir = Path(request["model_dir"])
    config = request["config"]
    nodes, edges, entry_node_id, exit_node_id = _build_nodes_edges(request)

    positive_prompts = request.get("positive_prompts", [])
    negative_prompts = request.get("negative_prompts", [])
    target_node_ids = request.get("target_node_ids", [])
    if not positive_prompts:
        raise GraphValidationError(tr("positive_prompts を1つ以上指定してください。", "Give at least one positive prompt."))
    if not negative_prompts:
        raise GraphValidationError(tr("negative_prompts を1つ以上指定してください。", "Give at least one negative prompt."))
    if not target_node_ids:
        raise GraphValidationError(tr("target_node_ids(ステアリング対象ノード)を1つ以上指定してください。", "Give at least one steering target node (target_node_ids)."))

    intervention_mode = request.get("intervention_mode", "add")
    if intervention_mode not in VALID_INTERVENTION_MODES:
        raise GraphValidationError(
            tr(f"intervention_mode は {VALID_INTERVENTION_MODES} のいずれかを指定してください(指定値: {intervention_mode!r})。", f"intervention_mode must be one of {VALID_INTERVENTION_MODES} (got {intervention_mode!r}).")
        )
    use_chat_template = bool(request.get("use_chat_template", False))

    node_ids = {n.id for n in nodes}
    unmatched = set(target_node_ids) - node_ids
    if unmatched:
        raise GraphValidationError(
            tr(f"ステアリング対象として指定されたノードIDがグラフ内に存在しません: {sorted(unmatched)}", f"Steering target node IDs are not in the graph: {sorted(unmatched)}")
        )

    import torch  # noqa: PLC0415

    from inference.graph_runner import LlamaGraphExecutor  # noqa: PLC0415
    from inference.sensitivity import compute_kl_divergence  # noqa: PLC0415
    from inference.steering import compute_steering_vectors  # noqa: PLC0415
    from inference.tokenizer_utils import decode_tokens, encode_text, load_tokenizer  # noqa: PLC0415

    executor = LlamaGraphExecutor(model_dir, config)
    executor.load_weights_for_nodes([n.id for n in nodes])

    tokenizer = load_tokenizer(model_dir)

    def encode(text: str) -> torch.Tensor:
        # 対照プロンプトもチャットテンプレートで包む(2026-09-22)。こうすると各プロンプトの
        # 「最後のトークン位置」がアシスタント応答の開始直前になり、「これから断るかどうか」が
        # 表れる位置の活性化値でベクトルを作ることになる(Arditi et al.と同じ位置の取り方)
        ids = encode_text(tokenizer, text, model_dir=model_dir, use_chat_template=use_chat_template)
        if not ids:
            raise GraphValidationError(tr(f"入力テキストのトークナイズ結果が空でした: {text!r}", f"The input text tokenized to nothing: {text!r}"))
        return torch.tensor([ids])

    positive_input_ids = [encode(p) for p in positive_prompts]
    negative_input_ids = [encode(p) for p in negative_prompts]

    # --- ステアリングベクトルの計算(対照プロンプト群の平均活性化値の差分) ---
    steering_vectors = compute_steering_vectors(
        executor, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
        positive_input_ids=positive_input_ids, negative_input_ids=negative_input_ids,
        target_node_ids=target_node_ids,
    )
    vector_norms = {nid: float(vec.norm()) for nid, vec in steering_vectors.items()}

    scale = float(request.get("scale", 1.0))

    # 介入方式(2026-09-22追加、Refusal Behavior Tuner): "add"は従来どおりscale倍して加算、
    # "ablate_direction"は差分ベクトルの方向の成分を射影で除去する(scaleは使わない)。
    if intervention_mode == "ablate_direction":
        intervention_kwargs: dict = {"ablation_directions": steering_vectors}
    else:
        intervention_kwargs = {"steering_vectors": steering_vectors, "steering_scale": scale}

    # --- 本題の入力に対して、通常時 vs ステアリング適用時のKLダイバージェンスを測定 ---
    input_text = request.get("input_text")
    input_ids_list: list[int]
    input_tokens_text: list[str] | None = None
    if input_text is not None:
        input_ids_list = encode_text(
            tokenizer, input_text, model_dir=model_dir, use_chat_template=use_chat_template
        )
        if not input_ids_list:
            raise GraphValidationError(tr("入力テキストのトークナイズ結果が空でした。", "The input text tokenized to nothing."))
        input_tokens_text = [decode_tokens(tokenizer, [tid]) for tid in input_ids_list]
    else:
        input_ids_list = request.get("input_ids", [1, 2, 3, 4, 5])

    input_ids = torch.tensor([input_ids_list])

    baseline_result = executor.execute(
        input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
        kv_cache={}, position_offset=0,
    )
    baseline_next_logits = baseline_result.logits[0, -1, :]

    steered_result = executor.execute(
        input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
        kv_cache={}, position_offset=0, **intervention_kwargs,
    )
    steered_next_logits = steered_result.logits[0, -1, :]

    kl = compute_kl_divergence(baseline_next_logits, steered_next_logits)
    baseline_top_id = int(torch.argmax(baseline_next_logits).item())
    steered_top_id = int(torch.argmax(steered_next_logits).item())

    result: dict = {
        "success": True,
        "target_node_ids": target_node_ids,
        "scale": scale,
        "vector_norms": vector_norms,
        "input_token_ids": input_ids_list,
        "input_tokens_text": input_tokens_text,
        "kl_divergence": kl,
        "top_token_changed": steered_top_id != baseline_top_id,
        "baseline_top_token_id": baseline_top_id,
        "steered_top_token_id": steered_top_id,
        "baseline_top_token_text": decode_tokens(tokenizer, [baseline_top_id]),
        "steered_top_token_text": decode_tokens(tokenizer, [steered_top_id]),
        "generated_text_baseline": None,
        "generated_text_steered": None,
        "sycophancy_phrase_count_baseline": None,
        "sycophancy_phrase_count_steered": None,
        "refusal_phrase_count_baseline": None,
        "refusal_phrase_count_steered": None,
        "intervention_mode": intervention_mode,
        "used_chat_template": use_chat_template,
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
        stop_token_ids = resolve_stop_token_ids(request, model_dir, tokenizer)

        gen_baseline = executor.generate(
            input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
            max_new_tokens=max_new_tokens, sampling_config=sampling_config,
            stop_token_ids=stop_token_ids,
        )
        gen_steered = executor.generate(
            input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
            max_new_tokens=max_new_tokens, sampling_config=sampling_config,
            stop_token_ids=stop_token_ids, **intervention_kwargs,
        )

        result["generated_text_baseline"] = decode_tokens(tokenizer, gen_baseline.generated_token_ids)
        result["generated_text_steered"] = decode_tokens(tokenizer, gen_steered.generated_token_ids)

        # --- Sycophancy Reducer(2026-09-15追加): 迎合的な定型語彙の出現回数を機械的指標として
        # 併用する(漣さんとの確認事項)。measure_sycophancy=trueかつ生成テキストが得られた
        # 場合のみ計算する(Activation Steering Studio本体には影響しない、オプトイン)。
        if request.get("measure_sycophancy", False):
            from inference.sycophancy_utils import count_sycophantic_phrases  # noqa: PLC0415

            result["sycophancy_phrase_count_baseline"] = count_sycophantic_phrases(
                result["generated_text_baseline"]
            )
            result["sycophancy_phrase_count_steered"] = count_sycophantic_phrases(
                result["generated_text_steered"]
            )

        # --- Refusal Behavior Tuner(2026-09-22追加): 拒否定型句の出現回数(Sycophancy Reducerと
        # 同じ評価方式、漣さんとの確認事項)。measure_refusal=trueの場合のみ(オプトイン)。
        if request.get("measure_refusal", False):
            from inference.refusal_utils import count_refusal_phrases  # noqa: PLC0415

            result["refusal_phrase_count_baseline"] = count_refusal_phrases(result["generated_text_baseline"])
            result["refusal_phrase_count_steered"] = count_refusal_phrases(result["generated_text_steered"])

    return result


if __name__ == "__main__":
    main()
