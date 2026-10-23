#!/usr/bin/env python3
"""
run_noise_injection.py

Noise Injection(2026-10-24追加、長編動画M4「AIを酔っぱらわせろ」用) — 選んだノードの出力に
ランダムな雑音を足し、通常時と比べる。強さを上げるとどこで文章が崩れるか(呂律が回らなくなるか)、
どの層が雑音に弱いか(酔いやすいか)を見る。

2026-10-24、漣さんとの確認事項(詳細は inference/noise_utils.py):
- 観察タブの新パネル。Layer Ablation Wizardと同じ「層のチェックボックス→実行→通常時との比較」
- 強さ = その位置の内部状態の大きさに対する割合(0.1 = 10%)
- スイープ: 強さを段階的に上げる(strength_sweep)/同じ強さで各層を1つずつ試す(layer_sweep)の両方
- 乱数: 生成の1ステップごとに新しい雑音。seedを固定するので同じ設定なら同じ結果

Layer Ablation Wizard(run_layer_ablation.py)と同じ設計パターン: 既存のグラフ実行エンジンに
noise_strengths/noise_generator を渡すだけで実現している。

入力フォーマット(stdin):
{
  "model_dir": "...", "config": {...}, "nodes": [...], "edges": [...],
  "input_text": "こんにちは", "use_chat_template": false,
  "target_node_ids": ["model.layers.5"],   // 雑音を足すノード(1つ以上)
  "mode": "single" | "strength_sweep" | "layer_sweep",   // 省略時 "single"
  "strength": 0.3,                  // single・layer_sweepで使う強さ(0〜10)
  "strengths": [0, 0.1, 0.3, 1.0],  // strength_sweepの強さの並び(省略時 noise_utils.DEFAULT_STRENGTHS)
  "seed": 0,                        // 乱数の種(省略時0)
  "compare_generation": true, "max_new_tokens": 20,
  "temperature": 0.0, "top_k": null, "top_p": null, "repetition_penalty": 1.0, "no_repeat_ngram_size": null
}
  single: target_node_ids すべてに strength の雑音を同時に足して1回比較
  strength_sweep: target_node_ids すべてに、strengths の各値で雑音を足して比較
  layer_sweep: target_node_ids の各ノードに1つずつ、strength の雑音を足して比較

出力フォーマット(成功時):
{
  "success": true, "mode": "...", "seed": 0, "target_node_ids": [...], "strength": 0.3 | null,
  "input_token_ids": [...], "input_tokens_text": [...] | null,
  "baseline_top_token_text": "..." | null, "baseline_top_prob": 0.83,
  "generated_text_baseline": "..." | null,
  "points": [   // singleは1要素
    {"strength": 0.3, "node_id": "model.layers.5" | null, "kl_divergence": 1.23,
     "top_token_text": "..." | null, "top_prob": 0.41, "top_token_changed": true,
     "generated_text": "..." | null}
  ],
  "executed_order": [...]
}
layer_sweepでは各要素の node_id が雑音を足したノード、strength_sweep・singleでは null(全対象に同時)。

出力フォーマット(エラー時):
{"success": false, "error_type": "GraphValidationError" | "UnexpectedError" | "InvalidInput" | "TokenizerLoadError", "message": "..."}

使い方: python run_noise_injection.py < noise_input.json
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.graph_validation import GraphValidationError  # noqa: E402
from inference.i18n import tr  # noqa: E402
from inference.noise_utils import VALID_MODES, normalize_strengths, validate_strength  # noqa: E402
from inference.tokenizer_utils import TokenizerLoadError, resolve_stop_token_ids  # noqa: E402
from run_layer_ablation import _build_nodes_edges  # noqa: E402


def main() -> None:
    raw_input = sys.stdin.read()
    try:
        request = json.loads(raw_input)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error_type": "InvalidInput", "message": str(e)}))
        sys.exit(1)

    try:
        result = run_noise_injection(request)
        print(json.dumps(result, ensure_ascii=False))
    except (GraphValidationError, ValueError) as e:
        error_type = "GraphValidationError" if isinstance(e, GraphValidationError) else "InvalidInput"
        print(json.dumps({"success": False, "error_type": error_type, "message": str(e)}, ensure_ascii=False))
        sys.exit(0)
    except TokenizerLoadError as e:
        print(json.dumps({"success": False, "error_type": "TokenizerLoadError", "message": str(e)}, ensure_ascii=False))
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"success": False, "error_type": "UnexpectedError", "message": str(e)}, ensure_ascii=False))
        sys.exit(1)


def run_noise_injection(request: dict) -> dict:
    model_dir = Path(request["model_dir"])
    config = request["config"]
    nodes, edges, entry_node_id, exit_node_id = _build_nodes_edges(request)

    mode = request.get("mode", "single")
    if mode not in VALID_MODES:
        raise ValueError(tr(f"mode は {VALID_MODES} のいずれかを指定してください(指定値: {mode!r})。",
                            f"mode must be one of {VALID_MODES} (got {mode!r})."))
    target_node_ids = request.get("target_node_ids", [])
    if not target_node_ids:
        raise GraphValidationError(tr("雑音を足すノードを1つ以上指定してください。", "Pick at least one node to add noise to."))
    unmatched = set(target_node_ids) - {n.id for n in nodes}
    if unmatched:
        raise GraphValidationError(
            tr(f"雑音を足す対象として指定されたノードIDがグラフ内に存在しません: {sorted(unmatched)}",
               f"Node IDs to add noise to are not in the graph: {sorted(unmatched)}")
        )
    strength = validate_strength(request.get("strength", 0.3)) if mode != "strength_sweep" else None
    strengths = normalize_strengths(request.get("strengths")) if mode == "strength_sweep" else None
    seed = int(request.get("seed", 0))

    import torch  # noqa: PLC0415

    from inference.graph_runner import LlamaGraphExecutor  # noqa: PLC0415
    from inference.sampling import SamplingConfig  # noqa: PLC0415
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
            raise GraphValidationError(tr("入力テキストのトークナイズ結果が空でした。", "The input text tokenized to nothing."))
        input_tokens_text = [decode_tokens(tokenizer, [tid]) for tid in input_ids_list]
    else:
        input_ids_list = request.get("input_ids", [1, 2, 3, 4, 5])
    input_ids = torch.tensor([input_ids_list])

    def decode(ids: list[int]) -> str | None:
        if tokenizer is None:
            return None
        from inference.tokenizer_utils import decode_tokens  # noqa: PLC0415

        return decode_tokens(tokenizer, ids)

    compare_generation = bool(request.get("compare_generation", True))
    sampling_config = SamplingConfig(
        temperature=float(request.get("temperature", 0.0)),
        top_k=request.get("top_k"),
        top_p=request.get("top_p"),
        repetition_penalty=float(request.get("repetition_penalty", 1.0)),
        no_repeat_ngram_size=request.get("no_repeat_ngram_size"),
    )
    max_new_tokens = int(request.get("max_new_tokens", 20))
    stop_ids = resolve_stop_token_ids(request, model_dir, tokenizer)

    def forward(noise: dict[str, float] | None):
        gen = torch.Generator().manual_seed(seed)
        return executor.execute(
            input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
            kv_cache={}, position_offset=0, noise_strengths=noise, noise_generator=gen,
        )

    def generate(noise: dict[str, float] | None) -> str | None:
        if not compare_generation:
            return None
        gen = torch.Generator().manual_seed(seed)
        out = executor.generate(
            input_ids, nodes, edges, entry_node_id=entry_node_id, exit_node_id=exit_node_id,
            max_new_tokens=max_new_tokens, sampling_config=sampling_config, stop_token_ids=stop_ids,
            noise_strengths=noise, noise_generator=gen,
        )
        return decode(out.generated_token_ids)

    baseline = forward(None)
    base_logits = baseline.logits[0, -1, :]
    base_probs = torch.softmax(base_logits.float(), dim=-1)
    base_top = int(torch.argmax(base_logits).item())
    gen_baseline = generate(None)

    def measure(noise: dict[str, float], strength_value: float, node_id: str | None) -> dict:
        if strength_value == 0:  # 雑音なし=通常時と同じ(計算を省く)
            logits, text = base_logits, gen_baseline
        else:
            logits = forward(noise).logits[0, -1, :]
            text = generate(noise)
        probs = torch.softmax(logits.float(), dim=-1)
        top = int(torch.argmax(logits).item())
        return {
            "strength": strength_value,
            "node_id": node_id,
            "kl_divergence": compute_kl_divergence(base_logits, logits),
            "top_token_text": decode([top]),
            "top_prob": float(probs[top].item()),
            "top_token_changed": top != base_top,
            "generated_text": text,
        }

    if mode == "single":
        points = [measure({nid: strength for nid in target_node_ids}, strength, None)]
    elif mode == "strength_sweep":
        points = [measure({nid: s for nid in target_node_ids}, s, None) for s in strengths]
    else:
        points = [measure({nid: strength}, strength, nid) for nid in target_node_ids]

    return {
        "success": True,
        "mode": mode,
        "seed": seed,
        "target_node_ids": target_node_ids,
        "strength": strength,
        "input_token_ids": input_ids_list,
        "input_tokens_text": input_tokens_text,
        "baseline_top_token_text": decode([base_top]),
        "baseline_top_prob": float(base_probs[base_top].item()),
        "generated_text_baseline": gen_baseline,
        "points": points,
        "executed_order": baseline.node_order,
    }


if __name__ == "__main__":
    main()
