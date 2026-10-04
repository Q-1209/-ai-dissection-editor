"""
debug_output_values.py

「実行成功はしたがmean/stdが0.0000と表示される」現象を切り分けるための
デバッグスクリプト。生のlogits値・各層通過後の中間値の統計を詳しく出力する。

【修正履歴】この問題は、Attention/MLPブロック実行時に正規化層(RMSNorm)と
残差接続(residual connection)が抜けていたことが原因と判明し、graph_runner.py
側で修正済み。このスクリプトは修正後の値が正常な範囲に収まることを確認するために
残している。

使い方:
  python python/debug_output_values.py "C:\\path\\to\\Qwen2.5-0.5B-Instruct"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from inference.graph_runner import LlamaGraphExecutor  # noqa: E402
from inference.graph_validation import GraphNode, topological_sort  # noqa: E402


def describe(name: str, t: torch.Tensor) -> None:
    t32 = t.float()
    print(
        f"{name}: shape={list(t.shape)} dtype={t.dtype} "
        f"mean={t32.mean().item():.10f} std={t32.std().item():.10f} "
        f"min={t32.min().item():.6f} max={t32.max().item():.6f} "
        f"has_nan={bool(torch.isnan(t).any())} has_inf={bool(torch.isinf(t).any())}"
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_output_values.py <model_dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1]).expanduser().resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    nodes = [
        GraphNode(id="model.layers.0.self_attn", role="attention_block", layer_index=0),
        GraphNode(id="model.layers.0.mlp", role="mlp_block", layer_index=0),
        GraphNode(id="model.layers.1.self_attn", role="attention_block", layer_index=1),
        GraphNode(id="model.layers.1.mlp", role="mlp_block", layer_index=1),
    ]
    edges = [
        ("model.layers.0.self_attn", "model.layers.0.mlp"),
        ("model.layers.0.mlp", "model.layers.1.self_attn"),
        ("model.layers.1.self_attn", "model.layers.1.mlp"),
    ]
    order_ids = topological_sort(nodes, edges)
    node_by_id = {n.id: n for n in nodes}
    ordered_nodes = [node_by_id[nid] for nid in order_ids]

    executor = LlamaGraphExecutor(model_dir, config)
    executor.load_weights_for_nodes([n.id for n in ordered_nodes])

    print(f"tie_word_embeddings = {executor.tie_word_embeddings}\n")

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    embed_w = executor._w("model.embed_tokens.weight")
    x = torch.nn.functional.embedding(input_ids, embed_w)
    describe("after embedding", x)

    from inference.layers import build_rope_cache
    cos, sin = build_rope_cache(executor.head_dim, 2048, base=executor.rope_base, dtype=x.dtype)

    x = executor.run_attention_block(x, "model.layers.0", cos, sin)
    describe("after layer0 attention (with norm+residual)", x)

    x = executor.run_mlp_block(x, "model.layers.0")
    describe("after layer0 mlp (with norm+residual)", x)

    x = executor.run_attention_block(x, "model.layers.1", cos, sin)
    describe("after layer1 attention (with norm+residual)", x)

    x = executor.run_mlp_block(x, "model.layers.1")
    describe("after layer1 mlp (with norm+residual)", x)

    print(f"\nExpected: 各段階でstdが極端に減衰せず、概ね同程度の桁数を保つはず")
    print(f"(残差接続により、値が完全に消失することはなくなる)")


if __name__ == "__main__":
    main()

