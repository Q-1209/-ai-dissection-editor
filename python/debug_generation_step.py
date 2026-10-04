"""
debug_generation_step.py

GUIで観測された「1トークン目から意味不明な出力になる」現象を切り分けるための
デバッグスクリプト。generate()と全く同じコードパスを辿りながら、
各段階の中間値・最終logitsのtop-5候補トークンを詳しく出力する。

使い方:
  python python/debug_generation_step.py "C:\\path\\to\\Qwen2.5-0.5B-Instruct"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from inference.graph_runner import LlamaGraphExecutor  # noqa: E402
from inference.graph_validation import GraphNode  # noqa: E402


def describe(name: str, t: torch.Tensor) -> None:
    t32 = t.float()
    print(
        f"{name}: shape={list(t.shape)} dtype={t.dtype} "
        f"mean={t32.mean().item():.6f} std={t32.std().item():.6f} "
        f"has_nan={bool(torch.isnan(t).any())}"
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_generation_step.py <model_dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1]).expanduser().resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    # GUIと全く同じ構成: 全24層をlayer_blockとして構築
    num_layers = config["num_hidden_layers"]
    nodes = [GraphNode(id="model.embed_tokens", role="embedding", layer_index=None)]
    for i in range(num_layers):
        nodes.append(GraphNode(id=f"model.layers.{i}", role="layer_block", layer_index=i))
    nodes.append(GraphNode(id="lm_head", role="lm_head", layer_index=None))

    node_ids = [n.id for n in nodes]
    edges = [(node_ids[i], node_ids[i + 1]) for i in range(len(node_ids) - 1)]
    entry_id, exit_id = node_ids[0], node_ids[-1]

    executor = LlamaGraphExecutor(model_dir, config)
    executor.load_weights_for_nodes(node_ids)

    # トークナイザーで実際にエンコード
    from inference.tokenizer_utils import decode_tokens, encode_text, load_tokenizer

    tokenizer = load_tokenizer(model_dir)
    prompt = "日本の首都は"
    prompt_ids = encode_text(tokenizer, prompt)
    print(f"Prompt: {prompt!r} -> token ids: {prompt_ids}")
    input_ids = torch.tensor([prompt_ids])

    # execute()を直接1回呼ぶ(generate()の1ステップ目と全く同じ呼び方)
    result = executor.execute(
        input_ids, nodes, edges, entry_node_id=entry_id, exit_node_id=exit_id,
        kv_cache={}, position_offset=0,
    )
    describe("final logits", result.logits)

    # 最後のトークン位置のlogitsからtop-5候補を見る
    last_logits = result.logits[0, -1, :]
    top5 = torch.topk(last_logits, 5)
    print("\nTop-5 next token candidates:")
    for score, token_id in zip(top5.values.tolist(), top5.indices.tolist()):
        token_text = decode_tokens(tokenizer, [token_id])
        print(f"  token_id={token_id:6d}  score={score:.4f}  text={token_text!r}")

    # 参考: embedding直後の統計(以前のデバッグと比較するため)
    embed_w = executor._w("model.embed_tokens.weight")
    x = torch.nn.functional.embedding(input_ids, embed_w)
    describe("\nafter embedding", x)


if __name__ == "__main__":
    main()
