"""
tiny_model.py — テスト用の極小モデルを一時フォルダに生成する(2026-09-22追加)。

実モデル(数百MB〜)なしで、PyTorch実装の実行経路(グラフ実行・生成・ステアリング・方向除去・
チャットテンプレート・停止トークン)をend-to-endに検証するためのもの。重みは乱数なので出力の
「意味」は検証できないが、「最後まで例外なく動く」「数式どおりの性質を満たす」ことは検証できる。

- Qwen2形式(Q/K/Vバイアスあり、GQA: KVヘッド1、tie_word_embeddings)、hidden=16・3層
- safetensorsは外部ライブラリを使わず自前で書き出す(load_weights.pyが読む形式と同じ)
- トークナイザーは1文字=1トークンの小さなWordLevel(tokenizersライブラリ)+ChatML特殊トークン
- tokenizer_config.jsonにChatML形式のchat_templateを入れる(with_chat_template=Falseで外せる)
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
    "{% endif %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

IM_END_ID = 3


def _build_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors  # noqa: PLC0415

    vocab = {"[UNK]": 0, "<bos>": 1, "<|im_start|>": 2, "<|im_end|>": IM_END_ID}
    for ch in "abcdefghijklmnopqrstuvwxyz \n":
        vocab[ch] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens(["<bos>", "<|im_start|>", "<|im_end|>"])
    tok.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 1)])
    return tok


def _write_safetensors(path: Path, tensors: dict) -> None:
    """float32のtorchテンソル辞書を、safetensors形式(8byteヘッダー長+JSONヘッダー+本体)で書き出す。"""
    header: dict = {}
    blobs: list[bytes] = []
    offset = 0
    for name, t in tensors.items():
        data = t.detach().contiguous().to("cpu").float().numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    header_bytes = json.dumps(header).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)


def build_tiny_model(model_dir: Path, *, with_chat_template: bool = True, seed: int = 0) -> dict:
    """model_dirに極小モデル一式を書き出し、config辞書を返す。"""
    import torch  # noqa: PLC0415

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tok = _build_tokenizer()
    tok.save(str(model_dir / "tokenizer.json"))

    vocab_size = tok.get_vocab_size()
    hidden, heads, kv_heads, inter, layers = 16, 2, 1, 32, 3
    head_dim = hidden // heads
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g) * 0.3

    w = {"model.embed_tokens.weight": r(vocab_size, hidden), "model.norm.weight": torch.ones(hidden)}
    for i in range(layers):
        b = f"model.layers.{i}"
        w.update({
            f"{b}.input_layernorm.weight": torch.ones(hidden),
            f"{b}.post_attention_layernorm.weight": torch.ones(hidden),
            f"{b}.self_attn.q_proj.weight": r(hidden, hidden),
            f"{b}.self_attn.q_proj.bias": r(hidden),
            f"{b}.self_attn.k_proj.weight": r(kv_heads * head_dim, hidden),
            f"{b}.self_attn.k_proj.bias": r(kv_heads * head_dim),
            f"{b}.self_attn.v_proj.weight": r(kv_heads * head_dim, hidden),
            f"{b}.self_attn.v_proj.bias": r(kv_heads * head_dim),
            f"{b}.self_attn.o_proj.weight": r(hidden, hidden),
            f"{b}.mlp.gate_proj.weight": r(inter, hidden),
            f"{b}.mlp.up_proj.weight": r(inter, hidden),
            f"{b}.mlp.down_proj.weight": r(hidden, inter),
        })
    _write_safetensors(model_dir / "model.safetensors", w)

    config = {
        "model_type": "qwen2", "hidden_size": hidden, "num_attention_heads": heads,
        "num_key_value_heads": kv_heads, "intermediate_size": inter, "num_hidden_layers": layers,
        "vocab_size": vocab_size, "tie_word_embeddings": True, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tok_config: dict = {"eos_token": "<|im_end|>"}
    if with_chat_template:
        tok_config["chat_template"] = CHATML_TEMPLATE
    (model_dir / "tokenizer_config.json").write_text(json.dumps(tok_config), encoding="utf-8")
    (model_dir / "generation_config.json").write_text(json.dumps({"eos_token_id": [IM_END_ID]}), encoding="utf-8")
    return config


def default_graph_request(model_dir: Path, config: dict) -> dict:
    """各エントリポイントに渡すリクエストの共通部分(埋め込み→layer_block×N→lm_headのグラフ)。"""
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from inference.battle_graph import build_default_execution_graph  # noqa: PLC0415

    nodes, edges = build_default_execution_graph(config["num_hidden_layers"])
    return {
        "model_dir": str(model_dir),
        "config": config,
        "nodes": [{"id": n.id, "role": n.role, "layer_index": n.layer_index} for n in nodes],
        "edges": [{"source": a, "target": b} for a, b in edges],
    }
