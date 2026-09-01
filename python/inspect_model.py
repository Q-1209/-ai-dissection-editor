#!/usr/bin/env python3
"""
inspect_model.py

HuggingFace形式のTransformer系モデルディレクトリ(config.json + *.safetensors)を読み込み、
ノードグラフ用のJSONを標準出力に1行で出力する。

設計方針:
- safetensorsファイルの「ヘッダー」だけを読む(重みテンソル本体はロードしない)。
  ヘッダーには各パラメータのキー名・shape・dtypeがJSON形式で入っているため、
  数GBのモデルでも一瞬で構造を取得できる。
- state_dictのキー名(例: "model.layers.0.self_attn.q_proj.weight")を正規表現で
  パースし、層番号と役割(attention/mlp/norm/embedding等)を推定してラベル付けする。
- 対応アーキテクチャはPhase 0時点でLlama系命名規則(Llama/Mistral/Qwen等、多くが
  同じ命名パターンを共有)に絞る。将来的にモデルごとのパーサーを追加していく。

使い方:
    python inspect_model.py /path/to/model_dir
"""

import io
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

# Windows環境ではデフォルトの標準出力エンコーディングがCP932(Shift-JIS系)になり、
# ラベルに含まれる em dash(—)などの記号でUnicodeEncodeErrorが発生する。
# stdout/stderrをUTF-8に固定してこれを回避する。
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


class InspectionError(Exception):
    pass


def read_safetensors_header(file_path: Path) -> dict[str, Any]:
    """safetensorsファイルの先頭からヘッダー(JSON)だけを読み取る。
    フォーマット: 先頭8byteがヘッダー長(little-endian uint64)、続いてそのbyte数分のUTF-8 JSON。
    """
    with open(file_path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise InspectionError(f"Invalid safetensors file (too short): {file_path}")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = f.read(header_len)
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise InspectionError(f"Failed to parse safetensors header in {file_path}: {e}")
    return header


def collect_all_tensor_entries(model_dir: Path) -> dict[str, dict[str, Any]]:
    """モデルディレクトリ内の全safetensorsファイルからテンソル情報を集約する。
    分割保存(model-00001-of-00003.safetensors等)にも対応。
    """
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise InspectionError(
            f"No .safetensors files found in {model_dir}. "
            "現時点ではsafetensors形式のみ対応しています(.binのpickle形式は未対応)。"
        )

    all_entries: dict[str, dict[str, Any]] = {}
    for st_file in safetensor_files:
        header = read_safetensors_header(st_file)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            all_entries[key] = meta
    return all_entries


# state_dictキー名 -> 役割ラベルへのマッピングルール。
# Llama/Mistral/Qwen等、多くのTransformer系OSSモデルはこの命名規則に概ね従う。
ROLE_PATTERNS: list[tuple[str, str, str]] = [
    # (正規表現, role, 表示ラベルのテンプレート)
    (r"embed_tokens", "embedding", "Token Embedding"),
    (r"layers\.(\d+)\.self_attn\.q_proj", "attention", "Attention (Q projection) — Layer {n}"),
    (r"layers\.(\d+)\.self_attn\.k_proj", "attention", "Attention (K projection) — Layer {n}"),
    (r"layers\.(\d+)\.self_attn\.v_proj", "attention", "Attention (V projection) — Layer {n}"),
    (r"layers\.(\d+)\.self_attn\.o_proj", "attention", "Attention (Output projection) — Layer {n}"),
    (r"layers\.(\d+)\.mlp\.gate_proj", "mlp", "MLP (Gate) — Layer {n}"),
    (r"layers\.(\d+)\.mlp\.up_proj", "mlp", "MLP (Up projection) — Layer {n}"),
    (r"layers\.(\d+)\.mlp\.down_proj", "mlp", "MLP (Down projection) — Layer {n}"),
    (r"layers\.(\d+)\.input_layernorm", "norm", "Input LayerNorm — Layer {n}"),
    (r"layers\.(\d+)\.post_attention_layernorm", "norm", "Post-Attention LayerNorm — Layer {n}"),
    # Gemma2/Gemma3はattention前後に加え、MLP前後にも正規化層を持つ(計4つ)。
    # Llama/Mistral/Qwen等には存在しないキーなので、該当モデルでのみ追加ノードとして現れる。
    (r"layers\.(\d+)\.pre_feedforward_layernorm", "norm", "Pre-Feedforward LayerNorm — Layer {n}"),
    (r"layers\.(\d+)\.post_feedforward_layernorm", "norm", "Post-Feedforward LayerNorm — Layer {n}"),
    (r"^model\.norm\b", "norm", "Final Norm"),
    (r"^lm_head", "lm_head", "LM Head"),
]


def classify_key(key: str) -> tuple[str, str, int | None]:
    """state_dictキーを (role, label, layer_index) に分類する。
    どのパターンにもマッチしない場合は role="other" として扱う。
    """
    for pattern, role, label_template in ROLE_PATTERNS:
        m = re.search(pattern, key)
        if m:
            layer_index = int(m.group(1)) if m.groups() else None
            label = label_template.format(n=layer_index) if layer_index is not None else label_template
            return role, label, layer_index
    return "other", key, None


def build_node_graph(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise InspectionError(f"config.json not found in {model_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type", "unknown")
    num_layers = config.get("num_hidden_layers", config.get("n_layer"))
    hidden_size = config.get("hidden_size", config.get("n_embd"))

    entries = collect_all_tensor_entries(model_dir)

    nodes: list[dict[str, Any]] = []
    errors: list[str] = []

    # .weight / .bias の重複を避けるため、weightのみをノード化の代表とする
    seen_base_keys: set[str] = set()

    for key, meta in sorted(entries.items()):
        base_key = key.rsplit(".", 1)[0] if key.endswith((".weight", ".bias")) else key
        if base_key in seen_base_keys:
            continue
        seen_base_keys.add(base_key)

        role, label, layer_index = classify_key(key)
        shape = meta.get("shape")

        nodes.append(
            {
                "id": base_key,
                "label": label,
                "role": role,
                "layer_index": layer_index,
                "shape": shape,
                "param_key": base_key,
            }
        )

    # レイヤー内: Q/K/V -> O -> MLP という素朴な逐次接続でエッジを生成(Phase 0の可視化用)
    edges: list[dict[str, str]] = []
    nodes_by_layer: dict[int, list[dict[str, Any]]] = {}
    for n in nodes:
        if n["layer_index"] is not None:
            nodes_by_layer.setdefault(n["layer_index"], []).append(n)

    order_hint = [
        "input_layernorm",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "post_feedforward_layernorm",
    ]

    def sort_key(n: dict[str, Any]) -> int:
        for i, frag in enumerate(order_hint):
            if frag in n["id"]:
                return i
        return len(order_hint)

    for layer_idx, layer_nodes in sorted(nodes_by_layer.items()):
        ordered = sorted(layer_nodes, key=sort_key)
        for a, b in zip(ordered, ordered[1:]):
            edges.append({"id": f"{a['id']}->{b['id']}", "source": a["id"], "target": b["id"]})

    return {
        "model_type": model_type,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "nodes": nodes,
        "edges": edges,
        "errors": errors,
    }


def main() -> None:
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: inspect_model.py <model_dir>"}), file=sys.stderr)
        sys.exit(1)

    model_dir = Path(sys.argv[1]).expanduser().resolve()
    if not model_dir.is_dir():
        print(json.dumps({"error": f"Not a directory: {model_dir}"}), file=sys.stderr)
        sys.exit(1)

    try:
        result = build_node_graph(model_dir)
    except InspectionError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 — 予期しないエラーもJSONで返す
        print(json.dumps({"error": f"Unexpected error: {e}"}), file=sys.stderr)
        sys.exit(1)

    # 標準出力にはJSONのみを出す(Electron側でパースするため)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
