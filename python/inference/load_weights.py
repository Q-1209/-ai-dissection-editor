"""
load_weights.py

safetensorsファイルから実際のテンソル値(重み本体)を読み込む。

Phase 0の inspect_model.py はヘッダーのみ読んで構造を把握するだけだったが、
Phase 1では実際に推論を実行するため、テンソルの実データが必要になる。

safetensorsのバイナリ形式:
  [8バイト: ヘッダー長(little-endian uint64)]
  [ヘッダー長ぶんのUTF-8 JSON: 各テンソルのdtype/shape/data_offsets]
  [テンソル本体のバイト列(ヘッダーのdata_offsetsが指す範囲)]

dtype文字列("F32", "F16", "BF16"等)をPyTorchのtorch.dtypeにマッピングし、
生バイト列をそのdtype・shapeのtorch.Tensorとして復元する。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import torch

# safetensorsのdtype文字列 -> torch.dtype のマッピング。
# BF16はnumpyにネイティブ型がないため、torch.frombufferで直接扱う。
DTYPE_MAP: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _read_safetensors_header(file_path: Path) -> tuple[dict[str, Any], int]:
    """ヘッダー(JSON)と、テンソル本体が始まるファイル内オフセットを返す。"""
    with open(file_path, "rb") as f:
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode("utf-8"))
    body_offset = 8 + header_len
    return header, body_offset


def load_tensors_from_file(
    file_path: Path, keys: set[str] | None = None
) -> dict[str, torch.Tensor]:
    """1つのsafetensorsファイルから、指定したキー(Noneなら全部)のテンソルを読み込む。

    data_offsetsを使ってファイル内の該当バイト範囲だけをシークして読むため、
    ファイル全体をメモリに載せる必要はない。
    """
    header, body_offset = _read_safetensors_header(file_path)
    result: dict[str, torch.Tensor] = {}

    with open(file_path, "rb") as f:
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if keys is not None and key not in keys:
                continue

            dtype_str = meta["dtype"]
            shape = meta["shape"]
            start, end = meta["data_offsets"]

            if dtype_str not in DTYPE_MAP:
                raise ValueError(f"Unsupported dtype '{dtype_str}' for key '{key}'")
            torch_dtype = DTYPE_MAP[dtype_str]

            f.seek(body_offset + start)
            raw_bytes = f.read(end - start)

            tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch_dtype)
            tensor = tensor.reshape(shape)
            result[key] = tensor

    return result


def list_available_keys(model_dir: Path) -> set[str]:
    """モデルディレクトリ内の全safetensorsファイルのヘッダーだけをスキャンし、
    実際に存在するテンソルキーの一覧を返す(テンソル本体は読まない、軽量な操作)。

    Qwen2のQ/K/V projectionバイアスのように、アーキテクチャによって存在したり
    しなかったりする重みを、実行前に「存在するか」判定するために使う。
    """
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    keys: set[str] = set()
    for st_file in safetensor_files:
        header, _ = _read_safetensors_header(st_file)
        keys |= {k for k in header.keys() if k != "__metadata__"}
    return keys


def load_model_weights(
    model_dir: Path, keys: set[str] | None = None
) -> dict[str, torch.Tensor]:
    """モデルディレクトリ内の全safetensorsファイル(分割保存含む)から、
    指定したキー(Noneなら全キー)の重みをまとめて読み込む。

    keysを絞ることで、グラフ実行に必要なノードの重みだけを読み込み、
    メモリ使用量を抑えられる(Phase1では「一部のレイヤーだけ使う」構成も
    想定されるため)。
    """
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    all_weights: dict[str, torch.Tensor] = {}
    remaining = set(keys) if keys is not None else None

    for st_file in safetensor_files:
        if remaining is not None and not remaining:
            break  # 必要なキーを全部読み終えたら残りのファイルは開かない
        loaded = load_tensors_from_file(st_file, remaining)
        all_weights.update(loaded)
        if remaining is not None:
            remaining -= loaded.keys()

    if keys is not None:
        missing = set(keys) - all_weights.keys()
        if missing:
            raise KeyError(f"Requested keys not found in model weights: {missing}")

    return all_weights
