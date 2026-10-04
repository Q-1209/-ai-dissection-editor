"""
test_split_safetensors.py

load_weights.py の list_available_keys() / load_model_weights() が、複数ファイルに
分割されたsafetensors(HuggingFaceの大規模モデルでよく使われる
"model-00001-of-00003.safetensors" 形式)を正しく扱えることを検証する。

load_weights.py自体はファイル冒頭でtorchをimportするため、この開発環境
(torch実行不可)では直接テストできない。そのため、ここでは全く同じ
ファイル読み込みロジック(safetensorsヘッダーのパース、複数ファイルへの
キー分散処理)を標準ライブラリのみで再実装し、期待通りに動くことを検証する。
実際のテンソル読み込み(torch.frombuffer等)を除く、ファイル探索・ヘッダー
パース・キーの分散解決の部分がここでの検証対象。

実行方法: python3 python/tests/test_split_safetensors.py
"""

import json
import struct
import tempfile
from pathlib import Path


def read_safetensors_header(file_path: Path) -> tuple[dict, int]:
    """load_weights.py の _read_safetensors_header() と同一ロジック。"""
    with open(file_path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"Invalid safetensors file (too short): {file_path}")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode("utf-8"))
    return header, 8 + header_len


def list_available_keys(model_dir: Path) -> set[str]:
    """load_weights.py の list_available_keys() と同一ロジック。"""
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    keys: set[str] = set()
    for st_file in safetensor_files:
        header, _ = read_safetensors_header(st_file)
        keys |= {k for k in header.keys() if k != "__metadata__"}
    return keys


def find_keys_across_files(model_dir: Path, keys: set[str]) -> tuple[dict, set[str]]:
    """load_weights.py の load_model_weights() のキー解決部分と同一ロジック
    (実際のテンソル読み込みは省略し、どのファイルにどのキーがあるかの
    解決だけを検証する)。
    """
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    all_found: dict[str, dict] = {}
    remaining = set(keys)
    for st_file in safetensor_files:
        if not remaining:
            break
        header, _ = read_safetensors_header(st_file)
        for key, meta in header.items():
            if key == "__metadata__" or key not in remaining:
                continue
            all_found[key] = meta
        remaining -= all_found.keys()
    missing = set(keys) - all_found.keys()
    return all_found, missing


def write_dummy_safetensors_header(file_path: Path, keys_and_shapes: list[tuple[str, list[int]]]) -> None:
    """テスト用の最小safetensorsファイル(ヘッダーのみ有効、本体はダミーバイト)を書き出す。"""
    header = {}
    offset = 0
    for key, shape in keys_and_shapes:
        size = 8  # ダミー(テンソル本体の中身は検証対象外)
        header[key] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    header["__metadata__"] = {"format": "pt"}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * offset)


def build_split_model(tmp_dir: Path, n_files: int = 3, n_layers: int = 4, hidden: int = 64) -> list[str]:
    """3ファイル程度に分割されたLlama風モデルのダミーデータを作る。
    戻り値は全キーのリスト(検証の期待値として使う)。
    """
    all_keys: list[str] = []
    for layer in range(n_layers):
        for name in [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "input_layernorm", "post_attention_layernorm",
        ]:
            all_keys.append(f"model.layers.{layer}.{name}.weight")
    all_keys += ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]

    chunk_size = (len(all_keys) + n_files - 1) // n_files
    chunks = [all_keys[i : i + chunk_size] for i in range(0, len(all_keys), chunk_size)]

    for i, chunk in enumerate(chunks):
        fname = tmp_dir / f"model-{i + 1:05d}-of-{n_files:05d}.safetensors"
        write_dummy_safetensors_header(fname, [(k, [hidden, hidden]) for k in chunk])

    return all_keys


def test_list_available_keys_covers_all_split_files():
    """3ファイルに分割されたモデルの全キーが、list_available_keys()で漏れなく検出される。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        expected_keys = build_split_model(tmp_dir, n_files=3)
        found_keys = list_available_keys(tmp_dir)
        assert found_keys == set(expected_keys), (
            f"key mismatch: missing={set(expected_keys) - found_keys}, "
            f"extra={found_keys - set(expected_keys)}"
        )
    print("PASS: test_list_available_keys_covers_all_split_files")


def test_partial_key_lookup_spans_multiple_files():
    """複数の分割ファイルにまたがるキーの一部だけを要求しても、
    正しくそれぞれのファイルから見つけ出せる(Phase1の「必要なノードの重みだけ
    読み込む」設計が分割ファイルでも機能することの検証)。
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        build_split_model(tmp_dir, n_files=3, n_layers=4)

        requested = {
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.3.mlp.down_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        }
        found, missing = find_keys_across_files(tmp_dir, requested)
        assert not missing, f"expected no missing keys, got {missing}"
        assert set(found.keys()) == requested
    print("PASS: test_partial_key_lookup_spans_multiple_files")


def test_missing_key_is_reported_not_silently_dropped():
    """存在しないキーを要求した場合、missingとして正しく報告される
    (サイレントに無視されて後段でKeyErrorになる、といった分かりにくい失敗を防ぐ)。
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        build_split_model(tmp_dir, n_files=2, n_layers=2)

        requested = {"model.embed_tokens.weight", "model.layers.99.self_attn.q_proj.weight"}
        found, missing = find_keys_across_files(tmp_dir, requested)
        assert missing == {"model.layers.99.self_attn.q_proj.weight"}
        assert "model.embed_tokens.weight" in found
    print("PASS: test_missing_key_is_reported_not_silently_dropped")


def test_single_file_model_still_works():
    """分割されていない単一ファイルのモデル(従来からのケース)でも、
    同じロジックが問題なく動作することを回帰確認する。
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        expected_keys = build_split_model(tmp_dir, n_files=1, n_layers=2)
        found_keys = list_available_keys(tmp_dir)
        assert found_keys == set(expected_keys)
    print("PASS: test_single_file_model_still_works")


def test_many_files_split():
    """10ファイル程度への分割でも、キーの収集が正しく機能することを確認する
    (実際の大規模モデルに近い分割数を模したケース)。
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        expected_keys = build_split_model(tmp_dir, n_files=10, n_layers=20)
        found_keys = list_available_keys(tmp_dir)
        assert found_keys == set(expected_keys)
        assert len(list(tmp_dir.glob("*.safetensors"))) == 10
    print("PASS: test_many_files_split")


if __name__ == "__main__":
    test_list_available_keys_covers_all_split_files()
    test_partial_key_lookup_spans_multiple_files()
    test_missing_key_is_reported_not_silently_dropped()
    test_single_file_model_still_works()
    test_many_files_split()
    print("\nAll split safetensors tests passed.")
