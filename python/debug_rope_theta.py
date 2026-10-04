"""
debug_rope_theta.py

GUI経由(execute_graph.py)で渡されるconfigに、実際にrope_thetaが正しく
含まれているかを確認するためのデバッグスクリプト。

使い方:
  python python/debug_rope_theta.py "C:\\path\\to\\Qwen2.5-0.5B-Instruct"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_rope_theta.py <model_dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1]).expanduser().resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    print("=== config.json から直接読んだ場合 ===")
    print(f"rope_theta in config: {config.get('rope_theta')}")
    print(f"config.get('rope_theta', 10000.0) = {config.get('rope_theta', 10000.0)}")

    # execute_graph.py が実際に行うのと同じ経路: JSON文字列化してから再パース
    # (Electron経由のIPC通信をシミュレート)
    json_roundtrip = json.loads(json.dumps(config))
    print("\n=== JSON文字列化 -> 再パース後 (Electron IPC相当) ===")
    print(f"rope_theta after roundtrip: {json_roundtrip.get('rope_theta')}")

    print("\n=== LlamaGraphExecutor に実際に渡した場合 ===")
    from inference.graph_runner import LlamaGraphExecutor

    executor = LlamaGraphExecutor(model_dir, config)
    print(f"executor.rope_base = {executor.rope_base}")
    print(f"executor.num_heads = {executor.num_heads}")
    print(f"executor.num_kv_heads = {executor.num_kv_heads}")
    print(f"executor.head_dim = {executor.head_dim}")

    if executor.rope_base != config.get("rope_theta"):
        print("\n!!! MISMATCH: executor.rope_base が config.json の rope_theta と一致しません !!!")
    else:
        print("\nOK: executor.rope_base は config.json の rope_theta と一致しています")


if __name__ == "__main__":
    main()
