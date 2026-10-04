"""
run_phase1_smoke_test.py

Phase 1推論エンジンの動作確認用スモークテスト。手元にPyTorchがインストール
されたWindows/Mac/Linux環境で実行することを想定している
(このプロジェクトの開発環境ではGPU依存必須のtorchビルドしか入手できず、
実行確認ができなかったため)。

事前準備:
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install tokenizers safetensors

実行方法:
  python python/run_phase1_smoke_test.py /path/to/model_dir

このスクリプトが検証する内容:
1. 実際のsafetensorsから重みが読み込めること
2. オリジナルのモデル構造通りの順序でグラフを組み、推論が最後まで通ること
   (エラーの有無のみ確認。出力の意味的な妥当性は目視確認)
3. ユーザーが層の順序を入れ替えた場合も、エラーなく計算が完走すること
   (Phase1のコア機能: 自由な繋ぎ替え)
4. 【回帰テスト】エッジを削除してグラフを分断した場合、その分断が無視されず
   正しくエラーとして検出されること(過去に発覚した重大バグの再発防止)
5. KVキャッシュを使った複数トークンの逐次生成がエラーなく完走すること
6. 実際のトークナイザーを使ったテキスト入力 -> 生成 -> テキスト出力が動くこと
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from inference.graph_runner import LlamaGraphExecutor  # noqa: E402
from inference.graph_validation import GraphNode, GraphValidationError  # noqa: E402


def build_graph(num_layers_to_use: int = 2) -> list[GraphNode]:
    """embedding -> layer群 -> lm_head という完全なノードリストを作る。

    全層を使うと大きなモデルでは重みロードに時間がかかるため、
    スモークテストでは最初の数層だけを使う設定にしている。
    """
    nodes: list[GraphNode] = [GraphNode(id="model.embed_tokens", role="embedding", layer_index=None)]
    for i in range(num_layers_to_use):
        nodes.append(GraphNode(id=f"model.layers.{i}.self_attn", role="attention_block", layer_index=i))
        nodes.append(GraphNode(id=f"model.layers.{i}.mlp", role="mlp_block", layer_index=i))
    nodes.append(GraphNode(id="lm_head", role="lm_head", layer_index=None))
    return nodes


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_phase1_smoke_test.py <model_dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1]).expanduser().resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    print(f"Model: {config.get('model_type')}, hidden_size={config['hidden_size']}")

    nodes = build_graph(num_layers_to_use=2)
    node_ids = [n.id for n in nodes]
    entry_id, exit_id = node_ids[0], node_ids[-1]

    executor = LlamaGraphExecutor(model_dir, config)
    executor.load_weights_for_nodes(node_ids)

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    # --- テスト1: オリジナル順序での1ステップ実行 ---
    print("\n--- Test 1: original layer order (single forward pass) ---")
    edges = [(node_ids[i], node_ids[i + 1]) for i in range(len(node_ids) - 1)]
    result = executor.execute(
        input_ids, nodes, edges, entry_node_id=entry_id, exit_node_id=exit_id,
        kv_cache={}, position_offset=0,
    )
    print(f"Output shape: {result.logits.shape}")
    print(f"Executed order: {result.node_order}")
    assert not torch.isnan(result.logits).any(), "Output contains NaN!"
    print("PASS: original order executed without error, no NaN in output")

    # --- テスト2: ユーザーによる層の入れ替え ---
    print("\n--- Test 2: user-reordered layers (Phase1 core feature) ---")
    reordered_edges = [
        (node_ids[0], node_ids[3]),  # embed -> layer1.attn
        (node_ids[3], node_ids[1]),  # layer1.attn -> layer0.attn
        (node_ids[1], node_ids[2]),  # layer0.attn -> layer0.mlp
        (node_ids[2], node_ids[4]),  # layer0.mlp -> layer1.mlp
        (node_ids[4], node_ids[5]),  # layer1.mlp -> lm_head
    ]
    result2 = executor.execute(
        input_ids, nodes, reordered_edges, entry_node_id=entry_id, exit_node_id=exit_id,
        kv_cache={}, position_offset=0,
    )
    print(f"Reordered execution order: {result2.node_order}")
    assert not torch.isnan(result2.logits).any(), "Output contains NaN!"
    print("PASS: reordered graph executed without error, no NaN in output")

    # --- テスト3(回帰テスト): エッジを削除してグラフを分断した場合、正しく拒否されること ---
    print("\n--- Test 3: disconnected edge must be rejected (regression test) ---")
    disconnected_edges = [
        (node_ids[0], node_ids[1]),  # embed -> layer0.attn
        (node_ids[2], node_ids[3]),  # layer0.mlp -> layer1.attn
        (node_ids[3], node_ids[4]),  # layer1.attn -> layer1.mlp
        (node_ids[4], node_ids[5]),  # layer1.mlp -> lm_head
    ]
    try:
        executor.execute(
            input_ids, nodes, disconnected_edges, entry_node_id=entry_id, exit_node_id=exit_id,
            kv_cache={}, position_offset=0,
        )
        print("FAIL: disconnected graph should have raised GraphValidationError, but it did not!")
        sys.exit(1)
    except GraphValidationError as e:
        print(f"PASS: disconnected graph correctly rejected: {e}")

    # --- テスト3b: 合流(複数入力)のグラフが実行できること ---
    print("\n--- Test 3b: merge (multiple inputs) graph execution ---")
    # layer0.attnとlayer1.attnの出力をlayer2.attnで合流させ、そのままlm_headに繋ぐ
    # 最小構成のグラフを組み立てる(build_graph()の全ノードは使わない —
    # 使わないノードを混ぜると「孤立ノード」として誤検出されるため、
    # このテストで実際に使うノードだけを明示的に列挙する)。
    from inference.graph_validation import GraphNode as _GraphNode  # noqa: PLC0415

    merge_nodes = [
        _GraphNode(id="model.embed_tokens", role="embedding", layer_index=None),
        _GraphNode(id="model.layers.0.self_attn", role="attention_block", layer_index=0),
        _GraphNode(id="model.layers.1.self_attn", role="attention_block", layer_index=1),
        _GraphNode(id="model.layers.2.self_attn", role="attention_block", layer_index=2),
        _GraphNode(id="lm_head", role="lm_head", layer_index=None),
    ]
    merge_node_ids = [n.id for n in merge_nodes]
    merge_entry_id, merge_exit_id = merge_node_ids[0], merge_node_ids[-1]
    merge_executor = LlamaGraphExecutor(model_dir, config)
    merge_executor.load_weights_for_nodes(merge_node_ids)

    merge_edges = [
        ("model.embed_tokens", "model.layers.0.self_attn"),  # embed -> l0.attn
        ("model.embed_tokens", "model.layers.1.self_attn"),  # embed -> l1.attn (l0.attnと並列)
        ("model.layers.0.self_attn", "model.layers.2.self_attn"),  # l0.attn -> l2.attn (合流元1)
        ("model.layers.1.self_attn", "model.layers.2.self_attn"),  # l1.attn -> l2.attn (合流元2)
        ("model.layers.2.self_attn", "lm_head"),  # l2.attn -> lm_head
    ]
    merge_result = merge_executor.execute(
        input_ids, merge_nodes, merge_edges, entry_node_id=merge_entry_id, exit_node_id=merge_exit_id,
        kv_cache={}, position_offset=0,
    )
    print(f"Output shape: {merge_result.logits.shape}")
    print(f"Executed order: {merge_result.node_order}")
    assert not torch.isnan(merge_result.logits).any(), "Output contains NaN!"
    print("PASS: merge graph (default merge_mode='add') executed without error, no NaN in output")

    # merge_modeを明示的に"average"に設定したノードでも実行できることを確認
    merge_nodes_avg = [
        _GraphNode(id=n.id, role=n.role, layer_index=n.layer_index, merge_mode="average")
        if n.id == "model.layers.2.self_attn"
        else n
        for n in merge_nodes
    ]
    merge_result_avg = merge_executor.execute(
        input_ids, merge_nodes_avg, merge_edges, entry_node_id=merge_entry_id, exit_node_id=merge_exit_id,
        kv_cache={}, position_offset=0,
    )
    assert not torch.isnan(merge_result_avg.logits).any(), "Output contains NaN!"
    # add と average は異なる結果になるはず(合成方法が違うため)
    assert not torch.allclose(merge_result.logits, merge_result_avg.logits), (
        "merge_mode='add' と 'average' で同じ結果になってしまっている(合成ロジックが効いていない可能性)"
    )
    print("PASS: merge graph (merge_mode='average') executed without error and differs from 'add'")

    # --- テスト4: KVキャッシュを使った複数トークンの逐次生成 ---
    print("\n--- Test 4: multi-token generation with KV cache ---")
    gen_result = executor.generate(
        input_ids, nodes, edges, entry_node_id=entry_id, exit_node_id=exit_id, max_new_tokens=8,
    )
    print(f"Generated token ids: {gen_result.generated_token_ids}")
    assert len(gen_result.generated_token_ids) == 8, "expected 8 generated tokens"
    print("PASS: multi-token generation completed without error")

    # --- テスト4b: サンプリング設定(temperature/top-k/top-p/repetition penalty)付き生成 ---
    print("\n--- Test 4b: generation with sampling config ---")
    from inference.sampling import SamplingConfig

    sampling_config = SamplingConfig(temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.2)
    gen_result_sampled = executor.generate(
        input_ids, nodes, edges, entry_node_id=entry_id, exit_node_id=exit_id,
        max_new_tokens=8, sampling_config=sampling_config,
    )
    print(f"Generated token ids (sampled): {gen_result_sampled.generated_token_ids}")
    assert len(gen_result_sampled.generated_token_ids) == 8, "expected 8 generated tokens"
    print("PASS: sampling-based generation completed without error")

    # --- テスト4c: no_repeat_ngram_size付きの貪欲デコード ---
    # (このテスト用の縮小構成では同じトークンの完全な繰り返しに陥りやすいため、
    #  no_repeat_ngram_size=2(bigram禁止)が効いていれば同じトークンが2回連続する
    #  ことはあっても、3回以上連続することは無いはずと期待できる)
    print("\n--- Test 4c: greedy decoding with no_repeat_ngram_size ---")
    no_repeat_config = SamplingConfig(temperature=0.0, no_repeat_ngram_size=2)
    gen_result_no_repeat = executor.generate(
        input_ids, nodes, edges, entry_node_id=entry_id, exit_node_id=exit_id,
        max_new_tokens=8, sampling_config=no_repeat_config,
    )
    print(f"Generated token ids (no_repeat_ngram_size=2): {gen_result_no_repeat.generated_token_ids}")
    assert len(gen_result_no_repeat.generated_token_ids) == 8, "expected 8 generated tokens"
    print("PASS: no_repeat_ngram generation completed without error")

    # --- テスト5: 実際のトークナイザーでテキスト入出力 ---
    print("\n--- Test 5: real tokenizer text in/out ---")
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"SKIP: tokenizer.json not found at {tokenizer_path}")
    else:
        from inference.tokenizer_utils import decode_tokens, encode_text, load_tokenizer

        tokenizer = load_tokenizer(model_dir)
        prompt = "こんにちは"
        prompt_ids = encode_text(tokenizer, prompt)
        print(f"Prompt: {prompt!r} -> token ids: {prompt_ids}")

        prompt_tensor = torch.tensor([prompt_ids])
        text_gen_result = executor.generate(
            prompt_tensor, nodes, edges, entry_node_id=entry_id, exit_node_id=exit_id,
            max_new_tokens=10,
        )
        generated_text = decode_tokens(tokenizer, text_gen_result.generated_token_ids)
        print(f"Generated token ids: {text_gen_result.generated_token_ids}")
        print(f"Generated text: {generated_text!r}")
        print(
            "NOTE: このグラフは先頭2層のみを使ったテスト用の小規模構成なので、"
            "生成されるテキストが意味を成すとは限りません(構造検証が目的)。"
        )
        print("PASS: tokenizer encode/decode + generation completed without error")

    # --- テスト6: 内部状態可視化(Phase2)のcapture_internals機能 ---
    print("\n--- Test 6: capture_internals (Phase2 internal state visualization) ---")

    # 6a: capture_internals=False(デフォルト)なら internals は空のまま
    result_no_capture = executor.execute(
        input_ids, nodes, edges, entry_node_id=entry_id, exit_node_id=exit_id,
        kv_cache={}, position_offset=0, capture_internals=False,
    )
    assert result_no_capture.internals == {}, (
        f"capture_internals=False なのに internals が空でない: {list(result_no_capture.internals.keys())}"
    )
    print("PASS: capture_internals=False leaves internals empty (no overhead when not requested)")

    # 6b: capture_internals=True で、Attention重み・活性化値の両方が記録される
    result_capture = executor.execute(
        input_ids, nodes, edges, entry_node_id=entry_id, exit_node_id=exit_id,
        kv_cache={}, position_offset=0, capture_internals=True,
    )
    seq_len = input_ids.shape[1]
    attn_key = "model.layers.0.self_attn.attn_weights"
    output_key = "model.layers.0.self_attn.output"
    mlp_output_key = "model.layers.0.mlp.output"

    assert attn_key in result_capture.internals, f"expected {attn_key} in internals"
    assert output_key in result_capture.internals, f"expected {output_key} in internals"
    assert mlp_output_key in result_capture.internals, f"expected {mlp_output_key} in internals"

    attn_w = result_capture.internals[attn_key]
    print(f"Attention weights shape: {attn_w.shape} (expected [1, num_heads, {seq_len}, {seq_len}])")
    assert attn_w.shape[0] == 1  # batch
    assert attn_w.shape[2] == seq_len
    assert attn_w.shape[3] == seq_len

    # Attention重みは各クエリ位置について確率分布(合計1)になっているはず(softmax後のため)。
    # 比較前にfloat32へ変換し、BFloat16(仮数部7bit、10進数で2〜3桁程度の精度)特有の
    # 丸め誤差を考慮してatolを緩めに設定する(BFloat16のまま1e-3で比較すると、
    # 正常な丸め誤差でも失敗することがある)。
    row_sums = attn_w.float().sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2), (
        f"attention weights should sum to 1 per query position, got sums: {row_sums}"
    )
    print("PASS: captured attention weights are a valid probability distribution (rows sum to 1)")

    block_output = result_capture.internals[output_key]
    print(f"Block output shape: {block_output.shape}, mean={block_output.float().mean():.4f}")
    assert not torch.isnan(block_output).any(), "captured block output contains NaN!"
    print("PASS: captured block output (activation) contains no NaN")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
