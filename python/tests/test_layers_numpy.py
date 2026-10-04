"""
test_layers_numpy.py

この開発環境ではGPU依存必須のPyTorchビルドしか入手できず、実行検証ができない
(詳細はプロジェクトのメモ参照)。そのため、layers.py と全く同じ数式を numpy で
独立に再実装し、既知の性質(RoPEの回転不変性、Attentionの出力形状、softmaxの
正規化など)を検証することでロジックの正しさを担保する。

実際のtorch実行によるエンドツーエンド検証は、ユーザーの手元環境(Windows、
torch動作確認済み)で行う。

実行方法: python3 python/tests/test_layers_numpy.py
"""

import numpy as np


def rms_norm_np(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    variance = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    x_normed = x.astype(np.float32) / np.sqrt(variance + eps)
    return weight * x_normed


def build_rope_cache_np(head_dim: int, max_position: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(max_position, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def rotate_half_np(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope_np(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = (q * cos) + (rotate_half_np(q) * sin)
    k_rot = (k * cos) + (rotate_half_np(k) * sin)
    return q_rot, k_rot


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def repeat_kv_np(x: np.ndarray, n_rep: int) -> np.ndarray:
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = np.broadcast_to(
        x[:, :, None, :, :], (batch, num_kv_heads, n_rep, seq_len, head_dim)
    )
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def attention_np(q, k, v, num_heads, num_kv_heads, causal=True, past_k=None, past_v=None):
    """layers.pyのattention()と全く同じロジックのnumpy版。KVキャッシュ対応。

    past_k, past_v: [batch, num_kv_heads, past_len, head_dim] または None
    戻り値: (output, attn_weights, updated_k_cache, updated_v_cache)
    """
    batch, seq_len, _ = q.shape
    head_dim = q.shape[-1] // num_heads

    q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k_new = k.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v_new = v.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    if past_k is not None and past_v is not None:
        k_full = np.concatenate([past_k, k_new], axis=2)
        v_full = np.concatenate([past_v, v_new], axis=2)
    else:
        k_full = k_new
        v_full = v_new

    updated_k_cache = k_full
    updated_v_cache = v_full

    n_rep = num_heads // num_kv_heads
    k_full = repeat_kv_np(k_full, n_rep)
    v_full = repeat_kv_np(v_full, n_rep)

    total_key_len = k_full.shape[2]
    past_len = total_key_len - seq_len

    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(q, k_full.transpose(0, 1, 3, 2)) * scale

    if causal:
        query_positions = np.arange(seq_len).reshape(-1, 1) + past_len
        key_positions = np.arange(total_key_len).reshape(1, -1)
        mask_bool = key_positions > query_positions
        mask = np.where(mask_bool, -np.inf, 0.0)
        scores = scores + mask

    attn_weights = softmax_np(scores, axis=-1)
    out = np.matmul(attn_weights, v_full)
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, num_heads * head_dim)
    return out, attn_weights, updated_k_cache, updated_v_cache

    if causal:
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        scores = scores + mask

    attn_weights = softmax_np(scores, axis=-1)
    out = np.matmul(attn_weights, v)
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, num_heads * head_dim)
    return out, attn_weights


# ---------------------------------------------------------------------------
# テスト本体
# ---------------------------------------------------------------------------

def test_rms_norm_scale_invariance():
    """RMSNormの性質: 入力をスカラー倍しても、正規化後の出力は(weight=1なら)ほぼ不変。"""
    x = np.random.randn(2, 8).astype(np.float32)
    weight = np.ones(8, dtype=np.float32)
    out1 = rms_norm_np(x, weight)
    out2 = rms_norm_np(x * 5.0, weight)
    assert np.allclose(out1, out2, atol=1e-4), "RMSNorm should be scale-invariant"
    print("PASS: test_rms_norm_scale_invariance")


def test_rms_norm_output_scale():
    """RMSNorm出力のRMS(二乗平均平方根)が、weight=1のとき概ね1になることを確認。"""
    x = np.random.randn(4, 16).astype(np.float32) * 3.0
    weight = np.ones(16, dtype=np.float32)
    out = rms_norm_np(x, weight)
    rms = np.sqrt(np.mean(out ** 2, axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-3), f"expected RMS~1.0, got {rms}"
    print("PASS: test_rms_norm_output_scale")


def test_rope_preserves_norm():
    """RoPEの核心的性質: 回転操作なのでベクトルのノルム(長さ)は変化しない。"""
    head_dim = 8
    seq_len = 4
    cos, sin = build_rope_cache_np(head_dim, seq_len)
    q = np.random.randn(1, 1, seq_len, head_dim).astype(np.float32)
    k = np.random.randn(1, 1, seq_len, head_dim).astype(np.float32)
    q_rot, k_rot = apply_rope_np(q, k, cos, sin)

    norm_before = np.linalg.norm(q, axis=-1)
    norm_after = np.linalg.norm(q_rot, axis=-1)
    assert np.allclose(norm_before, norm_after, atol=1e-4), "RoPE must preserve vector norm"
    print("PASS: test_rope_preserves_norm")


def test_rope_relative_position_property():
    """RoPEの核心的性質: 2つの位置のQ・K内積は、絶対位置ではなく相対位置(差)にのみ依存する。
    これがRoPEが位置エンコーディングとして機能する理由そのもの。
    """
    head_dim = 8
    max_pos = 10
    cos, sin = build_rope_cache_np(head_dim, max_pos)

    rng = np.random.default_rng(0)
    q_vec = rng.standard_normal(head_dim).astype(np.float32)
    k_vec = rng.standard_normal(head_dim).astype(np.float32)

    def dot_at_positions(pos_q, pos_k):
        q = q_vec[None, None, None, :]
        k = k_vec[None, None, None, :]
        cq = cos[pos_q][None, None, None, :]
        sq = sin[pos_q][None, None, None, :]
        ck = cos[pos_k][None, None, None, :]
        sk = sin[pos_k][None, None, None, :]
        q_rot = (q * cq) + (rotate_half_np(q) * sq)
        k_rot = (k * ck) + (rotate_half_np(k) * sk)
        return float(np.sum(q_rot * k_rot))

    # (posQ=2, posK=0) と (posQ=5, posK=3) は相対距離が同じ(=2)なので内積が一致するはず
    d1 = dot_at_positions(2, 0)
    d2 = dot_at_positions(5, 3)
    assert abs(d1 - d2) < 1e-3, f"RoPE relative-position property violated: {d1} vs {d2}"
    print("PASS: test_rope_relative_position_property")


def test_attention_output_shape():
    """Attention出力の形状が [batch, seq_len, num_heads*head_dim] になることを確認。"""
    batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 5, 4, 2, 8
    q = np.random.randn(batch, seq_len, num_heads * head_dim).astype(np.float32)
    k = np.random.randn(batch, seq_len, num_kv_heads * head_dim).astype(np.float32)
    v = np.random.randn(batch, seq_len, num_kv_heads * head_dim).astype(np.float32)
    out, attn_weights, _, _ = attention_np(q, k, v, num_heads, num_kv_heads, causal=True)
    assert out.shape == (batch, seq_len, num_heads * head_dim)
    assert attn_weights.shape == (batch, num_heads, seq_len, seq_len)
    print("PASS: test_attention_output_shape")


def test_attention_weights_sum_to_one():
    """Attention weight(softmax後)は各クエリ位置について合計1になる。"""
    batch, seq_len, num_heads, num_kv_heads, head_dim = 1, 4, 2, 2, 4
    q = np.random.randn(batch, seq_len, num_heads * head_dim).astype(np.float32)
    k = np.random.randn(batch, seq_len, num_kv_heads * head_dim).astype(np.float32)
    v = np.random.randn(batch, seq_len, num_kv_heads * head_dim).astype(np.float32)
    _, attn_weights, _, _ = attention_np(q, k, v, num_heads, num_kv_heads, causal=False)
    sums = np.sum(attn_weights, axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-5), "softmax output must sum to 1"
    print("PASS: test_attention_weights_sum_to_one")


def test_causal_mask_blocks_future():
    """causal maskがある場合、位置iのAttention weightは位置i以降(未来)に対してほぼ0になる。"""
    batch, seq_len, num_heads, num_kv_heads, head_dim = 1, 4, 1, 1, 4
    q = np.random.randn(batch, seq_len, num_heads * head_dim).astype(np.float32)
    k = np.random.randn(batch, seq_len, num_kv_heads * head_dim).astype(np.float32)
    v = np.random.randn(batch, seq_len, num_kv_heads * head_dim).astype(np.float32)
    _, attn_weights, _, _ = attention_np(q, k, v, num_heads, num_kv_heads, causal=True)
    # attn_weights: [batch, num_heads, seq_len(query), seq_len(key)]
    # query位置0はkey位置1,2,3(未来)への重みが0であるべき
    future_weights = attn_weights[0, 0, 0, 1:]
    assert np.allclose(future_weights, 0.0, atol=1e-6), "causal mask should block future positions"
    print("PASS: test_causal_mask_blocks_future")


def test_repeat_kv_gqa_shape():
    """GQA用repeat_kvが、KVヘッド数をQヘッド数に正しく拡張することを確認。"""
    batch, num_kv_heads, seq_len, head_dim = 1, 2, 3, 4
    n_rep = 4  # num_heads(8) / num_kv_heads(2)
    x = np.random.randn(batch, num_kv_heads, seq_len, head_dim).astype(np.float32)
    out = repeat_kv_np(x, n_rep)
    assert out.shape == (batch, num_kv_heads * n_rep, seq_len, head_dim)
    # 複製元と複製先の値が一致することも確認
    assert np.allclose(out[0, 0], x[0, 0])
    assert np.allclose(out[0, n_rep - 1], x[0, 0])  # 同じKVヘッドから複製されたグループ内は同一
    print("PASS: test_repeat_kv_gqa_shape")


def test_kv_cache_matches_full_recompute():
    """【KVキャッシュ実装の正しさを保証する決定的テスト】

    「KVキャッシュを使って1トークンずつ逐次的にAttentionを計算した結果」と
    「キャッシュなしで全トークン列を一度に計算した結果」が完全に一致することを検証する。
    これが一致しなければ、KVキャッシュの実装(位置エンコーディングの扱い、
    causal maskの構築、キャッシュの連結順序など)のどこかに誤りがある。
    """
    batch, num_heads, num_kv_heads, head_dim = 1, 4, 2, 8
    total_len = 5

    rng = np.random.default_rng(42)
    q_full = rng.standard_normal((batch, total_len, num_heads * head_dim)).astype(np.float32)
    k_full = rng.standard_normal((batch, total_len, num_kv_heads * head_dim)).astype(np.float32)
    v_full = rng.standard_normal((batch, total_len, num_kv_heads * head_dim)).astype(np.float32)

    # --- ケースA: キャッシュなしで全列を一括計算 ---
    out_full, _, _, _ = attention_np(q_full, k_full, v_full, num_heads, num_kv_heads, causal=True)

    # --- ケースB: KVキャッシュを使って1トークンずつ逐次計算 ---
    out_incremental = np.zeros_like(out_full)
    k_cache, v_cache = None, None
    for t in range(total_len):
        q_t = q_full[:, t : t + 1, :]  # このステップの新しいクエリ(1トークン分)
        k_t = k_full[:, t : t + 1, :]
        v_t = v_full[:, t : t + 1, :]
        out_t, _, k_cache, v_cache = attention_np(
            q_t, k_t, v_t, num_heads, num_kv_heads, causal=True, past_k=k_cache, past_v=v_cache
        )
        out_incremental[:, t : t + 1, :] = out_t

    assert np.allclose(out_full, out_incremental, atol=1e-5), (
        "KVキャッシュを使った逐次計算が、一括計算と一致しません。"
        f"max diff = {np.max(np.abs(out_full - out_incremental))}"
    )
    print("PASS: test_kv_cache_matches_full_recompute")


def test_kv_cache_with_rope_matches_full_recompute():
    """RoPE適用込みでも、KVキャッシュありの逐次計算と一括計算が一致することを検証する。

    これが特に重要なのは、RoPEは各トークンの「絶対位置」に依存する回転を適用するため、
    キャッシュ利用時に正しい位置のcos/sinをスライスできているかが問われるため。
    """
    batch, num_heads, num_kv_heads, head_dim = 1, 2, 2, 8
    total_len = 4

    rng = np.random.default_rng(7)
    q_full = rng.standard_normal((batch, total_len, num_heads * head_dim)).astype(np.float32)
    k_full = rng.standard_normal((batch, total_len, num_kv_heads * head_dim)).astype(np.float32)
    v_full = rng.standard_normal((batch, total_len, num_kv_heads * head_dim)).astype(np.float32)

    cos, sin = build_rope_cache_np(head_dim, total_len)

    def apply_rope_heads(q_flat, k_flat, n_heads, n_kv_heads, position_offset, length):
        b = q_flat.shape[0]
        q_h = q_flat.reshape(b, length, n_heads, head_dim).transpose(0, 2, 1, 3)
        k_h = k_flat.reshape(b, length, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        cos_slice = cos[position_offset : position_offset + length]
        sin_slice = sin[position_offset : position_offset + length]
        q_rot, k_rot = apply_rope_np(q_h, k_h, cos_slice, sin_slice)
        q_flat_rot = q_rot.transpose(0, 2, 1, 3).reshape(b, length, n_heads * head_dim)
        k_flat_rot = k_rot.transpose(0, 2, 1, 3).reshape(b, length, n_kv_heads * head_dim)
        return q_flat_rot, k_flat_rot

    # --- ケースA: 一括計算(全トークンに対して位置0..total_len-1のRoPEを適用) ---
    q_rot_full, k_rot_full = apply_rope_heads(
        q_full, k_full, num_heads, num_kv_heads, position_offset=0, length=total_len
    )
    out_full, _, _, _ = attention_np(
        q_rot_full, k_rot_full, v_full, num_heads, num_kv_heads, causal=True
    )

    # --- ケースB: KVキャッシュありで逐次計算(各トークンごとに正しい絶対位置のRoPEを適用) ---
    out_incremental = np.zeros_like(out_full)
    k_cache, v_cache = None, None
    for t in range(total_len):
        q_t = q_full[:, t : t + 1, :]
        k_t = k_full[:, t : t + 1, :]
        v_t = v_full[:, t : t + 1, :]
        q_rot_t, k_rot_t = apply_rope_heads(
            q_t, k_t, num_heads, num_kv_heads, position_offset=t, length=1
        )
        out_t, _, k_cache, v_cache = attention_np(
            q_rot_t, k_rot_t, v_t, num_heads, num_kv_heads, causal=True, past_k=k_cache, past_v=v_cache
        )
        out_incremental[:, t : t + 1, :] = out_t

    assert np.allclose(out_full, out_incremental, atol=1e-5), (
        "RoPE込みのKVキャッシュ逐次計算が、一括計算と一致しません。"
        f"max diff = {np.max(np.abs(out_full - out_incremental))}"
    )
    print("PASS: test_kv_cache_with_rope_matches_full_recompute")


def merge_tensors_np(tensors, mode):
    """layers.pyのmerge_tensors()と全く同じロジックのnumpy版。"""
    if len(tensors) == 1:
        return tensors[0]
    if mode == "add":
        result = tensors[0]
        for t in tensors[1:]:
            result = result + t
        return result
    if mode == "average":
        result = tensors[0]
        for t in tensors[1:]:
            result = result + t
        return result / len(tensors)
    raise ValueError(f"Unknown merge mode: {mode!r}")


def test_merge_add_sums_all_inputs():
    """merge_mode="add"は全入力の要素ごとの和になる。"""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    c = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    result = merge_tensors_np([a, b, c], mode="add")
    expected = np.array([111.0, 222.0, 333.0], dtype=np.float32)
    assert np.allclose(result, expected)
    print("PASS: test_merge_add_sums_all_inputs")


def test_merge_average_divides_by_count():
    """merge_mode="average"は全入力の要素ごとの平均になる。"""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    result = merge_tensors_np([a, b], mode="average")
    expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    assert np.allclose(result, expected)
    print("PASS: test_merge_average_divides_by_count")


def test_merge_single_input_is_passthrough():
    """入力が1つだけの場合は、合成不要でそのまま返る(値がコピーされる訳ではないが同一)。"""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = merge_tensors_np([a], mode="add")
    assert np.allclose(result, a)
    result_avg = merge_tensors_np([a], mode="average")
    assert np.allclose(result_avg, a)
    print("PASS: test_merge_single_input_is_passthrough")


def test_merge_add_and_average_differ_for_multiple_inputs():
    """add と average は(入力が2つ以上のとき)異なる結果になる
    (average = add / 入力数、という関係が成り立つことを確認)。
    """
    a = np.array([2.0, 4.0], dtype=np.float32)
    b = np.array([6.0, 8.0], dtype=np.float32)
    added = merge_tensors_np([a, b], mode="add")
    averaged = merge_tensors_np([a, b], mode="average")
    assert np.allclose(averaged, added / 2)
    assert not np.allclose(added, averaged), "add and average should differ when there are 2+ inputs"
    print("PASS: test_merge_add_and_average_differ_for_multiple_inputs")


def test_merge_unknown_mode_raises():
    """未知のmerge_modeを指定するとエラーになる(サイレントな誤動作を防ぐ)。"""
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    try:
        merge_tensors_np([a, b], mode="multiply")
        assert False, "expected ValueError for unknown merge mode"
    except ValueError:
        print("PASS: test_merge_unknown_mode_raises")


def test_merge_three_or_more_inputs():
    """3つ以上の入力の合流でも、add/averageが正しく計算される
    (2入力だけでなく一般のN入力に対応していることの確認)。
    """
    tensors = [
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([2.0, 2.0], dtype=np.float32),
        np.array([3.0, 3.0], dtype=np.float32),
        np.array([4.0, 4.0], dtype=np.float32),
    ]
    added = merge_tensors_np(tensors, mode="add")
    averaged = merge_tensors_np(tensors, mode="average")
    assert np.allclose(added, [10.0, 10.0])
    assert np.allclose(averaged, [2.5, 2.5])
    print("PASS: test_merge_three_or_more_inputs")


if __name__ == "__main__":
    test_rms_norm_scale_invariance()
    test_rms_norm_output_scale()
    test_rope_preserves_norm()
    test_rope_relative_position_property()
    test_attention_output_shape()
    test_attention_weights_sum_to_one()
    test_causal_mask_blocks_future()
    test_repeat_kv_gqa_shape()
    test_kv_cache_matches_full_recompute()
    test_kv_cache_with_rope_matches_full_recompute()
    test_merge_add_sums_all_inputs()
    test_merge_average_divides_by_count()
    test_merge_single_input_is_passthrough()
    test_merge_add_and_average_differ_for_multiple_inputs()
    test_merge_unknown_mode_raises()
    test_merge_three_or_more_inputs()
    print("\nAll numpy logic-verification tests passed.")
