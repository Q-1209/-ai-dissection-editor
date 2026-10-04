"""
layers.py

Llama系アーキテクチャ(Llama/Qwen/Mistral)の基本演算をPyTorchで自前実装する。
transformersライブラリには依存せず、必要な計算だけを直接書く。

含まれる演算:
- RMSNorm: Llama系で使われる正規化(LayerNormと違い平均を引かない)
- RoPE (Rotary Position Embedding): 位置情報をQ/Kに回転行列として埋め込む
- Attention: GQA(Grouped Query Attention)対応、causal mask付き
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: LayerNormと異なり平均を引かず、二乗平均平方根でスケールを揃える。

    x: [..., hidden_size]
    weight: [hidden_size] (学習済みスケールパラメータ)
    """
    # 数値安定性のためfloat32で計算してから元のdtypeに戻す(transformers実装の慣例に合わせる)
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x.to(input_dtype))


def build_rope_cache(
    head_dim: int, max_position: int, base: float = 10000.0, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPEのcos/sinテーブルを事前計算する。

    head_dim: 1ヘッドあたりの次元数(通常 hidden_size // num_heads)
    max_position: 事前計算しておく最大シーケンス長
    戻り値: cos, sin それぞれ [max_position, head_dim]
    """
    # 周波数: 1 / base^(2i/head_dim) for i in [0, head_dim/2)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_position, dtype=torch.float32)
    # freqs: [max_position, head_dim/2]
    freqs = torch.outer(positions, inv_freq)
    # RoPEの実装慣例: 前半と後半に同じ角度を複製して head_dim ぶんにする
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_position, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE適用のためのヘルパー: ベクトルの前半・後半を入れ替えて符号反転する。"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Q, Kに回転位置エンコーディングを適用する。

    q, k: [batch, num_heads, seq_len, head_dim]
    cos, sin: [seq_len, head_dim] (該当する位置ぶんだけスライスしたもの)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rotated = (q * cos) + (_rotate_half(q) * sin)
    k_rotated = (k * cos) + (_rotate_half(k) * sin)
    return q_rotated, k_rotated


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Linear層。バイアスはオプション(Noneならバイアスなし)。

    safetensorsの重みは [out_features, in_features] の形で保存されているため、
    F.linearの規約(weightを転置して掛ける)にそのまま渡せる。

    注意: Qwen2はQ/K/V projectionにバイアスを持つ(o_projやLlamaの各層はバイアスなし)、
    という一般的でないアーキテクチャ差異があるため、呼び出し側で必要に応じてbiasを渡す。
    """
    return F.linear(x, weight, bias)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA(Grouped Query Attention)用: KVヘッド数がQヘッド数より少ない場合、
    各KVヘッドをn_rep回複製してQヘッド数に合わせる。

    x: [batch, num_kv_heads, seq_len, head_dim]
    戻り値: [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    causal: bool = True,
    past_k: torch.Tensor | None = None,
    past_v: torch.Tensor | None = None,
    head_norm_capture: dict | None = None,
    head_norm_capture_key: str | None = None,
    head_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scaled Dot-Product Attention。GQA対応、causal maskはオプション。KVキャッシュ対応。

    q: [batch, seq_len, num_heads * head_dim]
    k, v: [batch, seq_len, num_kv_heads * head_dim] (今回新しく計算した分のみ)
    past_k, past_v: [batch, num_kv_heads, past_len, head_dim] または None
                    (これまでにキャッシュされたK/V。ヘッド分割済みの形で渡す)

    戻り値: (attention出力, attention重み, 更新後のK cache, 更新後のV cache)
            cacheはヘッド分割済み [batch, num_kv_heads, total_len, head_dim] の形。
            attention重みは [batch, num_heads, seq_len, total_key_len] の形
            (Phase2の内部状態可視化で「どのトークンがどのトークンに注目しているか」
            のヒートマップ表示に使う。呼び出し側で不要なら単に無視すればよい)。

    KVキャッシュを使う場合の典型的な呼び出しパターン:
      - 1トークンずつ生成する場合、qは新しいトークン1個分だけ([batch, 1, ...])、
        past_k/past_vにはこれまでの全トークン分を渡す。
      - causal maskは「新しいクエリ位置」が「これまでの全キー位置+自分自身」を
        見られるように、query_len × total_key_len の矩形で構築する
        (通常のcausal maskの下三角部分を、seq_len分だけ右にずらした形になる)。
    """
    batch, seq_len, _ = q.shape
    head_dim = q.shape[-1] // num_heads

    # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k_new = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v_new = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    if past_k is not None and past_v is not None:
        k_full = torch.cat([past_k, k_new], dim=2)  # 時系列方向(dim=2)に連結
        v_full = torch.cat([past_v, v_new], dim=2)
    else:
        k_full = k_new
        v_full = v_new

    updated_k_cache = k_full
    updated_v_cache = v_full

    n_rep = num_heads // num_kv_heads
    k_full = repeat_kv(k_full, n_rep)
    v_full = repeat_kv(v_full, n_rep)

    total_key_len = k_full.shape[2]
    past_len = total_key_len - seq_len

    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale  # [batch, num_heads, seq_len, total_key_len]

    if causal:
        # 新しいクエリ位置 i (0-indexed, 実際の絶対位置は past_len+i) は、
        # 絶対位置 past_len+i 以下のキーだけを見られる。
        # つまり query行iでは、key列の [0, past_len+i] までがマスク対象外(見える)。
        query_positions = torch.arange(seq_len, device=q.device).unsqueeze(1) + past_len
        key_positions = torch.arange(total_key_len, device=q.device).unsqueeze(0)
        mask_bool = key_positions > query_positions  # True の位置を -inf でマスク
        mask = torch.zeros(seq_len, total_key_len, device=q.device, dtype=q.dtype)
        mask = mask.masked_fill(mask_bool, float("-inf"))
        scores = scores + mask

    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v_full)  # [batch, num_heads, seq_len, head_dim]

    if head_scale is not None:
        # Live Slider(Phase3、2026-09-24追加): ヘッドごとの出力に倍率を掛ける([num_heads])。
        # o_projでヘッドが混ざる前に掛けるので、1ヘッドだけを弱める/強めることができる。
        out = out * head_scale.to(dtype=out.dtype, device=out.device).view(1, -1, 1, 1)

    if head_norm_capture is not None and head_norm_capture_key is not None:
        # Neuron Search(2026-09-12追加)用: 各ヘッドの出力ベクトル(head_dim次元)のL2ノルムを、
        # そのヘッドの「その位置での活性化の強さ」の代理指標として記録する。
        # o_proj(ヘッドをまとめる線形層)を通す前の、ヘッドごとに独立した値であることが重要
        # (o_proj後は全ヘッドの寄与が混ざってしまい、ヘッド単位の探索ができなくなる)。
        head_norm_capture[head_norm_capture_key] = out.norm(dim=-1).detach()  # [batch, num_heads, seq_len]

    out = out.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)
    return out, attn_weights, updated_k_cache, updated_v_cache


def swiglu_mlp(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    neuron_capture: dict | None = None,
    neuron_capture_key: str | None = None,
    neuron_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Llama系で使われるSwiGLU MLP: down(silu(gate(x)) * up(x))

    neuron_scale: Noneでなければ、中間活性化値(intermediate_size次元)に要素ごとの倍率を掛けてから
    down_projに通す(Live Slider、2026-09-24追加)。倍率1の次元は変化しない。

    neuron_capture: Noneでなければ、down_projを通す前の中間活性化値
    (silu(gate(x)) * up(x)、次元数=intermediate_size)をneuron_capture_key で
    この辞書に記録する(Neuron Search、2026-09-12追加用)。中間層の「1ニューロン」を
    MLPのgate/up projectionの特定次元として扱う際の値そのものになる。
    """
    gate = F.silu(linear(x, gate_w))
    up = linear(x, up_w)
    intermediate = gate * up
    if neuron_scale is not None:
        intermediate = intermediate * neuron_scale.to(dtype=intermediate.dtype, device=intermediate.device)
    if neuron_capture is not None and neuron_capture_key is not None:
        neuron_capture[neuron_capture_key] = intermediate.detach()  # [batch, seq_len, intermediate_size]
    return linear(intermediate, down_w)


def merge_tensors(tensors: list[torch.Tensor], mode: str) -> torch.Tensor:
    """複数ノードの出力(合流)を1つのテンソルに組み合わせる。

    mode="add": 要素ごとの加算(残差接続と同じ発想。複数の情報経路を単純に足し合わせる)。
    mode="average": 要素ごとの平均(各入力の影響を均等にしたい場合)。

    全てのテンソルは同じshapeであることを前提とする(呼び出し側で
    validate_shape_compatibilityにより保証される)。
    """
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
    raise ValueError(f"Unknown merge mode: {mode!r}. Expected 'add' or 'average'.")
