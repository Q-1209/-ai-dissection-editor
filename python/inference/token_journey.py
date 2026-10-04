"""
token_journey.py

Phase2: Token Journey の中核ロジック。

各ブロック(Attention直後・MLP直後)を通過した後のトークンの隠れ状態ベクトルを、
埋め込み行列(model.embed_tokens.weight)の各行とのコサイン類似度で比較し、
「その時点で最も意味が近い単語」を求める(logit lensに近い考え方)。

層を追うごとにこの「最も近い単語」がどう変化するかを見ることで、トークンの
意味がモデル内部でどう変遷していくかを追跡できる。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .layers import rms_norm


@dataclass
class NearestWordMatch:
    token_id: int
    similarity: float


def find_nearest_words(
    hidden_vector: torch.Tensor, embedding_matrix: torch.Tensor, top_k: int = 1
) -> list[NearestWordMatch]:
    """1つの隠れ状態ベクトルに対して、埋め込み行列内で最もコサイン類似度が
    高い単語(トークン)を上位top_k件求める。

    hidden_vector: [hidden_size]
    embedding_matrix: [vocab_size, hidden_size] (通常はmodel.embed_tokens.weight)
    戻り値: 類似度が高い順のNearestWordMatchのリスト(長さtop_k)
    """
    hidden_norm = F.normalize(hidden_vector.float().unsqueeze(0), dim=-1)  # [1, hidden_size]
    embed_norm = F.normalize(embedding_matrix.float(), dim=-1)  # [vocab_size, hidden_size]

    similarities = (hidden_norm @ embed_norm.T).squeeze(0)  # [vocab_size]

    top_k = min(top_k, similarities.shape[0])
    top_values, top_indices = torch.topk(similarities, top_k)

    return [
        NearestWordMatch(token_id=int(idx.item()), similarity=float(val.item()))
        for val, idx in zip(top_values, top_indices)
    ]


def compute_token_journey(
    internals: dict[str, torch.Tensor],
    block_order: list[str],
    embedding_matrix: torch.Tensor,
    seq_len: int,
    top_k: int = 1,
    final_norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> dict[str, list[list[NearestWordMatch]]]:
    """全ブロックの内部状態(internals)から、各トークン位置についての
    Token Journey(層を追った意味の変遷)を計算する。

    internals: execute(capture_internals=True)が返すExecutionResult.internals。
               キーは "{node_id}.output" の形式(attn_weightsは無視する)。
    block_order: どのブロックをどの順序で追跡するか(通常はexecuted_orderから
                 ".output"を持つものだけを抽出したもの)。
    embedding_matrix: [vocab_size, hidden_size]
    seq_len: 入力文のトークン数(各ブロック出力からこの長さぶんの位置を取り出す)
    top_k: 各(トークン位置, ブロック)ごとに何件の近傍単語を求めるか

    final_norm_weight: モデル最終段のRMSNorm重み("model.norm.weight")。指定した場合、
        各ブロック出力をこのRMSNorm(lm_headが実際に適用するのと同じもの)に通してから
        埋め込み行列との類似度を取る("logit lens"の標準的な手法。nostalgebraist等の
        interpretability研究で使われる補正)。Noneの場合は生の隠れ状態ベクトルをそのまま
        使う(補正なし)。

        補正が必要な理由: Llama系(特にQwen2.5)の中間層は「Massive Activations」という
        既知の現象(arxiv 2402.17762)を持ち、ごく一部の次元が異常に大きな値を取る。
        これを補正せずに生の隠れ状態同士でコサイン類似度を取ると、その一部の次元の
        向きだけで類似度の大小がほぼ決まってしまい、多くの異なるブロック・トークン位置が
        (本来の意味とは無関係な)同じ単語に類似度最大として吸着してしまう
        (実機観測: Qwen2.5-0.5Bで層2以降、無関係な同一の低頻度トークンに収束する現象を確認)。
        RMSNormの学習済みスケール(final_norm_weight)は、モデルがLM Head直前で
        まさにこの外れ値次元を抑えて意味のある方向を取り出すために学習した重みなので、
        これを中間層にも適用することで、外れ値次元への吸着をある程度緩和できる
        (完全な補正ではない。中間層専用に学習されたものではないため、"tuned lens"のような
        層ごとに学習した補正には及ばないが、無補正よりは大幅に改善する)。
    norm_eps: RMSNormのeps(final_norm_weightを渡す場合のみ使用)。

    戻り値: {block_id: [[トークン位置0の上位top_k件], [トークン位置1の上位top_k件], ...]}
            の辞書(block_orderの順序で並ぶ)。
    """
    journey: dict[str, list[list[NearestWordMatch]]] = {}

    for block_id in block_order:
        output_key = f"{block_id}.output"
        if output_key not in internals:
            continue
        block_output = internals[output_key]  # [batch, seq_len, hidden_size]
        if final_norm_weight is not None:
            block_output = rms_norm(block_output, final_norm_weight, eps=norm_eps)

        per_position_matches: list[list[NearestWordMatch]] = []
        for pos in range(seq_len):
            vector = block_output[0, pos, :]  # [hidden_size]
            matches = find_nearest_words(vector, embedding_matrix, top_k=top_k)
            per_position_matches.append(matches)

        journey[block_id] = per_position_matches

    return journey
