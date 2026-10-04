"""
sensitivity.py

Phase2: Prompt Sensitivity Map の中核ロジック。

入力文の各トークンを1つずつ「削除」または「置換」し、その都度モデルに
forward passさせて、元の出力(次トークンのlogits分布)からどれだけ変化したかを
測定する。KLダイバージェンスを主指標とし、最上位予測トークンが変わったかどうかも
補足情報として記録する。

「どのトークンが出力に一番強く影響しているか」を、総当たりの再推論と差分計測
だけで求めるシンプルな手法(実装は単純だが、seq_len回の追加forward passが
必要になるため、長い入力ではそれなりに時間がかかる)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

ReplacementMode = Literal["delete", "pad", "random", "custom"]


@dataclass
class TokenSensitivity:
    position: int
    original_token_id: int
    kl_divergence: float
    top_token_changed: bool
    original_top_token_id: int
    modified_top_token_id: int


def compute_kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """2つのlogits分布間のKLダイバージェンス KL(P || Q) を計算する。

    p_logits, q_logits: [vocab_size] (1トークン分の生logits、softmax前)
    値が大きいほど、Qの分布がPから大きく変化していることを示す。
    KLダイバージェンスは非対称(KL(P||Q) != KL(Q||P))だが、ここでは
    「オリジナルの分布Pを基準に、変更後の分布Qがどれだけ乖離したか」を
    測りたいので、P=original, Q=modified として計算する。
    """
    p_log_probs = F.log_softmax(p_logits.float(), dim=-1)
    q_log_probs = F.log_softmax(q_logits.float(), dim=-1)
    p_probs = p_log_probs.exp()
    kl = (p_probs * (p_log_probs - q_log_probs)).sum()
    return float(kl)


def build_modified_input(
    input_ids: torch.Tensor,
    position: int,
    mode: ReplacementMode,
    pad_token_id: int | None = None,
    custom_token_id: int | None = None,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """指定位置のトークンを削除または置換した新しい入力テンソルを作る。

    input_ids: [1, seq_len]
    mode:
      "delete": 指定位置のトークンを取り除く(seq_lenが1減る)
      "pad": パディングトークン(pad_token_id、無ければ0)に置き換える
      "random": 語彙全体からランダムに選んだトークンに置き換える(vocab_size必須)
      "custom": custom_token_idで指定したトークンに置き換える
    """
    seq_len = input_ids.shape[1]
    if mode == "delete":
        keep = [i for i in range(seq_len) if i != position]
        return input_ids[:, keep]

    modified = input_ids.clone()
    if mode == "pad":
        replacement = pad_token_id if pad_token_id is not None else 0
    elif mode == "random":
        if vocab_size is None:
            raise ValueError("vocab_size is required for mode='random'")
        replacement = int(torch.randint(0, vocab_size, (1,)).item())
    elif mode == "custom":
        if custom_token_id is None:
            raise ValueError("custom_token_id is required for mode='custom'")
        replacement = custom_token_id
    else:
        raise ValueError(f"Unknown replacement mode: {mode!r}")

    modified[0, position] = replacement
    return modified


def run_sensitivity_analysis(
    forward_fn,
    input_ids: torch.Tensor,
    mode: ReplacementMode,
    pad_token_id: int | None = None,
    custom_token_id: int | None = None,
    vocab_size: int | None = None,
) -> list[TokenSensitivity]:
    """入力文の各トークン位置について、削除/置換による影響度を測定する。

    forward_fn: torch.Tensor([1, seq_len]) -> torch.Tensor([1, seq_len, vocab_size])
                (入力トークン列を受け取り、logitsを返す関数。呼び出し側で
                 executor.execute()をラップして渡す想定)
    input_ids: [1, seq_len] 分析対象の元の入力
    戻り値: 各トークン位置ごとのTokenSensitivityのリスト(position順)
    """
    seq_len = input_ids.shape[1]

    original_logits = forward_fn(input_ids)  # [1, seq_len, vocab_size]
    original_next_logits = original_logits[0, -1, :]  # 最後の位置の次トークン予測
    original_top_token_id = int(torch.argmax(original_next_logits).item())

    results: list[TokenSensitivity] = []
    for pos in range(seq_len):
        modified_input = build_modified_input(
            input_ids, pos, mode,
            pad_token_id=pad_token_id, custom_token_id=custom_token_id, vocab_size=vocab_size,
        )
        modified_logits = forward_fn(modified_input)
        modified_next_logits = modified_logits[0, -1, :]
        modified_top_token_id = int(torch.argmax(modified_next_logits).item())

        kl = compute_kl_divergence(original_next_logits, modified_next_logits)

        results.append(
            TokenSensitivity(
                position=pos,
                original_token_id=int(input_ids[0, pos].item()),
                kl_divergence=kl,
                top_token_changed=(modified_top_token_id != original_top_token_id),
                original_top_token_id=original_top_token_id,
                modified_top_token_id=modified_top_token_id,
            )
        )

    return results
