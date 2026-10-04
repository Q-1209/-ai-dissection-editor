"""
sampling.py

次トークンを選ぶためのサンプリング戦略。貪欲デコード(常に最大確率のトークンを選ぶ)
は決定的だが、同じ文脈に戻ると同じ予測を繰り返すため、生成が短いループに
陥りやすいという既知の性質がある。ここでは以下を実装する:

- temperature: logitsをtemperatureで割ってから確率分布を作る。1.0は変化なし、
  0より大きく1未満で分布を鋭く(より確定的に)、1より大きくで分布を滑らかに
  (よりランダムに)する。0はサポートしない(下記のgreedy相当の扱いに任せる)。
- top-k: 確率上位k個のトークンだけに絞ってからサンプリングする。
- top-p (nucleus sampling): 確率の高い順に累積確率がpを超えるまでのトークン集合
  だけを残してサンプリングする。
- repetition penalty: これまでに生成済み(または入力に含まれる)トークンのlogitsを
  下げることで、同じトークンの繰り返しを抑制する。

これらは組み合わせて使うことができる(適用順序: repetition penalty -> temperature
-> top-k -> top-p -> サンプリング、これはHuggingFace transformersの実装慣例に
合わせている)。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingConfig:
    temperature: float = 1.0  # 1.0で変化なし。0は貪欲デコード(argmax)として特別扱い
    top_k: int | None = None  # Noneまたは0で無効
    top_p: float | None = None  # Noneまたは1.0で無効
    repetition_penalty: float = 1.0  # 1.0で無効
    no_repeat_ngram_size: int | None = None  # Noneまたは0で無効


def apply_repetition_penalty(
    logits: torch.Tensor, generated_token_ids: list[int], penalty: float
) -> torch.Tensor:
    """既に生成済みのトークンのlogitsにペナルティを適用する。

    HuggingFace transformersの実装慣例に合わせ、正のlogitはpenaltyで割り、
    負のlogitはpenaltyを掛ける(どちらの場合も値を「小さく」する方向に働く)。

    logits: [vocab_size] (1トークン分)
    """
    if penalty == 1.0 or not generated_token_ids:
        return logits
    logits = logits.clone()
    unique_ids = set(generated_token_ids)
    for token_id in unique_ids:
        val = logits[token_id]
        logits[token_id] = val / penalty if val > 0 else val * penalty
    return logits


def apply_no_repeat_ngram(
    logits: torch.Tensor, generated_token_ids: list[int], ngram_size: int
) -> torch.Tensor:
    """直前の(ngram_size-1)トークンと同じ並びが過去に出現していた場合、
    その時「次に続いた」トークンを禁止する(logitsを-infにする)。

    これにより、同じnトークンの並び(n-gram)が2回以上出現することを防ぐ。
    repetition_penaltyが「既出トークン単体」を弱めるだけなのに対し、
    no_repeat_ngramは「既出の並び」を完全に禁止するため、長い定型句の
    繰り返し(例: 同じ文パターンの反復)により強く効く。

    HuggingFace transformersの実装慣例と同じアルゴリズム:
    過去の生成列を(ngram_size-1)個の窓でスキャンし、現在の直前
    (ngram_size-1)トークンと一致する窓を探して、その直後のトークンを
    禁止候補として集める。

    logits: [vocab_size]
    generated_token_ids: これまでに生成済みのトークンID列(プロンプト部分は含まない)
    """
    if not ngram_size or ngram_size <= 0:
        return logits
    prefix_len = ngram_size - 1
    if len(generated_token_ids) < prefix_len:
        return logits  # まだ比較できるだけの長さがない

    current_prefix = tuple(generated_token_ids[-prefix_len:]) if prefix_len > 0 else ()

    banned_tokens: set[int] = set()
    for i in range(len(generated_token_ids) - prefix_len):
        window = tuple(generated_token_ids[i : i + prefix_len])
        if window == current_prefix:
            banned_tokens.add(generated_token_ids[i + prefix_len])

    if not banned_tokens:
        return logits

    # 安全策: 万が一(語彙が極端に小さいテスト等で)禁止対象が語彙全体を覆って
    # しまう場合、全て-infにするとsoftmaxがNaNを生むため、その場合は適用しない
    if len(banned_tokens) >= logits.shape[-1]:
        return logits

    logits = logits.clone()
    for token_id in banned_tokens:
        logits[token_id] = float("-inf")
    return logits


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """logitsをtemperatureで割る。temperature=1.0は変化なし。"""
    if temperature == 1.0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """確率上位k個以外のlogitsを-infにする(以降のsoftmaxで確率0になる)。"""
    if top_k is None or top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, top_k)
    min_kept_value = values[-1]
    return torch.where(logits < min_kept_value, torch.full_like(logits, float("-inf")), logits)


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """累積確率がtop_pを超える手前までのトークンだけを残し、それ以外を-infにする
    (nucleus sampling)。
    """
    if top_p is None or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 累積確率がtop_pを初めて超えた位置より後ろを削る。
    # 最初のトークン(最大確率のもの)は必ず残す(全て削ってしまうのを防ぐ)。
    sorted_mask = cumulative_probs > top_p
    sorted_mask[0] = False

    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # ソート前の順序に戻す
    result = torch.full_like(logits, float("-inf"))
    result.scatter_(0, sorted_indices, sorted_logits)
    return result


def sample_next_token(
    logits: torch.Tensor,
    config: SamplingConfig,
    generated_token_ids: list[int],
) -> int:
    """1トークン分のlogitsから、設定に従って次のトークンを選ぶ。

    logits: [vocab_size]
    no_repeat_ngram_sizeは、貪欲デコード(temperature=0.0)・サンプリングの
    どちらでも適用される(繰り返し抑制はデコード方式に関わらず有効にすべきため)。
    config.temperature == 0.0 の場合は、no_repeat_ngram適用後に貪欲デコード
    (argmax)として扱う(他のサンプリングパラメータは無視される)。
    """
    if config.no_repeat_ngram_size:
        logits = apply_no_repeat_ngram(logits, generated_token_ids, config.no_repeat_ngram_size)

    if config.temperature == 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = apply_repetition_penalty(logits, generated_token_ids, config.repetition_penalty)
    logits = apply_temperature(logits, config.temperature)
    if config.top_k:
        logits = apply_top_k(logits, config.top_k)
    if config.top_p:
        logits = apply_top_p(logits, config.top_p)

    probs = torch.softmax(logits, dim=-1)
    next_token_id = int(torch.multinomial(probs, num_samples=1).item())
    return next_token_id
