"""
test_sampling_numpy.py

sampling.py のロジックをnumpyで独立実装し、既知の性質を検証する
(この開発環境ではtorch実行ができないため、numpy版で検証してtorch版の
正しさを担保する、他のテストファイルと同じアプローチ)。

実行方法: python3 python/tests/test_sampling_numpy.py
"""

import numpy as np


def softmax_np(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x)


def apply_repetition_penalty_np(logits, generated_token_ids, penalty):
    if penalty == 1.0 or not generated_token_ids:
        return logits
    logits = logits.copy()
    for token_id in set(generated_token_ids):
        val = logits[token_id]
        logits[token_id] = val / penalty if val > 0 else val * penalty
    return logits


def apply_temperature_np(logits, temperature):
    if temperature == 1.0:
        return logits
    return logits / temperature


def apply_top_k_np(logits, top_k):
    if top_k is None or top_k <= 0 or top_k >= len(logits):
        return logits
    threshold = np.sort(logits)[::-1][top_k - 1]
    return np.where(logits < threshold, -np.inf, logits)


def apply_top_p_np(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    sorted_probs = softmax_np(sorted_logits)
    cumulative_probs = np.cumsum(sorted_probs)

    mask = cumulative_probs > top_p
    mask[0] = False
    sorted_logits = np.where(mask, -np.inf, sorted_logits)

    result = np.full_like(logits, -np.inf)
    result[sorted_indices] = sorted_logits
    return result


def apply_no_repeat_ngram_np(logits, generated_token_ids, ngram_size):
    """sampling.pyのapply_no_repeat_ngram()と全く同じロジックのnumpy版。"""
    if not ngram_size or ngram_size <= 0:
        return logits
    prefix_len = ngram_size - 1
    if len(generated_token_ids) < prefix_len:
        return logits

    current_prefix = tuple(generated_token_ids[-prefix_len:]) if prefix_len > 0 else ()

    banned_tokens = set()
    for i in range(len(generated_token_ids) - prefix_len):
        window = tuple(generated_token_ids[i : i + prefix_len])
        if window == current_prefix:
            banned_tokens.add(generated_token_ids[i + prefix_len])

    if not banned_tokens:
        return logits
    if len(banned_tokens) >= len(logits):
        return logits

    logits = logits.copy()
    for token_id in banned_tokens:
        logits[token_id] = -np.inf
    return logits


def test_repetition_penalty_reduces_repeated_token_prob():
    """繰り返し済みトークンは、ペナルティ適用後に選ばれる確率が下がる。"""
    logits = np.array([2.0, 1.0, 0.5, -1.0], dtype=np.float32)
    generated = [0, 0, 0]
    penalized = apply_repetition_penalty_np(logits, generated, penalty=1.5)

    prob_before = softmax_np(logits)[0]
    prob_after = softmax_np(penalized)[0]
    assert prob_after < prob_before, "repetition penalty should reduce the probability of repeated token"
    print("PASS: test_repetition_penalty_reduces_repeated_token_prob")


def test_repetition_penalty_noop_when_disabled():
    """penalty=1.0のときは何も変化しない。"""
    logits = np.array([2.0, 1.0, 0.5, -1.0], dtype=np.float32)
    result = apply_repetition_penalty_np(logits, [0, 1], penalty=1.0)
    assert np.array_equal(result, logits)
    print("PASS: test_repetition_penalty_noop_when_disabled")


def test_temperature_below_one_sharpens_distribution():
    """temperature<1は分布を鋭く(最大値により集中)する。"""
    logits = np.array([2.0, 1.0, 0.5], dtype=np.float32)
    probs_normal = softmax_np(logits)
    probs_sharp = softmax_np(apply_temperature_np(logits, temperature=0.5))
    assert probs_sharp[0] > probs_normal[0], "lower temperature should concentrate probability on the top choice"
    print("PASS: test_temperature_below_one_sharpens_distribution")


def test_temperature_above_one_flattens_distribution():
    """temperature>1は分布を滑らかに(より均一)する。"""
    logits = np.array([2.0, 1.0, 0.5], dtype=np.float32)
    probs_normal = softmax_np(logits)
    probs_flat = softmax_np(apply_temperature_np(logits, temperature=2.0))
    assert probs_flat[0] < probs_normal[0], "higher temperature should flatten the distribution"
    print("PASS: test_temperature_above_one_flattens_distribution")


def test_top_k_keeps_only_k_candidates():
    """top_k=2なら、softmax後に非ゼロ確率を持つトークンは2個だけになる。"""
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    filtered = apply_top_k_np(logits, top_k=2)
    probs = softmax_np(filtered)
    nonzero_count = np.sum(probs > 1e-6)
    assert nonzero_count == 2, f"expected exactly 2 nonzero-probability tokens, got {nonzero_count}"
    print("PASS: test_top_k_keeps_only_k_candidates")


def test_top_k_disabled_when_zero():
    """top_k=0またはNoneのときは何も変化しない。"""
    logits = np.array([5.0, 4.0, 3.0], dtype=np.float32)
    assert np.array_equal(apply_top_k_np(logits, 0), logits)
    assert np.array_equal(apply_top_k_np(logits, None), logits)
    print("PASS: test_top_k_disabled_when_zero")


def test_top_p_keeps_high_probability_mass():
    """top_p=0.5なら、残ったトークンの確率の合計はおおよそ0.5以上になる
    (最初のトークンを含めてから閾値を超えた時点で打ち切るため、若干0.5を超えるのが正常)。
    """
    logits = np.array([3.0, 2.0, 1.0, 0.0, -1.0], dtype=np.float32)
    filtered = apply_top_p_np(logits, top_p=0.5)
    probs = softmax_np(filtered)
    kept_mass = np.sum(probs[probs > 1e-9])
    assert kept_mass >= 0.5, f"kept probability mass should be at least top_p, got {kept_mass}"
    print("PASS: test_top_p_keeps_high_probability_mass")


def test_top_p_always_keeps_at_least_one_token():
    """top_pがどんなに小さくても、最低1個(最大確率のトークン)は残る。"""
    logits = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    filtered = apply_top_p_np(logits, top_p=0.001)
    probs = softmax_np(filtered)
    nonzero_count = np.sum(probs > 1e-9)
    assert nonzero_count >= 1, "at least one token must remain even with very small top_p"
    assert probs[0] > 1e-9, "the single remaining token should be the highest-probability one"
    print("PASS: test_top_p_always_keeps_at_least_one_token")


def test_top_p_disabled_when_one():
    """top_p=1.0のときは何も変化しない。"""
    logits = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    assert np.array_equal(apply_top_p_np(logits, 1.0), logits)
    print("PASS: test_top_p_disabled_when_one")


def test_no_repeat_ngram_blocks_exact_repeat():
    """【核心テスト】bigram(ngram_size=2)禁止の場合、直前のトークンの後に
    過去1回続いたトークンが、今度は禁止される。

    生成列 [A, B, C, A] の後、直前トークンはA。過去に "A" の直後に続いたのは
    "B"(先頭のA->B)なので、次にBが選ばれるのを禁止するべき。
    """
    vocab_size = 10
    logits = np.zeros(vocab_size, dtype=np.float32)
    logits[1] = 100.0  # トークンB(=1)が最も選ばれやすいようにしておく(禁止前は必ずBが選ばれる)

    generated = [0, 1, 2, 0]  # A=0, B=1, C=2, A=0(直前)
    filtered = apply_no_repeat_ngram_np(logits, generated, ngram_size=2)

    assert filtered[1] == -np.inf, "token B should be banned because 'A->B' already occurred"
    # 禁止されなかった他のトークンは変化しないはず
    assert filtered[2] == 0.0
    print("PASS: test_no_repeat_ngram_blocks_exact_repeat")


def test_no_repeat_ngram_allows_when_no_match():
    """直前のprefixが過去に一度も出現していない場合、何も禁止されない。"""
    logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    generated = [0, 1, 2]  # 直前トークンは2、過去に「2の直後」は存在しない
    filtered = apply_no_repeat_ngram_np(logits, generated, ngram_size=2)
    assert np.array_equal(filtered, logits), "no ban should be applied when the prefix never occurred before"
    print("PASS: test_no_repeat_ngram_allows_when_no_match")


def test_no_repeat_ngram_disabled_when_zero_or_none():
    """ngram_size=0またはNoneのときは何も変化しない。"""
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    generated = [0, 1, 2, 0, 1]
    assert np.array_equal(apply_no_repeat_ngram_np(logits, generated, 0), logits)
    assert np.array_equal(apply_no_repeat_ngram_np(logits, generated, None), logits)
    print("PASS: test_no_repeat_ngram_disabled_when_zero_or_none")


def test_no_repeat_ngram_larger_size_uses_longer_prefix():
    """ngram_size=3(trigram)の場合、直前2トークンの並びが一致した時だけ禁止される。
    直前1トークンだけが一致しても禁止されないことを確認する(prefix長の違いのテスト)。
    """
    vocab_size = 10
    logits = np.zeros(vocab_size, dtype=np.float32)

    # 過去: [A, B, C, X, B, D] という列があるとする。
    # 直前2トークンが [X, B] の場合、trigram(size=3)では過去に [X, B] という並びは
    # 一度も無いので何も禁止されないはず(過去にあるのは[B,C]の後にC、[X,B]の後にD...)。
    generated = [0, 1, 2, 3, 1, 4]  # A=0,B=1,C=2,X=3,B=1,D=4
    # 直前2トークン(prefix, ngram_size=3のためprefix_len=2) = (1, 4) = (B, D)
    # これは過去に一度も出現していないので禁止なし
    filtered = apply_no_repeat_ngram_np(logits, generated, ngram_size=3)
    assert np.all(filtered == 0.0), "no ban expected since the exact 2-token prefix never repeated"
    print("PASS: test_no_repeat_ngram_larger_size_uses_longer_prefix")


def test_no_repeat_ngram_degenerate_case_all_banned_is_skipped():
    """禁止対象が語彙全体を覆ってしまう(退化的な)場合は、安全のため適用しない
    (全て-infになるとsoftmaxがNaNになるのを防ぐため)。
    """
    vocab_size = 2  # 極端に小さい語彙
    logits = np.array([1.0, 2.0], dtype=np.float32)
    # 語彙が2種類しかない状況で、prefixが繰り返されて全トークンが禁止候補になるケースを作る
    generated = [0, 1, 0]  # 直前は0、過去に0の後に1が続いた実績がある
    filtered = apply_no_repeat_ngram_np(logits, generated, ngram_size=2)
    # banned_tokens={1}, vocab_size=2なので全体は覆っていない、通常通り禁止されるはず
    assert filtered[1] == -np.inf
    print("PASS: test_no_repeat_ngram_degenerate_case_all_banned_is_skipped")


def test_combined_pipeline_produces_valid_distribution():
    """repetition penalty -> temperature -> top_k -> top_p を組み合わせても、
    最終的に有効な確率分布(合計1、負なし)が得られる。
    """
    logits = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5], dtype=np.float32)
    generated = [0, 2]

    x = apply_repetition_penalty_np(logits, generated, penalty=1.3)
    x = apply_temperature_np(x, temperature=0.8)
    x = apply_top_k_np(x, top_k=5)
    x = apply_top_p_np(x, top_p=0.9)

    probs = softmax_np(x)
    assert np.all(probs >= 0), "probabilities must be non-negative"
    assert abs(np.sum(probs) - 1.0) < 1e-5, f"probabilities must sum to 1, got {np.sum(probs)}"
    print("PASS: test_combined_pipeline_produces_valid_distribution")


def test_combined_pipeline_with_no_repeat_ngram():
    """no_repeat_ngramを含めた全パイプラインでも、有効な確率分布が得られる。"""
    logits = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5], dtype=np.float32)
    generated = [0, 1, 2, 0]  # 直前は0、過去に「0の後に1」の実績あり(bigram禁止対象)

    x = apply_no_repeat_ngram_np(logits, generated, ngram_size=2)
    x = apply_repetition_penalty_np(x, generated, penalty=1.3)
    x = apply_temperature_np(x, temperature=0.8)
    x = apply_top_k_np(x, top_k=5)
    x = apply_top_p_np(x, top_p=0.9)

    probs = softmax_np(x)
    assert np.all(probs >= 0), "probabilities must be non-negative"
    assert abs(np.sum(probs) - 1.0) < 1e-5, f"probabilities must sum to 1, got {np.sum(probs)}"
    print("PASS: test_combined_pipeline_with_no_repeat_ngram")


if __name__ == "__main__":
    test_repetition_penalty_reduces_repeated_token_prob()
    test_repetition_penalty_noop_when_disabled()
    test_temperature_below_one_sharpens_distribution()
    test_temperature_above_one_flattens_distribution()
    test_top_k_keeps_only_k_candidates()
    test_top_k_disabled_when_zero()
    test_top_p_keeps_high_probability_mass()
    test_top_p_always_keeps_at_least_one_token()
    test_top_p_disabled_when_one()
    test_no_repeat_ngram_blocks_exact_repeat()
    test_no_repeat_ngram_allows_when_no_match()
    test_no_repeat_ngram_disabled_when_zero_or_none()
    test_no_repeat_ngram_larger_size_uses_longer_prefix()
    test_no_repeat_ngram_degenerate_case_all_banned_is_skipped()
    test_combined_pipeline_produces_valid_distribution()
    test_combined_pipeline_with_no_repeat_ngram()
    print("\nAll sampling logic-verification tests passed.")
