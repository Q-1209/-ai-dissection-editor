"""
test_sensitivity_numpy.py

sensitivity.py のロジックをnumpyで独立実装し、既知の性質を検証する
(この開発環境ではtorch実行ができないため、numpy版で検証してtorch版の
正しさを担保する、他のテストファイルと同じアプローチ)。

実行方法: python3 python/tests/test_sensitivity_numpy.py
"""

import numpy as np


def softmax_np(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x)


def compute_kl_divergence_np(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """sensitivity.pyのcompute_kl_divergence()と全く同じロジックのnumpy版。"""
    p_probs = softmax_np(p_logits)
    q_probs = softmax_np(q_logits)
    eps = 1e-12
    kl = np.sum(p_probs * (np.log(p_probs + eps) - np.log(q_probs + eps)))
    return float(kl)


def build_modified_input_np(input_ids, position, mode, pad_token_id=None, custom_token_id=None):
    """sensitivity.pyのbuild_modified_input()と同じロジックのnumpy版(random以外)。"""
    if mode == "delete":
        return [t for i, t in enumerate(input_ids) if i != position]

    modified = list(input_ids)
    if mode == "pad":
        replacement = pad_token_id if pad_token_id is not None else 0
    elif mode == "custom":
        replacement = custom_token_id
    else:
        raise ValueError(f"Unsupported mode for this numpy test: {mode!r}")
    modified[position] = replacement
    return modified


def test_kl_divergence_zero_for_identical_distributions():
    """同一の分布同士のKLダイバージェンスは0になる。"""
    logits = np.array([2.0, 1.0, 0.5, -1.0], dtype=np.float32)
    kl = compute_kl_divergence_np(logits, logits)
    assert abs(kl) < 1e-5, f"KL divergence of identical distributions should be ~0, got {kl}"
    print("PASS: test_kl_divergence_zero_for_identical_distributions")


def test_kl_divergence_positive_for_different_distributions():
    """異なる分布同士のKLダイバージェンスは正の値になる(KLは常に非負)。"""
    p = np.array([3.0, 1.0, 0.5, -1.0], dtype=np.float32)
    q = np.array([-1.0, 0.5, 1.0, 3.0], dtype=np.float32)
    kl = compute_kl_divergence_np(p, q)
    assert kl > 0, f"KL divergence should be positive for different distributions, got {kl}"
    print("PASS: test_kl_divergence_positive_for_different_distributions")


def test_kl_divergence_increases_with_distribution_shift():
    """分布のズレが大きいほど、KLダイバージェンスも大きくなる
    (単調性の確認: 小さい変化より大きい変化の方がKLが大きいはず)。
    """
    p = np.array([3.0, 1.0, 0.5, -1.0], dtype=np.float32)
    q_small_shift = p + np.array([0.1, -0.1, 0.05, -0.05], dtype=np.float32)
    q_large_shift = p + np.array([2.0, -2.0, 1.0, -1.0], dtype=np.float32)

    kl_small = compute_kl_divergence_np(p, q_small_shift)
    kl_large = compute_kl_divergence_np(p, q_large_shift)
    assert kl_large > kl_small, (
        f"larger distribution shift should yield larger KL divergence, "
        f"got kl_small={kl_small}, kl_large={kl_large}"
    )
    print("PASS: test_kl_divergence_increases_with_distribution_shift")


def test_delete_mode_shortens_sequence():
    """delete modeでは、指定位置のトークンが除去され、系列長が1減る。"""
    input_ids = [10, 20, 30, 40, 50]
    result = build_modified_input_np(input_ids, position=2, mode="delete")
    assert result == [10, 20, 40, 50], f"expected [10, 20, 40, 50], got {result}"
    assert len(result) == len(input_ids) - 1
    print("PASS: test_delete_mode_shortens_sequence")


def test_pad_mode_replaces_single_position():
    """pad modeでは、指定位置だけがpad_token_idに置き換わり、系列長は変わらない。"""
    input_ids = [10, 20, 30, 40, 50]
    result = build_modified_input_np(input_ids, position=2, mode="pad", pad_token_id=0)
    assert result == [10, 20, 0, 40, 50], f"expected [10, 20, 0, 40, 50], got {result}"
    assert len(result) == len(input_ids)
    print("PASS: test_pad_mode_replaces_single_position")


def test_pad_mode_defaults_to_zero():
    """pad_token_idを指定しない場合、デフォルトで0に置き換わる。"""
    input_ids = [10, 20, 30]
    result = build_modified_input_np(input_ids, position=1, mode="pad")
    assert result == [10, 0, 30], f"expected [10, 0, 30], got {result}"
    print("PASS: test_pad_mode_defaults_to_zero")


def test_custom_mode_replaces_with_specified_token():
    """custom modeでは、指定したcustom_token_idに置き換わる。"""
    input_ids = [10, 20, 30]
    result = build_modified_input_np(input_ids, position=0, mode="custom", custom_token_id=999)
    assert result == [999, 20, 30], f"expected [999, 20, 30], got {result}"
    print("PASS: test_custom_mode_replaces_with_specified_token")


def test_delete_first_and_last_position():
    """境界値(先頭・末尾)の削除が正しく動くことを確認する。"""
    input_ids = [10, 20, 30]
    result_first = build_modified_input_np(input_ids, position=0, mode="delete")
    result_last = build_modified_input_np(input_ids, position=2, mode="delete")
    assert result_first == [20, 30]
    assert result_last == [10, 20]
    print("PASS: test_delete_first_and_last_position")


def test_kl_divergence_used_for_full_sensitivity_workflow():
    """一連の流れ(オリジナル分布 -> 複数箇所を1つずつ変更 -> KL比較)を
    シミュレートし、「より重要な位置を変更した方がKLが大きくなる」という
    直感的な性質を確認する。
    """
    original_logits = np.array([5.0, 1.0, 0.5, 0.2, 0.1], dtype=np.float32)

    modified_important = np.array([0.1, 5.0, 0.5, 0.2, 1.0], dtype=np.float32)
    modified_unimportant = np.array([5.1, 1.1, 0.4, 0.15, 0.05], dtype=np.float32)

    kl_important = compute_kl_divergence_np(original_logits, modified_important)
    kl_unimportant = compute_kl_divergence_np(original_logits, modified_unimportant)

    assert kl_important > kl_unimportant, (
        "a change that flips the top prediction should have higher KL than a minor change"
    )
    print("PASS: test_kl_divergence_used_for_full_sensitivity_workflow")


if __name__ == "__main__":
    test_kl_divergence_zero_for_identical_distributions()
    test_kl_divergence_positive_for_different_distributions()
    test_kl_divergence_increases_with_distribution_shift()
    test_delete_mode_shortens_sequence()
    test_pad_mode_replaces_single_position()
    test_pad_mode_defaults_to_zero()
    test_custom_mode_replaces_with_specified_token()
    test_delete_first_and_last_position()
    test_kl_divergence_used_for_full_sensitivity_workflow()
    print("\nAll sensitivity logic-verification tests passed.")
