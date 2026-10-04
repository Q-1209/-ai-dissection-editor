"""
test_steering_numpy.py

inference/steering_utils.py (Activation Steering Studioの中核ロジック、torch非依存)を
numpyでの独立計算と突き合わせて検証する(この開発環境ではtorch実行ができないため、
ロジックの正しさをここで担保する、他のPhase2機能テストと同じアプローチ)。

実行方法: python3 python/tests/test_steering_numpy.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.steering_utils import (  # noqa: E402
    apply_scaled_steering,
    diff_vector,
    mean_vector,
    vector_norm,
)


def test_mean_vector_matches_numpy():
    vectors = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 0.0, 1.0]]
    expected = np.mean(np.array(vectors), axis=0)
    result = mean_vector(vectors)
    assert np.allclose(result, expected)
    print("PASS: test_mean_vector_matches_numpy")


def test_mean_vector_single_vector_is_identity():
    vectors = [[1.5, -2.5, 0.0]]
    assert np.allclose(mean_vector(vectors), vectors[0])
    print("PASS: test_mean_vector_single_vector_is_identity")


def test_mean_vector_rejects_empty():
    try:
        mean_vector([])
    except ValueError:
        print("PASS: test_mean_vector_rejects_empty")
        return
    raise AssertionError("expected ValueError for empty vectors")


def test_mean_vector_rejects_dimension_mismatch():
    try:
        mean_vector([[1.0, 2.0], [1.0, 2.0, 3.0]])
    except ValueError:
        print("PASS: test_mean_vector_rejects_dimension_mismatch")
        return
    raise AssertionError("expected ValueError for dimension mismatch")


def test_diff_vector_matches_numpy():
    pos = [4.0, 5.0, 6.0]
    neg = [1.0, 2.0, 2.0]
    expected = np.array(pos) - np.array(neg)
    assert np.allclose(diff_vector(pos, neg), expected)
    print("PASS: test_diff_vector_matches_numpy")


def test_diff_vector_zero_when_equal():
    v = [1.0, 2.0, 3.0]
    assert np.allclose(diff_vector(v, v), [0.0, 0.0, 0.0])
    print("PASS: test_diff_vector_zero_when_equal")


def test_diff_vector_rejects_dimension_mismatch():
    try:
        diff_vector([1.0, 2.0], [1.0, 2.0, 3.0])
    except ValueError:
        print("PASS: test_diff_vector_rejects_dimension_mismatch")
        return
    raise AssertionError("expected ValueError for dimension mismatch")


def test_vector_norm_matches_numpy():
    v = [3.0, 4.0]
    assert np.isclose(vector_norm(v), np.linalg.norm(v))  # 3-4-5の直角三角形、norm=5
    print("PASS: test_vector_norm_matches_numpy")


def test_vector_norm_zero_vector():
    assert vector_norm([0.0, 0.0, 0.0]) == 0.0
    print("PASS: test_vector_norm_zero_vector")


def test_apply_scaled_steering_matches_numpy():
    hidden = [1.0, 2.0, 3.0]
    steer = [0.5, -1.0, 2.0]
    scale = 3.0
    expected = np.array(hidden) + scale * np.array(steer)
    assert np.allclose(apply_scaled_steering(hidden, steer, scale), expected)
    print("PASS: test_apply_scaled_steering_matches_numpy")


def test_apply_scaled_steering_zero_scale_is_noop():
    hidden = [1.0, 2.0, 3.0]
    steer = [9.0, -9.0, 9.0]
    assert np.allclose(apply_scaled_steering(hidden, steer, 0.0), hidden)
    print("PASS: test_apply_scaled_steering_zero_scale_is_noop")


def test_apply_scaled_steering_negative_scale_flips_direction():
    hidden = [0.0, 0.0]
    steer = [1.0, -1.0]
    positive = apply_scaled_steering(hidden, steer, 2.0)
    negative = apply_scaled_steering(hidden, steer, -2.0)
    assert np.allclose(positive, [2.0, -2.0])
    assert np.allclose(negative, [-2.0, 2.0])
    print("PASS: test_apply_scaled_steering_negative_scale_flips_direction")


def test_apply_scaled_steering_rejects_dimension_mismatch():
    try:
        apply_scaled_steering([1.0, 2.0], [1.0, 2.0, 3.0], 1.0)
    except ValueError:
        print("PASS: test_apply_scaled_steering_rejects_dimension_mismatch")
        return
    raise AssertionError("expected ValueError for dimension mismatch")


def test_end_to_end_contrastive_prompt_pipeline():
    """positive/negativeプロンプト群それぞれの複数活性化ベクトルから、差分ベクトルを
    計算し、それをスケール適用して隠れ状態に加算するまでの一連の流れを、
    numpyで独立に計算した結果と突き合わせる(実際のcompute_steering_vectors()相当の
    流れを、ロジックレベルで検証)。"""
    positive_acts = [[2.0, 4.0, 0.0], [4.0, 6.0, 2.0]]  # 2プロンプト分
    negative_acts = [[0.0, 0.0, 0.0], [2.0, 2.0, -2.0]]

    pos_mean = mean_vector(positive_acts)
    neg_mean = mean_vector(negative_acts)
    steering_vector = diff_vector(pos_mean, neg_mean)

    expected_pos_mean = np.mean(np.array(positive_acts), axis=0)
    expected_neg_mean = np.mean(np.array(negative_acts), axis=0)
    expected_steering_vector = expected_pos_mean - expected_neg_mean
    assert np.allclose(steering_vector, expected_steering_vector)

    hidden = [1.0, 1.0, 1.0]
    scale = 2.0
    steered = apply_scaled_steering(hidden, steering_vector, scale)
    expected_steered = np.array(hidden) + scale * expected_steering_vector
    assert np.allclose(steered, expected_steered)
    print("PASS: test_end_to_end_contrastive_prompt_pipeline")


if __name__ == "__main__":
    test_mean_vector_matches_numpy()
    test_mean_vector_single_vector_is_identity()
    test_mean_vector_rejects_empty()
    test_mean_vector_rejects_dimension_mismatch()
    test_diff_vector_matches_numpy()
    test_diff_vector_zero_when_equal()
    test_diff_vector_rejects_dimension_mismatch()
    test_vector_norm_matches_numpy()
    test_vector_norm_zero_vector()
    test_apply_scaled_steering_matches_numpy()
    test_apply_scaled_steering_zero_scale_is_noop()
    test_apply_scaled_steering_negative_scale_flips_direction()
    test_apply_scaled_steering_rejects_dimension_mismatch()
    test_end_to_end_contrastive_prompt_pipeline()
    print("\nAll activation-steering logic-verification tests passed.")
