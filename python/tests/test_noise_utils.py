"""noise_utils.py(Noise Injection、2026-10-24追加)のロジックを、numpyの独立実装と比べて確かめる。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.noise_utils import (  # noqa: E402
    DEFAULT_STRENGTHS,
    MAX_STRENGTH,
    add_relative_noise,
    normalize_strengths,
    validate_strength,
)


def add_relative_noise_np(h, eps, s):
    h, eps = np.asarray(h, float), np.asarray(eps, float)
    return h + s * np.linalg.norm(h) * eps / np.linalg.norm(eps)


@pytest.mark.parametrize("s", [0.0, 0.1, 0.5, 2.0])
def test_matches_numpy(s):
    rng = np.random.default_rng(0)
    h, eps = rng.normal(size=16) * 3, rng.normal(size=16)
    assert np.allclose(add_relative_noise(h.tolist(), eps.tolist(), s), add_relative_noise_np(h, eps, s))


def test_noise_size_is_strength_times_hidden_norm():
    rng = np.random.default_rng(1)
    h, eps = rng.normal(size=32) * 7, rng.normal(size=32) * 0.01  # epsの大きさは結果に影響しない
    out = np.array(add_relative_noise(h.tolist(), eps.tolist(), 0.3))
    assert np.isclose(np.linalg.norm(out - h), 0.3 * np.linalg.norm(h))


def test_zero_strength_returns_hidden():
    assert add_relative_noise([1.0, -2.0], [5.0, 5.0], 0.0) == [1.0, -2.0]


def test_dimension_mismatch():
    with pytest.raises(ValueError):
        add_relative_noise([1.0], [1.0, 2.0], 0.1)


@pytest.mark.parametrize("bad", [-0.1, MAX_STRENGTH + 1, float("nan"), float("inf")])
def test_validate_strength_rejects(bad):
    with pytest.raises(ValueError):
        validate_strength(bad)


def test_normalize_strengths():
    assert normalize_strengths(None) == DEFAULT_STRENGTHS
    assert normalize_strengths([0.5, 0.1, 0.5, 0]) == [0.0, 0.1, 0.5]
    with pytest.raises(ValueError):
        normalize_strengths([0.1, -1])
