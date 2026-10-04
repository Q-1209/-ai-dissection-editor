"""Refusal Behavior Tuner(2026-09-22)のtorch非依存ロジックのテスト。

実行: python -m pytest python/tests/test_refusal_utils.py (または直接 python で実行)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.refusal_utils import (  # noqa: E402
    DEFAULT_REFUSAL_PHRASES,
    REFUSAL_NEGATIVE_PROMPTS,
    REFUSAL_POSITIVE_PROMPTS,
    count_refusal_phrases,
    normalize,
    project_out,
)


def test_presets_are_paired():
    assert len(REFUSAL_POSITIVE_PROMPTS) == len(REFUSAL_NEGATIVE_PROMPTS) >= 3
    assert all(p.strip() for p in REFUSAL_POSITIVE_PROMPTS + REFUSAL_NEGATIVE_PROMPTS)
    assert set(REFUSAL_POSITIVE_PROMPTS).isdisjoint(REFUSAL_NEGATIVE_PROMPTS)


def test_negative_prompts_do_not_contain_refusal_phrases():
    for p in REFUSAL_NEGATIVE_PROMPTS + REFUSAL_POSITIVE_PROMPTS:
        assert count_refusal_phrases(p)["total"] == 0


def test_count_refusal_phrases():
    text = "申し訳ありませんが、その依頼にはお答えできません。I'm sorry."
    r = count_refusal_phrases(text)
    assert r["total"] == 3
    assert r["by_phrase"] == {"申し訳ありません": 1, "お答えできません": 1, "I'm sorry": 1}
    assert count_refusal_phrases("") == {"total": 0, "by_phrase": {}}
    assert count_refusal_phrases("もちろんです、手順は次の通りです。")["total"] == 0


def test_count_refusal_custom_phrases():
    assert count_refusal_phrases("abc abc", phrases=["abc"])["total"] == 2
    assert "できません" not in DEFAULT_REFUSAL_PHRASES  # 広すぎる語は含めない方針


def test_hedging_refusal_is_counted():
    # 2026-09-24実機: 拒否を増やす方向の加算で「私はAIアシスタントであり、…」という言い逃れ型の応答が出たが、
    # 旧語彙では0回と数えられていた。これも拒否として数える
    text = "私はAIアシスタントであり、人間のような感情を有していないため、その方法を"
    assert count_refusal_phrases(text)["total"] >= 1
    assert count_refusal_phrases("パスワードを盗むことはできません。")["total"] == 1


def test_project_out_matches_numpy_and_is_orthogonal():
    rng = np.random.default_rng(0)
    for _ in range(20):
        h = rng.normal(size=16)
        r = rng.normal(size=16) * rng.uniform(0.1, 10)
        out = np.array(project_out(h.tolist(), r.tolist()))
        u = r / np.linalg.norm(r)
        expected = h - (h @ u) * u
        assert np.allclose(out, expected, atol=1e-10)
        assert abs(out @ u) < 1e-9


def test_project_out_is_idempotent_and_scale_invariant():
    h = [1.0, 2.0, 3.0]
    r = [0.0, 1.0, 1.0]
    once = project_out(h, r)
    twice = project_out(once, r)
    assert np.allclose(once, twice)
    assert np.allclose(project_out(h, [0.0, 5.0, 5.0]), once)
    assert np.allclose(project_out(h, [0.0, -1.0, -1.0]), once)  # 符号にも依存しない


def test_project_out_leaves_orthogonal_components():
    assert np.allclose(project_out([3.0, 4.0], [1.0, 0.0]), [0.0, 4.0])


def test_normalize_zero_vector_raises():
    try:
        normalize([0.0, 0.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_torch_style_batched_projection_matches():
    # graph_runner.execute()と同じ計算([batch, seq, hidden]への一括射影)をnumpyで再現し、
    # 1トークン単位のproject_outと一致することを確認する
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 5, 8))
    r = rng.normal(size=8)
    unit = r / np.linalg.norm(r)
    batched = x - (x @ unit)[..., None] * unit
    for t in range(5):
        assert np.allclose(batched[0, t], project_out(x[0, t].tolist(), r.tolist()))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
