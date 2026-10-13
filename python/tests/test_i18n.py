"""
test_i18n.py

英語UI(2026-10-13追加)のPython側を検証する。torch非依存。
- inference/i18n.py の tr(): 環境変数 AIDE_LANG で日本語/英語が切り替わる(未設定は日本語)
- エラー文が実際に英語になる(エラーを投げる関数を1つ通しで確認)
- python/presets/steering_presets.json の英語版の文例と、日英両方の語彙での語彙カウント
- 子プロセス(Electronと同じ起動の仕方)でもエラー文が英語になる

実行方法: python -m pytest python/tests/test_i18n.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PYTHON_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PYTHON_DIR))

from inference.i18n import current_lang, tr  # noqa: E402
from inference.neuron_search_utils import resolve_target_layers  # noqa: E402
from inference.presets import phrases, preset  # noqa: E402
from inference.refusal_utils import count_refusal_phrases  # noqa: E402
from inference.sycophancy_utils import count_sycophantic_phrases  # noqa: E402


def test_tr_defaults_to_japanese(monkeypatch):
    monkeypatch.delenv("AIDE_LANG", raising=False)
    assert current_lang() == "ja"
    assert tr("日本語", "English") == "日本語"


def test_tr_english_when_env_is_en(monkeypatch):
    monkeypatch.setenv("AIDE_LANG", "en")
    assert current_lang() == "en"
    assert tr("日本語", "English") == "English"
    monkeypatch.setenv("AIDE_LANG", "fr")  # 未知の値は日本語
    assert tr("日本語", "English") == "日本語"


def test_error_message_follows_language(monkeypatch):
    monkeypatch.setenv("AIDE_LANG", "en")
    with pytest.raises(ValueError, match="this model has 4 layers"):
        resolve_target_layers(4, [9])
    monkeypatch.setenv("AIDE_LANG", "ja")
    with pytest.raises(ValueError, match="このモデルは4層"):
        resolve_target_layers(4, [9])


def test_english_presets_match_japanese_shape():
    for name in ("sycophancy", "refusal"):
        ja, en = preset(name), preset(name, "en")
        assert set(ja) == set(en)
        assert len(en["positive_prompts"]) == len(ja["positive_prompts"]) > 0
        assert len(en["negative_prompts"]) == len(ja["negative_prompts"]) > 0
        # 英語の文例に日本語が混ざっていない
        for p in en["positive_prompts"] + en["negative_prompts"]:
            assert p.isascii(), p


def test_phrase_counts_cover_both_languages():
    # 語彙は日英の和集合で、重複しない
    for name in ("sycophancy", "refusal"):
        ps = phrases(name)
        assert len(ps) == len(set(ps))
        assert set(preset(name)["phrases"]) <= set(ps)
        assert set(preset(name, "en")["phrases"]) <= set(ps)
    assert count_sycophantic_phrases("You're absolutely right, what a brilliant idea!")["total"] >= 2
    assert count_refusal_phrases("I'm sorry, but I can't help with that.")["total"] >= 2
    assert count_refusal_phrases("申し訳ありませんが、お答えできません。")["total"] >= 2


def test_subprocess_error_is_english_with_env():
    """Electronと同じく AIDE_LANG を付けて子プロセスを起動すると、エラー文が英語で返る。"""
    request = {"model_dir": ".", "config": {}, "nodes": [], "edges": []}
    proc = subprocess.run(
        [sys.executable, str(PYTHON_DIR / "execute_graph.py")],
        input=json.dumps(request), capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "AIDE_LANG": "en", "PYTHONIOENCODING": "utf-8"}, timeout=120,
    )
    out = json.loads(proc.stdout)
    assert out["success"] is False
    assert out["message"] == "The graph has no nodes.", out
