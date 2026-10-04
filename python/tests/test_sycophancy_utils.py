"""
test_sycophancy_utils.py

inference/sycophancy_utils.py の count_sycophantic_phrases()(純粋なPythonロジック、
torch非依存)を検証する。文字列の部分一致カウントだけなので、この開発環境でも実行できる。

実行方法: python3 python/tests/test_sycophancy_utils.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.sycophancy_utils import (  # noqa: E402
    DEFAULT_SYCOPHANTIC_PHRASES,
    SYCOPHANCY_NEGATIVE_PROMPTS,
    SYCOPHANCY_POSITIVE_PROMPTS,
    count_sycophantic_phrases,
)


def test_counts_single_occurrence():
    result = count_sycophantic_phrases("素晴らしいアイデアですね", phrases=["素晴らしい"])
    assert result["total"] == 1
    assert result["by_phrase"] == {"素晴らしい": 1}
    print("PASS: test_counts_single_occurrence")


def test_counts_multiple_occurrences_of_same_phrase():
    text = "素晴らしいですね。本当に素晴らしいと思います。素晴らしい。"
    result = count_sycophantic_phrases(text, phrases=["素晴らしい"])
    assert result["total"] == 3
    assert result["by_phrase"] == {"素晴らしい": 3}
    print("PASS: test_counts_multiple_occurrences_of_same_phrase")


def test_counts_across_multiple_distinct_phrases():
    text = "おっしゃる通りです。さすがです。"
    result = count_sycophantic_phrases(text, phrases=["おっしゃる通り", "さすがです", "ごもっとも"])
    assert result["total"] == 2
    assert result["by_phrase"] == {"おっしゃる通り": 1, "さすがです": 1}
    print("PASS: test_counts_across_multiple_distinct_phrases")


def test_no_match_returns_zero():
    result = count_sycophantic_phrases("今日は晴れです", phrases=["素晴らしい"])
    assert result["total"] == 0
    assert result["by_phrase"] == {}
    print("PASS: test_no_match_returns_zero")


def test_empty_text_returns_zero():
    result = count_sycophantic_phrases("", phrases=["素晴らしい"])
    assert result["total"] == 0
    assert result["by_phrase"] == {}
    print("PASS: test_empty_text_returns_zero")


def test_default_phrase_list_used_when_none_given():
    text = "ごもっともです、素晴らしいと思います。"
    result = count_sycophantic_phrases(text)  # phrases省略 -> DEFAULT_SYCOPHANTIC_PHRASESを使う
    assert result["total"] == 2
    assert result["by_phrase"] == {"ごもっとも": 1, "素晴らしい": 1}
    print("PASS: test_default_phrase_list_used_when_none_given")


def test_default_phrase_list_is_nonempty():
    assert len(DEFAULT_SYCOPHANTIC_PHRASES) > 0
    print("PASS: test_default_phrase_list_is_nonempty")


def test_preset_prompts_are_nonempty_and_distinct_direction():
    # positive(迎合的)例文にはDEFAULT_SYCOPHANTIC_PHRASESの語彙が実際に含まれ、
    # negative(中立的)例文には含まれない(=対照プロンプトとして意味を成す)ことを確認する。
    assert len(SYCOPHANCY_POSITIVE_PROMPTS) > 0
    assert len(SYCOPHANCY_NEGATIVE_PROMPTS) > 0

    positive_joined = "".join(SYCOPHANCY_POSITIVE_PROMPTS)
    negative_joined = "".join(SYCOPHANCY_NEGATIVE_PROMPTS)

    positive_counts = count_sycophantic_phrases(positive_joined)
    negative_counts = count_sycophantic_phrases(negative_joined)

    assert positive_counts["total"] > 0, "positiveプロンプトに迎合語彙が含まれていません"
    assert negative_counts["total"] == 0, "negativeプロンプトに迎合語彙が含まれてしまっています"
    print("PASS: test_preset_prompts_are_nonempty_and_distinct_direction")


if __name__ == "__main__":
    test_counts_single_occurrence()
    test_counts_multiple_occurrences_of_same_phrase()
    test_counts_across_multiple_distinct_phrases()
    test_no_match_returns_zero()
    test_empty_text_returns_zero()
    test_default_phrase_list_used_when_none_given()
    test_default_phrase_list_is_nonempty()
    test_preset_prompts_are_nonempty_and_distinct_direction()
    print("\nAll sycophancy-utils logic-verification tests passed.")
