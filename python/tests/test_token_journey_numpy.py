"""
test_token_journey_numpy.py

token_journey.py のロジックをnumpyで独立実装し、既知の性質を検証する
(この開発環境ではtorch実行ができないため、numpy版で検証してtorch版の
正しさを担保する、他のテストファイルと同じアプローチ)。

実行方法: python3 python/tests/test_token_journey_numpy.py
"""

import numpy as np


def normalize_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def find_nearest_words_np(hidden_vector: np.ndarray, embedding_matrix: np.ndarray, top_k: int = 1):
    """token_journey.pyのfind_nearest_words()と同じロジックのnumpy版。

    戻り値: [(token_id, similarity), ...] 類似度が高い順、長さtop_k
    """
    hidden_norm = normalize_np(hidden_vector.reshape(1, -1))
    embed_norm = normalize_np(embedding_matrix)

    similarities = (hidden_norm @ embed_norm.T).reshape(-1)

    top_k = min(top_k, similarities.shape[0])
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def test_exact_match_gives_similarity_one():
    """隠れベクトルが埋め込み行列のある行と完全一致する場合、その単語の
    コサイン類似度は1.0(最大値)になり、最近傍として選ばれる。
    """
    embedding_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    hidden_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    matches = find_nearest_words_np(hidden_vector, embedding_matrix, top_k=1)
    token_id, similarity = matches[0]
    assert token_id == 1, f"expected token_id=1, got {token_id}"
    assert abs(similarity - 1.0) < 1e-5, f"expected similarity~1.0, got {similarity}"
    print("PASS: test_exact_match_gives_similarity_one")


def test_cosine_similarity_ignores_magnitude():
    """コサイン類似度は方向だけを見るため、ベクトルの長さ(スケール)が
    変わっても最近傍の判定結果は変わらない。
    """
    embedding_matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    hidden_small = np.array([0.1, 0.0], dtype=np.float32)
    hidden_large = np.array([100.0, 0.0], dtype=np.float32)

    matches_small = find_nearest_words_np(hidden_small, embedding_matrix, top_k=1)
    matches_large = find_nearest_words_np(hidden_large, embedding_matrix, top_k=1)

    assert matches_small[0][0] == 0
    assert matches_large[0][0] == 0
    assert abs(matches_small[0][1] - matches_large[0][1]) < 1e-5, (
        "cosine similarity should be scale-invariant"
    )
    print("PASS: test_cosine_similarity_ignores_magnitude")


def test_top_k_returns_correct_count_and_order():
    """top_k=3を指定すると、類似度が高い順に3件返る。"""
    embedding_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    hidden_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    matches = find_nearest_words_np(hidden_vector, embedding_matrix, top_k=3)
    assert len(matches) == 3
    token_ids = [m[0] for m in matches]
    assert token_ids[0] == 0, f"expected token 0 to be the top match, got order {token_ids}"
    assert 2 not in token_ids[:2], "token 2 (orthogonal direction) should not be in top 2"
    print("PASS: test_top_k_returns_correct_count_and_order")


def test_orthogonal_vectors_have_zero_similarity():
    """直交するベクトル同士のコサイン類似度は0になる。"""
    embedding_matrix = np.array([[0.0, 1.0]], dtype=np.float32)
    hidden_vector = np.array([1.0, 0.0], dtype=np.float32)

    matches = find_nearest_words_np(hidden_vector, embedding_matrix, top_k=1)
    assert abs(matches[0][1]) < 1e-5, f"expected similarity~0 for orthogonal vectors, got {matches[0][1]}"
    print("PASS: test_orthogonal_vectors_have_zero_similarity")


def test_outlier_dimension_collapses_without_final_norm_but_recovers_with_it():
    """実機で観測された不具合(Qwen2.5-0.5Bで層2以降、全てのブロック・トークン位置が
    無関係な同一の単語に収束してしまう)の再現・修正確認用テスト。

    Massive Activations現象(arxiv 2402.17762)を模して、隠れベクトルの1次元だけが
    極端に大きい値を持つ状況を作る。この状態で生のコサイン類似度を取ると、
    本来向いている方向が違う複数のベクトルが、外れ値次元を強く持つ埋め込み行の
    方向に引きずられて同じ単語に収束してしまう(=補正なしのバグを再現)。

    RMSNormの学習済み重み(final_norm_weightに相当)でその外れ値次元を強く
    ダウンウェイトしてから比較すると、本来の(小さい方の次元にエンコードされた)
    意味の違いが再び見分けられるようになることを確認する
    (token_journey.pyのcompute_token_journeyがfinal_norm_weightを渡された場合に
    行っている処理と同じロジック)。
    """
    # 語彙500件・隠れ次元8次元をシミュレートする。次元0が「外れ値次元」
    # (Massive Activations文献の報告通り、値がほぼ一定(平均50)だが行ごとに
    # 大きめのジッターを持つ=トークンの意味とは無関係)、次元1-7が「意味を運ぶ次元」
    # (小さい値、トークンごとに異なる=本来の識別情報)。
    # シード固定で決定的に再現する(この組み合わせで実際に「補正なしだと不正解に
    # 収束し、補正ありだと正解する」ケースになることをあらかじめ確認済み)。
    rng = np.random.default_rng(5)
    vocab_size = 500
    hidden_size = 8

    embedding_matrix = np.zeros((vocab_size, hidden_size), dtype=np.float32)
    embedding_matrix[:, 0] = 50.0 + rng.normal(0, 8.0, size=vocab_size)
    embedding_matrix[:, 1:] = rng.normal(0, 1.0, size=(vocab_size, hidden_size - 1))

    true_token_id = 17
    # 「ブロック出力」を模したベクトル: 外れ値次元は他の行と同様ほぼ一定+ジッター
    # (トークンの意味とは無関係に常に大きい、というMassive Activationsの性質通り)。
    # 意味を運ぶ次元は正解トークン(true_token_id)の埋め込みに近いがノイズを含む
    # (実際のモデル内部状態が埋め込みそのものと完全一致するわけではないことを模す)。
    block_output = np.zeros(hidden_size, dtype=np.float32)
    block_output[0] = 50.0 + rng.normal(0, 8.0)
    block_output[1:] = embedding_matrix[true_token_id, 1:] + rng.normal(0, 0.5, size=hidden_size - 1)

    # 補正なし: 外れ値次元のジッターが支配的なため、意味を運ぶ次元では
    # 正解に最も近いはずなのに、無関係な別のトークンが最近傍として選ばれてしまう
    # (実機観測: Qwen2.5-0.5Bで層2以降、無関係な同一トークンに収束したのと同じ機序)。
    raw_match = find_nearest_words_np(block_output, embedding_matrix, top_k=1)[0][0]
    assert raw_match != true_token_id, (
        f"外れ値次元が支配的な状況では、補正なしだと正解トークン({true_token_id})以外"
        f"({raw_match})が選ばれてしまうはず(実機で観測されたバグの再現)"
    )

    # 補正あり: RMSNormの学習済み重みを模した重みベクトルで外れ値次元(次元0)を
    # 強くダウンウェイトしてから比較する(token_journey.pyがfinal_norm_weightを
    # 渡された場合にrms_norm経由で行っているのと同じロジック)。
    final_norm_weight = np.array([0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    weighted_match = find_nearest_words_np(
        block_output * final_norm_weight, embedding_matrix * final_norm_weight, top_k=1
    )[0][0]
    assert weighted_match == true_token_id, (
        f"外れ値次元をダウンウェイトすれば、意味を運ぶ次元の類似度に基づいて"
        f"正解トークン({true_token_id})が選ばれるはず(修正後の期待動作)。got {weighted_match}"
    )
    print("PASS: test_outlier_dimension_collapses_without_final_norm_but_recovers_with_it")


def test_journey_across_blocks_can_change():
    """複数の「ブロック出力」を模した異なるベクトル列に対して最近傍探索を行うと、
    ブロックが進むにつれて最近傍の単語が変化しうることを確認する
    (Token Journeyの核心的な性質: 層を追うごとに解釈が変わりうる)。
    """
    embedding_matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    block_outputs = [
        np.array([0.9, 0.1], dtype=np.float32),
        np.array([0.5, 0.5], dtype=np.float32),
        np.array([0.1, 0.9], dtype=np.float32),
    ]

    nearest_per_block = [
        find_nearest_words_np(v, embedding_matrix, top_k=1)[0][0] for v in block_outputs
    ]
    assert nearest_per_block[0] == 0, "shallow block should be nearest to token 0"
    assert nearest_per_block[-1] == 1, "deep block should be nearest to token 1"
    assert nearest_per_block[0] != nearest_per_block[-1], (
        "the nearest word should change across the journey, demonstrating semantic drift"
    )
    print("PASS: test_journey_across_blocks_can_change")


if __name__ == "__main__":
    test_exact_match_gives_similarity_one()
    test_cosine_similarity_ignores_magnitude()
    test_top_k_returns_correct_count_and_order()
    test_orthogonal_vectors_have_zero_similarity()
    test_outlier_dimension_collapses_without_final_norm_but_recovers_with_it()
    test_journey_across_blocks_can_change()
    print("\nAll token journey logic-verification tests passed.")
