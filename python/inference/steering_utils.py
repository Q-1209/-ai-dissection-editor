"""
steering_utils.py

Activation Steering Studio(Phase3、2026-09-15着手)の中核ロジックのうち、
torchに依存しない部分(平均ベクトル・差分ベクトルの計算、ノルム計算、スケール適用)を
純粋なPythonで実装する。この開発環境(クラウドサンドボックス)ではtorch実行・実モデル
重みが使えないため、ここでロジックの正しさをnumpyとの比較で担保する方針(他のPhase2
機能と同じ、CODE_REFERENCE.md参照)。

実際のforward pass・テンソル計算(torch依存)は steering.py が担当する。
graph_runner.py の execute()/generate() へのベクトル加算そのもの(ブロードキャスト)は
このモジュールの関数を直接は使わず同等のロジックをtorchテンソルに対して行うが、
「平均差分ベクトルを計算する」「スケールを掛けて加算する」というアルゴリズムの
正しさ自体はここでテストする。
"""

from __future__ import annotations


def mean_vector(vectors: list[list[float]]) -> list[float]:
    """複数のベクトル(同じ次元数)の要素ごとの平均を計算する。

    Activation Steering Studioでは、対照的なプロンプト群(positive/negative)それぞれの
    複数の活性化値ベクトルを1つの代表ベクトル(平均)にまとめるために使う。
    """
    if not vectors:
        raise ValueError("vectors must not be empty")
    dim = len(vectors[0])
    for v in vectors:
        if len(v) != dim:
            raise ValueError("all vectors must have the same dimension")
    sums = [0.0] * dim
    for v in vectors:
        for i, val in enumerate(v):
            sums[i] += val
    return [s / len(vectors) for s in sums]


def diff_vector(positive_mean: list[float], negative_mean: list[float]) -> list[float]:
    """対照的な2群の平均活性化値の差分ベクトルを計算する(positive - negative)。

    これが「ステアリングベクトル」そのもの(2026-09-15、漣さんとの確認事項:
    対照プロンプトの差分方式を採用)。
    """
    if len(positive_mean) != len(negative_mean):
        raise ValueError("positive_mean and negative_mean must have the same dimension")
    return [p - n for p, n in zip(positive_mean, negative_mean)]


def vector_norm(vector: list[float]) -> float:
    """ベクトルのL2ノルム。GUI表示用(ベクトルの大きさの目安を数値で示す)。"""
    return sum(v * v for v in vector) ** 0.5


def apply_scaled_steering(
    hidden_vector: list[float], steering_vector: list[float], scale: float
) -> list[float]:
    """隠れ状態ベクトルに、スケール済みステアリングベクトルを加算する。

    実際のtorch実装(graph_runner.execute内)では、この計算をテンソル全体
    ([batch, seq_len, hidden_size])に対してブロードキャストで行う(全トークン位置・
    バッチに同じベクトルを加算する)。ここでは1トークン分のロジックとして、
    純粋Pythonでテスト可能な形にしている。

    scale=0.0 は「ベクトルを足さない」(ステアリングなし)と数学的に等価。
    scaleの符号を反転させると、差分ベクトルの向きが逆になる
    (positiveの方向に強めるか、negativeの方向に強めるか)。
    """
    if len(hidden_vector) != len(steering_vector):
        raise ValueError("hidden_vector and steering_vector must have the same dimension")
    return [h + scale * s for h, s in zip(hidden_vector, steering_vector)]
