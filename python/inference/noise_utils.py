"""
noise_utils.py

Noise Injection(2026-10-24追加、長編動画M4「AIを酔っぱらわせろ」用)の中核ロジックのうち、
torchに依存しない部分。他の*_utils.pyと同じく、numpyの独立実装と比べてテストする。

2026-10-24、漣さんとの確認事項:
- 置き場所: 観察タブの新パネル(Layer Ablation Wizardと同じ、層のチェックボックス→実行→通常時との比較)
- 強さ: その位置の内部状態の大きさに対する割合(0.1 = 10%)。層ごとに値の大きさが違っても公平に比べられる
- スイープ: 強さを段階的に上げる比較と、同じ強さで各層を1つずつ試す比較の両方
- 乱数: 1語(生成の1ステップ)ごとに新しい雑音。乱数の種(seed)を固定するので、同じ設定なら同じ結果

実際のtorchでの加算は graph_runner.execute() が、この add_relative_noise と同じ式でテンソル全体
([batch, seq_len, hidden_size]、各位置ごとに別の雑音)に対して行う。
"""

from __future__ import annotations

import math

# 強さスイープの既定値(0は通常時と同じ。グラフの起点として入れる)
DEFAULT_STRENGTHS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
MAX_STRENGTH = 10.0
VALID_MODES = ("single", "strength_sweep", "layer_sweep")


def add_relative_noise(hidden: list[float], eps: list[float], strength: float) -> list[float]:
    """1トークン分の内部状態 hidden に、大きさ strength × ‖hidden‖ の雑音を足す。

    eps は雑音の向き(正規分布から引いたベクトル。大きさは問わず、ここで単位ベクトルに直す)。
    strength=0 なら hidden をそのまま返す(雑音なしと同じ)。足した雑音の大きさは
    ちょうど strength × ‖hidden‖ になる(向きだけがランダム)。
    """
    if len(hidden) != len(eps):
        raise ValueError("hidden and eps must have the same dimension")
    h_norm = math.sqrt(sum(v * v for v in hidden))
    e_norm = max(math.sqrt(sum(v * v for v in eps)), 1e-12)
    k = strength * h_norm / e_norm
    return [h + k * e for h, e in zip(hidden, eps)]


def validate_strength(strength: float) -> float:
    """強さが 0 以上 MAX_STRENGTH 以下の有限の数か確かめて、floatで返す。"""
    s = float(strength)
    if not math.isfinite(s) or s < 0 or s > MAX_STRENGTH:
        raise ValueError(f"strength must be between 0 and {MAX_STRENGTH} (got {strength!r})")
    return s


def normalize_strengths(strengths: list[float] | None) -> list[float]:
    """強さスイープの値の並びを検証し、重複を除いて小さい順に並べる。省略時は DEFAULT_STRENGTHS。"""
    if not strengths:
        return list(DEFAULT_STRENGTHS)
    return sorted({validate_strength(s) for s in strengths})
