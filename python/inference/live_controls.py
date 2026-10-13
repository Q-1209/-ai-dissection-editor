"""
live_controls.py

Phase3: スライダーによるリアルタイム出力操作(Live Slider、2026-09-24着手)の、torchに依存しない部分。
スライダーの値(コントロール)の検証・正規化、スイープ用の倍率列の生成、コマンドの検証を行う。

2026-09-24、漣さんとの確認事項:
- 操作対象: すべて(切り替え可能)
  - 層・ブロックの倍率(node_scales): そのブロックが足した分(残差への寄与)に倍率を掛ける。1=通常、0=素通り、2=強調
  - 個別ニューロンの倍率(neurons): MLP中間層の1次元、またはAttentionの1ヘッドの出力に倍率を掛ける
  - ステアリングの強さ(steering_scale): Activation Steeringのベクトルを何倍で足すか
- 反映方式: 生成中に反映。スライダーを動かすと、次に生成されるトークンから新しい値が使われる
  (それまでに生成したトークンの計算結果=KVキャッシュはそのまま)
- 表示: 通常時と操作後の文章・数値の比較に加え、倍率を刻んだときのKLの変化グラフ(スイープ)も出す

コントロールの形式(JSON、GUIからそのまま届く):
{
  "node_scales": {"model.layers.13": 1.5, ...},
  "neurons": [{"layer": 13, "kind": "mlp" | "attention_head", "index": 120, "scale": 0.0}, ...],
  "steering_scale": 0.0
}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from .i18n import tr

SCALE_MIN = -10.0
SCALE_MAX = 10.0
NEURON_KINDS = ("mlp", "attention_head")
VALID_COMMANDS = ("init", "generate", "set_controls", "stop", "sweep", "set_steering", "shutdown")


class ControlError(ValueError):
    """コントロール・コマンドの内容が不正な場合。メッセージはそのままGUIに表示される。"""


@dataclass(frozen=True)
class NeuronControl:
    layer: int
    kind: str  # "mlp" | "attention_head"
    index: int
    scale: float


@dataclass(frozen=True)
class Controls:
    node_scales: dict[str, float] = field(default_factory=dict)
    neurons: tuple[NeuronControl, ...] = ()
    steering_scale: float = 0.0

    def is_identity(self) -> bool:
        """何も変えない(通常時と同じ計算になる)コントロールか。"""
        return (
            all(s == 1.0 for s in self.node_scales.values())
            and all(n.scale == 1.0 for n in self.neurons)
            and self.steering_scale == 0.0
        )

    def to_dict(self) -> dict:
        return {
            "node_scales": dict(self.node_scales),
            "neurons": [
                {"layer": n.layer, "kind": n.kind, "index": n.index, "scale": n.scale} for n in self.neurons
            ],
            "steering_scale": self.steering_scale,
        }


@dataclass(frozen=True)
class ModelLimits:
    """コントロールの検証に使うモデルの寸法。"""

    node_ids: frozenset[str]  # 倍率を掛けられるノード(attention_block/mlp_block/layer_block)
    num_layers: int
    intermediate_size: int
    num_heads: int


def _scale(value, what: str) -> float:
    try:
        s = float(value)
    except (TypeError, ValueError):
        raise ControlError(tr(f"{what} の倍率が数値ではありません: {value!r}", f"Scale for {what} is not a number: {value!r}")) from None
    if not math.isfinite(s):
        raise ControlError(tr(f"{what} の倍率が有限の数値ではありません: {value!r}", f"Scale for {what} is not a finite number: {value!r}"))
    if not (SCALE_MIN <= s <= SCALE_MAX):
        raise ControlError(tr(f"{what} の倍率は {SCALE_MIN}〜{SCALE_MAX} の範囲で指定してください(指定値: {s})", f"Scale for {what} must be between {SCALE_MIN} and {SCALE_MAX} (got {s})"))
    return s


def _int(value, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ControlError(tr(f"{what} は整数で指定してください(指定値: {value!r})", f"{what} must be an integer (got {value!r})"))
    return int(value)


def normalize_controls(raw: dict | None, limits: ModelLimits) -> Controls:
    """GUIから届いたコントロールを検証し、Controlsに正規化する。不正ならControlError。

    - 未知のノードID、範囲外の層・次元・ヘッド番号、範囲外の倍率はエラー
    - 同じニューロン(層・種類・番号)が複数あれば後のものを採用する
    - 倍率1(何も変えない)のノード・ニューロンはここで取り除く(計算を軽くするため)
    """
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ControlError(tr("controls はオブジェクト(辞書)で指定してください。", "controls must be an object (dict)."))

    node_scales: dict[str, float] = {}
    for node_id, value in (raw.get("node_scales") or {}).items():
        if node_id not in limits.node_ids:
            raise ControlError(
                tr(f"倍率を掛けられないノードです: {node_id}(attention/MLP/層ブロックのみ。グラフ内に存在する必要があります)", f"Cannot scale node {node_id} (only attention, MLP and layer blocks that exist in the graph)")
            )
        s = _scale(value, node_id)
        if s != 1.0:
            node_scales[node_id] = s

    neurons: dict[tuple[int, str, int], NeuronControl] = {}
    for item in raw.get("neurons") or []:
        if not isinstance(item, dict):
            raise ControlError(tr("neurons の各要素はオブジェクトで指定してください。", "Each item in neurons must be an object."))
        kind = item.get("kind")
        if kind not in NEURON_KINDS:
            raise ControlError(tr(f"ニューロンの種類は {NEURON_KINDS} のいずれかです(指定値: {kind!r})", f"Neuron kind must be one of {NEURON_KINDS} (got {kind!r})"))
        layer = _int(item.get("layer"), tr("層番号", "layer index"))
        if not (0 <= layer < limits.num_layers):
            raise ControlError(tr(f"層番号は 0〜{limits.num_layers - 1} です(指定値: {layer})", f"Layer index must be 0-{limits.num_layers - 1} (got {layer})"))
        index = _int(item.get("index"), tr("ニューロン番号", "neuron index"))
        upper = limits.intermediate_size if kind == "mlp" else limits.num_heads
        if not (0 <= index < upper):
            label = tr("MLPの次元番号", "MLP dimension index") if kind == "mlp" else tr("ヘッド番号", "head index")
            raise ControlError(tr(f"{label}は 0〜{upper - 1} です(指定値: {index})", f"{label} must be 0-{upper - 1} (got {index})"))
        s = _scale(item.get("scale", 1.0), f"L{layer} {kind} #{index}")
        neurons[(layer, kind, index)] = NeuronControl(layer, kind, index, s)
    neuron_list = tuple(n for n in neurons.values() if n.scale != 1.0)

    steering_scale = _scale(raw.get("steering_scale", 0.0), tr("ステアリング", "steering"))
    return Controls(node_scales=node_scales, neurons=neuron_list, steering_scale=steering_scale)


def neuron_scale_spec(controls: Controls) -> dict[str, dict[int, float]]:
    """ニューロン倍率を、graph_runner.execute(neuron_scales=...)のキーごとにまとめる。

    戻り値: {"model.layers.N.mlp": {次元番号: 倍率}, "model.layers.N.self_attn": {ヘッド番号: 倍率}}
    (実際のtorchの倍率ベクトルへの変換はlive_session.pyが行う)
    """
    spec: dict[str, dict[int, float]] = {}
    for n in controls.neurons:
        suffix = "mlp" if n.kind == "mlp" else "self_attn"
        spec.setdefault(f"model.layers.{n.layer}.{suffix}", {})[n.index] = n.scale
    return spec


def build_scale_vector(size: int, entries: dict[int, float]) -> list[float]:
    """長さsizeの倍率ベクトル(既定1.0)に、指定した番号だけ倍率を入れる。"""
    vec = [1.0] * size
    for i, s in entries.items():
        vec[i] = s
    return vec


def override_target(controls: Controls, target: dict, value: float) -> Controls:
    """スイープ用: 現在のコントロールのうち、targetで指定した1つだけをvalueに置き換えたものを返す。

    target: {"type": "node", "id": ...} / {"type": "neuron", "layer", "kind", "index"} / {"type": "steering"}
    """
    t = (target or {}).get("type")
    if t == "node":
        node_id = target.get("id")
        scales = dict(controls.node_scales)
        scales[node_id] = value
        return Controls(scales, controls.neurons, controls.steering_scale)
    if t == "neuron":
        key = (int(target.get("layer")), target.get("kind"), int(target.get("index")))
        others = tuple(n for n in controls.neurons if (n.layer, n.kind, n.index) != key)
        return Controls(controls.node_scales, others + (NeuronControl(key[0], key[1], key[2], value),), controls.steering_scale)
    if t == "steering":
        return Controls(controls.node_scales, controls.neurons, value)
    raise ControlError(tr(f"スイープ対象の type は node / neuron / steering のいずれかです(指定値: {t!r})", f"Sweep target type must be node, neuron or steering (got {t!r})"))


def make_sweep_scales(start: float, stop: float, steps: int) -> list[float]:
    """start〜stopを等間隔にsteps点(両端を含む)。表示用に小数第4位で丸める。"""
    steps = int(steps)
    if steps < 2 or steps > 41:
        raise ControlError(tr("スイープの点数は 2〜41 で指定してください。", "The number of sweep steps must be between 2 and 41."))
    start, stop = _scale(start, tr("スイープ開始", "sweep start")), _scale(stop, tr("スイープ終了", "sweep stop"))
    if start == stop:
        raise ControlError(tr("スイープの開始と終了が同じです。", "Sweep start and stop are the same."))
    return [round(start + (stop - start) * i / (steps - 1), 4) for i in range(steps)]


def validate_command(msg) -> dict:
    """stdinから届いた1行分のコマンドを検証する。"""
    if not isinstance(msg, dict) or msg.get("cmd") not in VALID_COMMANDS:
        raise ControlError(tr(f"不明なコマンドです: {msg!r}(使えるもの: {', '.join(VALID_COMMANDS)})", f"Unknown command: {msg!r} (available: {', '.join(VALID_COMMANDS)})"))
    return msg

