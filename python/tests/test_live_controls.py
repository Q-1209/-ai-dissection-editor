"""Live Slider(2026-09-24)のtorch非依存ロジック(inference/live_controls.py)のテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.live_controls import (  # noqa: E402
    ControlError,
    Controls,
    ModelLimits,
    NeuronControl,
    build_scale_vector,
    make_sweep_scales,
    neuron_scale_spec,
    normalize_controls,
    override_target,
    validate_command,
)

LIM = ModelLimits(node_ids=frozenset({"model.layers.0", "model.layers.1"}), num_layers=2, intermediate_size=8, num_heads=4)


def test_empty_is_identity():
    c = normalize_controls(None, LIM)
    assert c.is_identity() and c == Controls()


def test_identity_entries_are_dropped():
    c = normalize_controls({"node_scales": {"model.layers.0": 1.0, "model.layers.1": 0.5},
                            "neurons": [{"layer": 0, "kind": "mlp", "index": 3, "scale": 1}]}, LIM)
    assert c.node_scales == {"model.layers.1": 0.5}
    assert c.neurons == ()
    assert not c.is_identity()


def test_steering_scale_alone_is_not_identity():
    assert not normalize_controls({"steering_scale": 2}, LIM).is_identity()


@pytest.mark.parametrize("raw,msg", [
    ({"node_scales": {"lm_head": 0.5}}, "倍率を掛けられない"),
    ({"node_scales": {"model.layers.0": "abc"}}, "数値ではありません"),
    ({"node_scales": {"model.layers.0": float("nan")}}, "有限"),
    ({"node_scales": {"model.layers.0": 11}}, "範囲"),
    ({"neurons": [{"layer": 2, "kind": "mlp", "index": 0, "scale": 0}]}, "層番号"),
    ({"neurons": [{"layer": 0, "kind": "mlp", "index": 8, "scale": 0}]}, "MLPの次元番号"),
    ({"neurons": [{"layer": 0, "kind": "attention_head", "index": 4, "scale": 0}]}, "ヘッド番号"),
    ({"neurons": [{"layer": 0, "kind": "embedding", "index": 0, "scale": 0}]}, "種類"),
    ({"neurons": [{"layer": 0.5, "kind": "mlp", "index": 0, "scale": 0}]}, "整数"),
    ({"steering_scale": -11}, "範囲"),
    ("text", "オブジェクト"),
])
def test_invalid_controls(raw, msg):
    with pytest.raises(ControlError, match=msg):
        normalize_controls(raw, LIM)


def test_duplicate_neuron_last_wins():
    c = normalize_controls({"neurons": [{"layer": 1, "kind": "attention_head", "index": 2, "scale": 0},
                                        {"layer": 1, "kind": "attention_head", "index": 2, "scale": 3}]}, LIM)
    assert c.neurons == (NeuronControl(1, "attention_head", 2, 3.0),)


def test_neuron_scale_spec_and_vector():
    c = normalize_controls({"neurons": [{"layer": 0, "kind": "mlp", "index": 3, "scale": 0},
                                        {"layer": 0, "kind": "mlp", "index": 5, "scale": 2},
                                        {"layer": 1, "kind": "attention_head", "index": 1, "scale": 0.5}]}, LIM)
    spec = neuron_scale_spec(c)
    assert spec == {"model.layers.0.mlp": {3: 0.0, 5: 2.0}, "model.layers.1.self_attn": {1: 0.5}}
    assert build_scale_vector(8, spec["model.layers.0.mlp"]) == [1, 1, 1, 0, 1, 2, 1, 1]


def test_override_target_each_type():
    base = normalize_controls({"node_scales": {"model.layers.0": 0.5}, "steering_scale": 1,
                               "neurons": [{"layer": 0, "kind": "mlp", "index": 1, "scale": 0}]}, LIM)
    assert override_target(base, {"type": "node", "id": "model.layers.1"}, 2).node_scales == {"model.layers.0": 0.5, "model.layers.1": 2}
    assert override_target(base, {"type": "node", "id": "model.layers.0"}, 3).node_scales == {"model.layers.0": 3}
    n = override_target(base, {"type": "neuron", "layer": 0, "kind": "mlp", "index": 1}, 4).neurons
    assert n == (NeuronControl(0, "mlp", 1, 4.0),)
    s = override_target(base, {"type": "steering"}, -2)
    assert s.steering_scale == -2 and s.node_scales == base.node_scales
    with pytest.raises(ControlError):
        override_target(base, {"type": "bogus"}, 1)


def test_sweep_scales():
    assert make_sweep_scales(0, 2, 5) == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert make_sweep_scales(2, -2, 3) == [2.0, 0.0, -2.0]
    assert len(make_sweep_scales(0, 1, 41)) == 41
    for args in [(0, 2, 1), (0, 2, 42), (1, 1, 5), (0, 99, 5)]:
        with pytest.raises(ControlError):
            make_sweep_scales(*args)


def test_validate_command():
    assert validate_command({"cmd": "generate"})["cmd"] == "generate"
    for bad in [{"cmd": "rm"}, {}, "generate", None]:
        with pytest.raises(ControlError):
            validate_command(bad)


def test_to_dict_roundtrip():
    raw = {"node_scales": {"model.layers.0": 0.25}, "steering_scale": 1.5,
           "neurons": [{"layer": 1, "kind": "mlp", "index": 7, "scale": 0.0}]}
    c = normalize_controls(raw, LIM)
    assert normalize_controls(c.to_dict(), LIM) == c
