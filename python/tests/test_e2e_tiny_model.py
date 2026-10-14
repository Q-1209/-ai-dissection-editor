"""
test_e2e_tiny_model.py — 極小モデルによるend-to-endテスト(2026-09-22追加)。

実モデルなし・CPUのみで、PyTorch実装の実行経路を一通り動かす。torch/tokenizers/jinja2の
いずれかが無い環境では全テストをスキップする(漣さんのWindows環境・クラウド環境の両方で実行できる)。

実行: python -m pytest python/tests/test_e2e_tiny_model.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

PYTHON_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tiny_model import IM_END_ID, build_tiny_model, default_graph_request  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    d = Path(tempfile.mkdtemp()) / "tiny"
    config = build_tiny_model(d)
    return d, config, default_graph_request(d, config)


# GUI(electron/main.ts)と同じく、子プロセスの標準入出力をUTF-8に固定する(Windowsの既定はcp932)
UTF8_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def run_entry(script: str, request: dict) -> dict:
    """エントリポイントをGUIと同じ形(stdin JSON→stdout JSON)で実行する。"""
    proc = subprocess.run(
        [sys.executable, str(PYTHON_DIR / script)],
        input=json.dumps(request), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=300, env=UTF8_ENV,
    )
    assert proc.stdout, proc.stderr
    return json.loads(proc.stdout)


STEER = {
    "positive_prompts": ["bad thing", "evil act"],
    "negative_prompts": ["good thing", "kind act"],
    "target_node_ids": ["model.layers.1"],
    "input_text": "hello",
    "max_new_tokens": 8,
}


@pytest.mark.parametrize("use_chat_template", [False, True])
@pytest.mark.parametrize("script,extra", [
    ("execute_graph.py", {}),
    ("inspect_internals.py", {}),
    ("analyze_sensitivity.py", {}),
    ("compute_token_journey.py", {}),
    ("run_neuron_search.py", {}),
    ("run_layer_ablation.py", {"ablate_node_ids": ["model.layers.0"]}),
])
def test_all_entry_points_run(tiny, script, extra, use_chat_template):
    _, _, base = tiny
    r = run_entry(script, {**base, "input_text": "hi", "max_new_tokens": 5, "use_chat_template": use_chat_template, **extra})
    assert r["success"], r


def test_battle_view_runs(tiny):
    d, config, _ = tiny
    r = run_entry("run_battle_view.py", {"models": [{"model_dir": str(d), "config": config, "label": "a"}],
                                         "input_text": "hi", "use_chat_template": True, "max_new_tokens": 5})
    assert r["success"], r
    assert r["results"][0]["success"], r


@pytest.mark.parametrize("mode", ["add", "ablate_direction"])
@pytest.mark.parametrize("use_chat_template", [False, True])
def test_steering_modes(tiny, mode, use_chat_template):
    _, _, base = tiny
    r = run_entry("run_activation_steering.py", {**base, **STEER, "scale": 1.0, "intervention_mode": mode,
                                                  "use_chat_template": use_chat_template,
                                                  "measure_refusal": True, "measure_sycophancy": True})
    assert r["success"], r
    assert r["intervention_mode"] == mode and r["used_chat_template"] == use_chat_template
    assert r["kl_divergence"] >= 0
    assert r["refusal_phrase_count_baseline"] is not None and r["sycophancy_phrase_count_steered"] is not None


def test_invalid_intervention_mode_is_rejected(tiny):
    _, _, base = tiny
    r = run_entry("run_activation_steering.py", {**base, **STEER, "intervention_mode": "bogus"})
    assert r["success"] is False and r["error_type"] == "GraphValidationError"


def test_missing_chat_template_gives_clear_error():
    d = Path(tempfile.mkdtemp()) / "base_model"
    config = build_tiny_model(d, with_chat_template=False)
    r = run_entry("execute_graph.py", {**default_graph_request(d, config), "input_text": "hi", "use_chat_template": True})
    assert r["success"] is False and r["error_type"] == "TokenizerLoadError"
    assert "chat_template" in r["message"]


def _executor(tiny):
    from inference.graph_runner import LlamaGraphExecutor  # noqa: PLC0415
    from inference.graph_validation import GraphNode  # noqa: PLC0415

    d, config, base = tiny
    nodes = [GraphNode(id=n["id"], role=n["role"], layer_index=n["layer_index"]) for n in base["nodes"]]
    edges = [(e["source"], e["target"]) for e in base["edges"]]
    ex = LlamaGraphExecutor(d, config)
    ex.load_weights_for_nodes([n.id for n in nodes])
    return ex, nodes, edges


def test_direction_ablation_removes_component(tiny):
    """方向除去後の対象ノード出力は、方向ベクトルとほぼ直交する(内積≈0)。"""
    ex, nodes, edges = _executor(tiny)
    sub_ids = {"model.embed_tokens", "model.layers.0", "model.layers.1"}
    sub = [n for n in nodes if n.id in sub_ids]
    sub_edges = [(a, b) for a, b in edges if a in sub_ids and b in sub_ids]
    ids = torch.tensor([[5, 6, 7, 8]])
    direction = torch.randn(16)
    unit = direction / direction.norm()
    kw = dict(entry_node_id="model.embed_tokens", exit_node_id="model.layers.1", kv_cache={}, position_offset=0)
    before = ex.execute(ids, sub, sub_edges, **kw).logits.float()
    after = ex.execute(ids, sub, sub_edges, **{**kw, "kv_cache": {}},
                       ablation_directions={"model.layers.1": direction}).logits.float()
    assert float((before @ unit).abs().max()) > 1e-3
    assert float((after @ unit).abs().max()) < 1e-4


def test_stop_token_ends_generation(tiny):
    ex, nodes, edges = _executor(tiny)
    ids = torch.tensor([[5, 6, 7]])
    kw = dict(entry_node_id="model.embed_tokens", exit_node_id="lm_head", max_new_tokens=10)
    full = ex.generate(ids, nodes, edges, **kw).generated_token_ids
    stop = full[3]
    cut = ex.generate(ids, nodes, edges, **kw, stop_token_ids=[stop]).generated_token_ids
    assert cut == full[: full.index(stop)]
    assert IM_END_ID not in ex.generate(ids, nodes, edges, **kw, stop_token_ids=[IM_END_ID]).generated_token_ids


def test_phase3_check_script_runs(tiny):
    """run_phase3_check.pyが極小モデルで最後まで走り、MarkdownとJSONを保存すること。"""
    d, _, _ = tiny
    out = Path(tempfile.mkdtemp())
    proc = subprocess.run(
        [sys.executable, str(PYTHON_DIR / "run_phase3_check.py"), str(d), "--max-new-tokens", "5", "--out-dir", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        # 親(pytest)側の環境がUTF-8でない場合も、スクリプト自身が子プロセスをUTF-8に揃えることを確かめる
        env={**os.environ, "PYTHONIOENCODING": "cp932:surrogateescape"},
    )
    assert proc.returncode == 0, proc.stderr
    mds = list(out.glob("*-phase3-check.md"))
    jsons = list(out.glob("*-phase3-check.json"))
    assert len(mds) == 1 and len(jsons) == 1
    results = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert len(results) == 6
    assert all(r["result"]["success"] for r in results), results
    assert "エラー" not in mds[0].read_text(encoding="utf-8")
    # 日本語の入力文が文字化けせずに渡っていること(Windowsでの文字化け不具合の再発防止、2026-09-22)。
    # 極小モデルの語彙に日本語は無いので[UNK]になるが、「こんにちは」の5文字=5トークン(+BOS)になるはず
    first = results[0]["result"]
    assert len(first["input_token_ids"]) == 1 + len("こんにちは")


# ---------------------------------------------------------------------------
# Live Slider(2026-09-24追加)
# ---------------------------------------------------------------------------

def _logits(ex, nodes, edges, ids, **kw):
    return ex.execute(ids, nodes, edges, entry_node_id="model.embed_tokens", exit_node_id="lm_head",
                      kv_cache={}, position_offset=0, **kw).logits.float()


def test_node_scale_one_is_exact_identity_and_zero_skips_block(tiny):
    ex, nodes, edges = _executor(tiny)
    ids = torch.tensor([[5, 6, 7, 8]])
    base = _logits(ex, nodes, edges, ids)
    assert torch.equal(base, _logits(ex, nodes, edges, ids, node_scales={"model.layers.1": 1.0}))
    # 倍率0 = その層を素通り = グラフからその層を外して前後をつないだのと同じ
    skipped = _logits(ex, nodes, edges, ids, node_scales={"model.layers.1": 0.0})
    rewired_nodes = [n for n in nodes if n.id != "model.layers.1"]
    rewired_edges = [(a, b) for a, b in edges if "model.layers.1" not in (a, b)] + [("model.layers.0", "model.layers.2")]
    assert torch.allclose(skipped, _logits(ex, rewired_nodes, rewired_edges, ids), atol=1e-5)
    # 倍率2は寄与を2倍にする(通常時とも素通りとも違う)
    doubled = _logits(ex, nodes, edges, ids, node_scales={"model.layers.1": 2.0})
    assert not torch.allclose(doubled, base) and not torch.allclose(doubled, skipped)


def test_neuron_scales(tiny):
    ex, nodes, edges = _executor(tiny)
    ids = torch.tensor([[5, 6, 7, 8]])
    base = _logits(ex, nodes, edges, ids)
    ones = {"model.layers.1.mlp": torch.ones(32), "model.layers.1.self_attn": torch.ones(2)}
    assert torch.allclose(base, _logits(ex, nodes, edges, ids, neuron_scales=ones), atol=1e-6)
    # 全ヘッド0かつ全MLP次元0 = 層1が何も足さない = 層1の倍率0と同じ
    zeros = {"model.layers.1.mlp": torch.zeros(32), "model.layers.1.self_attn": torch.zeros(2)}
    assert torch.allclose(_logits(ex, nodes, edges, ids, neuron_scales=zeros),
                          _logits(ex, nodes, edges, ids, node_scales={"model.layers.1": 0.0}), atol=1e-5)
    # 1次元だけ0にすると変化する
    one_dim = torch.ones(32)
    one_dim[3] = 0.0
    assert not torch.allclose(base, _logits(ex, nodes, edges, ids, neuron_scales={"model.layers.1.mlp": one_dim}))


def _live_session(tiny):
    from inference.graph_validation import GraphNode  # noqa: PLC0415
    from inference.live_session import LiveSession  # noqa: PLC0415
    from inference.tokenizer_utils import load_tokenizer  # noqa: PLC0415

    d, config, base = tiny
    nodes = [GraphNode(id=n["id"], role=n["role"], layer_index=n["layer_index"]) for n in base["nodes"]]
    edges = [(e["source"], e["target"]) for e in base["edges"]]
    return LiveSession(d, config, nodes, edges, "model.embed_tokens", "lm_head", load_tokenizer(d), use_chat_template=False)


def test_live_generate_applies_controls_mid_generation(tiny):
    from inference.live_controls import Controls  # noqa: PLC0415
    from inference.sampling import SamplingConfig  # noqa: PLC0415

    s = _live_session(tiny)
    events = []
    calls = {"n": 0}
    strong = Controls(node_scales={"model.layers.1": 3.0, "model.layers.2": 0.0})

    def get_controls():
        calls["n"] += 1
        return (0, Controls()) if calls["n"] <= 3 else (1, strong)  # 4トークン目から操作

    s.generate("hello", 8, SamplingConfig(temperature=0.0), get_controls, lambda: False, events.append)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "started" and kinds[1] == "baseline" and kinds[-1] == "done"
    tokens = [e for e in events if e["event"] == "token"]
    assert len(tokens) == 8
    assert all(t["kl"] == 0.0 for t in tokens[:3]) and all(t["controls_version"] == 0 for t in tokens[:3])
    assert all(t["controls_version"] == 1 for t in tokens[3:])
    assert max(t["kl"] for t in tokens[3:]) > 1e-4
    # コントロールなしの区間は通常時の文章と一致する
    baseline = events[1]["text"]
    assert baseline.startswith(tokens[2]["text"])


def test_live_sweep(tiny):
    from inference.live_controls import Controls  # noqa: PLC0415

    s = _live_session(tiny)
    points, base = s.sweep("hello", {"type": "node", "id": "model.layers.1"}, [0.0, 1.0, 2.0], Controls())
    assert [p["scale"] for p in points] == [0.0, 1.0, 2.0]
    assert points[1]["kl"] < 1e-6  # 倍率1 = 通常時
    assert points[0]["kl"] > 0 and points[2]["kl"] > 0
    assert "baseline_top_token_text" in base


def test_live_server_protocol_and_stop(tiny, capsys):
    """run_live_session.pyのServerを同じプロセスで動かし、コマンド列→イベント列を確認する。"""
    import queue  # noqa: PLC0415

    import run_live_session as rls  # noqa: PLC0415

    d, config, base = tiny
    inbox: queue.Queue = queue.Queue()
    for msg in [
        {"cmd": "generate", "input_text": "hi"},  # init前 → エラー
        {"cmd": "init", **{k: base[k] for k in ("model_dir", "config", "nodes", "edges")}},
        {"cmd": "set_controls", "controls": {"node_scales": {"lm_head": 2}}},  # 不正 → エラー
        {"cmd": "set_controls", "controls": {"node_scales": {"model.layers.1": 0.5}}},
        {"cmd": "generate", "input_text": "hello", "max_new_tokens": 4, "compare_baseline": False},
        {"cmd": "set_steering", "positive_prompts": ["bad"], "negative_prompts": ["good"], "target_node_ids": ["model.layers.1"]},
        {"cmd": "sweep", "input_text": "hello", "target": {"type": "steering"}, "start": -1, "stop": 1, "steps": 3},
        {"cmd": "generate", "input_text": "hello", "max_new_tokens": 20},
        {"cmd": "stop"},  # 生成中に取り込まれて打ち切られる
        {"cmd": "shutdown"},
    ]:
        inbox.put(msg)
    rls.Server(inbox).serve()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    kinds = [e["event"] for e in events]
    assert events[0]["event"] == "error" and "init" in events[0]["message"]
    assert "ready" in kinds and "steering_ready" in kinds
    errors = [e for e in events if e["event"] == "error"]
    assert len(errors) == 2 and errors[1]["error_type"] == "ControlError"
    applied = [e for e in events if e["event"] == "controls_applied"]
    assert applied[0]["version"] == 1 and applied[0]["controls"]["node_scales"] == {"model.layers.1": 0.5}
    dones = [e for e in events if e["event"] == "done"]
    assert dones[0]["token_count"] == 4 and dones[0]["reason"] == "max_new_tokens"
    assert dones[1]["reason"] == "stopped"
    sweep = next(e for e in events if e["event"] == "sweep_result")
    assert [p["scale"] for p in sweep["points"]] == [-1.0, 0.0, 1.0]


def test_live_session_subprocess_roundtrip(tiny):
    """実際にプロセスとして起動し、stdin/stdoutの1行1JSONでやり取りできること(Electronと同じ経路)。"""
    d, config, base = tiny
    lines = [
        {"cmd": "init", **{k: base[k] for k in ("model_dir", "config", "nodes", "edges")}, "use_chat_template": True},
        {"cmd": "generate", "input_text": "こんにちは hi", "max_new_tokens": 3},
        {"cmd": "shutdown"},
    ]
    proc = subprocess.run(
        [sys.executable, str(PYTHON_DIR / "run_live_session.py")],
        input="\n".join(json.dumps(m, ensure_ascii=False) for m in lines) + "\n",
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        env={**os.environ, "PYTHONIOENCODING": "cp932:surrogateescape"},  # 親がUTF-8でなくても自分でUTF-8に揃える
    )
    events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "ready", (kinds, proc.stderr)
    assert "baseline" in kinds and kinds[-1] == "done"
    started = next(e for e in events if e["event"] == "started")
    assert started["input_token_count"] > 20  # チャットテンプレートで包まれている


def test_single_token_prompt_warning():
    """2026-10-14: 1トークンのプロンプトでは、文頭の巨大な活性化を拾うので警告を返す(テンプレートなしのとき)。"""
    import torch  # noqa: PLC0415
    from inference.steering import single_token_prompt_warnings  # noqa: PLC0415

    ids = [torch.tensor([[5]]), torch.tensor([[5, 6, 7]])]
    w = single_token_prompt_warnings(["a", "abc"], ids)
    assert len(w) == 1 and "'a'" in w[0] and "'abc'" not in w[0]
    assert single_token_prompt_warnings(["abc"], [torch.tensor([[1, 2]])]) == []


def test_steering_result_has_warnings_field(tiny):
    _, _, base = tiny
    r = run_entry("run_activation_steering.py", {**base, **STEER, "scale": 1.0, "use_chat_template": True})
    assert r["success"], r
    assert r["warnings"] == []  # テンプレートで包むと1トークンにはならない
