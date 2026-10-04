"""
live_session.py

Phase3: Live Slider(スライダーによるリアルタイム出力操作、2026-09-24着手)の中核(torch依存)。

1回モデルを読み込んだら、そのまま何度でも生成・スイープできる「セッション」。生成は1トークンずつ行い、
各トークンの直前に最新のコントロール(スライダーの値)を取得するので、生成中にスライダーを動かすと
次のトークンから反映される(漣さんとの確認事項「生成中に反映」)。

各トークンで、同じ文脈(ここまでに生成した同じトークン列)に対する「コントロールなし」の次トークン分布も
並行して計算し、KLダイバージェンス(通常時→操作後)を返す。これで「今のスライダーの値が、この瞬間の
予測をどれだけ変えているか」がトークンごとにわかる(ベースライン用に別のKVキャッシュを持つので計算量は約2倍)。

torch非依存の検証ロジック(コントロールの正規化・スイープ点の生成など)は live_controls.py。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from .graph_runner import LlamaGraphExecutor
from .graph_validation import GraphNode, GraphValidationError
from .live_controls import (
    Controls,
    ModelLimits,
    build_scale_vector,
    neuron_scale_spec,
    override_target,
)
from .sampling import SamplingConfig, sample_next_token
from .sensitivity import compute_kl_divergence
from .tokenizer_utils import decode_tokens, encode_text, get_stop_token_ids

SCALABLE_ROLES = ("attention_block", "mlp_block", "layer_block")


class LiveSession:
    def __init__(
        self,
        model_dir: Path,
        config: dict,
        nodes: list[GraphNode],
        edges: list[tuple[str, str]],
        entry_node_id: str,
        exit_node_id: str,
        tokenizer,
        use_chat_template: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.config = config
        self.nodes = nodes
        self.edges = edges
        self.entry = entry_node_id
        self.exit = exit_node_id
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.executor = LlamaGraphExecutor(self.model_dir, config)
        self.executor.load_weights_for_nodes([n.id for n in nodes])
        self.limits = ModelLimits(
            node_ids=frozenset(n.id for n in nodes if n.role in SCALABLE_ROLES),
            num_layers=int(config["num_hidden_layers"]),
            intermediate_size=int(config["intermediate_size"]),
            num_heads=int(config["num_attention_heads"]),
        )
        self.steering_vectors: dict[str, torch.Tensor] | None = None
        self.stop_token_ids = get_stop_token_ids(self.model_dir, tokenizer) if use_chat_template else []

    # --- 準備 ---------------------------------------------------------------

    def encode(self, text: str) -> torch.Tensor:
        ids = encode_text(self.tokenizer, text, model_dir=self.model_dir, use_chat_template=self.use_chat_template)
        if not ids:
            raise GraphValidationError(f"入力テキストのトークナイズ結果が空でした: {text!r}")
        return torch.tensor([ids])

    def set_steering(self, positive_prompts: list[str], negative_prompts: list[str], target_node_ids: list[str]) -> dict:
        """ステアリングベクトルを計算して保持する(スライダーでは強さだけを動かす)。戻り値: 各ノードのノルム。"""
        from .steering import compute_steering_vectors  # noqa: PLC0415

        unknown = set(target_node_ids) - {n.id for n in self.nodes}
        if unknown:
            raise GraphValidationError(f"ステアリング対象のノードがグラフ内にありません: {sorted(unknown)}")
        if not positive_prompts or not negative_prompts or not target_node_ids:
            raise GraphValidationError("ステアリングには positive/negative のプロンプトと対象ノードが1つ以上必要です。")
        self.steering_vectors = compute_steering_vectors(
            self.executor, self.nodes, self.edges, entry_node_id=self.entry, exit_node_id=self.exit,
            positive_input_ids=[self.encode(p) for p in positive_prompts],
            negative_input_ids=[self.encode(p) for p in negative_prompts],
            target_node_ids=list(target_node_ids),
        )
        return {nid: float(v.norm()) for nid, v in self.steering_vectors.items()}

    def execute_kwargs(self, controls: Controls) -> dict:
        """コントロールをgraph_runner.execute()の引数に変換する。"""
        kw: dict = {}
        if controls.node_scales:
            kw["node_scales"] = dict(controls.node_scales)
        spec = neuron_scale_spec(controls)
        if spec:
            tensors = {}
            for key, entries in spec.items():
                size = self.limits.intermediate_size if key.endswith(".mlp") else self.limits.num_heads
                tensors[key] = torch.tensor(build_scale_vector(size, entries), dtype=torch.float32)
            kw["neuron_scales"] = tensors
        if controls.steering_scale != 0.0:
            if self.steering_vectors is None:
                raise GraphValidationError("ステアリングの強さを0以外にするには、先にステアリングベクトルを計算してください。")
            kw["steering_vectors"] = self.steering_vectors
            kw["steering_scale"] = controls.steering_scale
        return kw

    def _forward(self, ids: torch.Tensor, cache: dict, position: int, controls: Controls | None) -> torch.Tensor:
        kw = self.execute_kwargs(controls) if controls is not None else {}
        result = self.executor.execute(
            ids, self.nodes, self.edges, entry_node_id=self.entry, exit_node_id=self.exit,
            kv_cache=cache, position_offset=position, **kw,
        )
        return result.logits[0, -1, :]

    def _decode(self, ids: list[int]) -> str:
        return decode_tokens(self.tokenizer, ids) if ids else ""

    # --- 生成 ---------------------------------------------------------------

    def generate(
        self,
        input_text: str,
        max_new_tokens: int,
        sampling: SamplingConfig,
        get_controls: Callable[[], tuple[int, Controls]],
        should_stop: Callable[[], bool],
        emit: Callable[[dict], None],
        compare_baseline: bool = True,
    ) -> dict:
        """1トークンずつ生成し、トークンごとにemit({"event": "token", ...})する。

        get_controls(): (コントロールの版数, Controls) を返す。毎トークンの直前に呼ぶので、生成中に
        スライダーが動けば次のトークンから反映される。
        should_stop(): Trueなら次のトークンに進まず打ち切る(GUIの「停止」)。
        compare_baseline: Trueなら、先にコントロールなしで最後まで生成した「通常時」の文章を
        emit({"event": "baseline", ...})で返す。
        """
        input_ids = self.encode(input_text)
        emit({"event": "started", "input_token_count": int(input_ids.shape[1])})

        if compare_baseline:
            gen = self.executor.generate(
                input_ids, self.nodes, self.edges, entry_node_id=self.entry, exit_node_id=self.exit,
                max_new_tokens=max_new_tokens, sampling_config=sampling,
                stop_token_ids=self.stop_token_ids or None,
            )
            emit({"event": "baseline", "text": self._decode(gen.generated_token_ids),
                  "token_count": len(gen.generated_token_ids)})

        ctrl_cache: dict = {}
        base_cache: dict = {}
        position = 0
        current = input_ids
        generated: list[int] = []
        reason = "max_new_tokens"
        kls: list[float] = []

        for step in range(max_new_tokens):
            if should_stop():
                reason = "stopped"
                break
            version, controls = get_controls()
            logits = self._forward(current, ctrl_cache, position, controls)
            base_logits = self._forward(current, base_cache, position, None)
            kl = float(compute_kl_divergence(base_logits, logits))
            kls.append(kl)
            next_id = sample_next_token(logits, sampling, generated)
            if self.stop_token_ids and next_id in self.stop_token_ids:
                reason = "end_of_turn"
                break
            generated.append(next_id)
            base_top = int(torch.argmax(base_logits).item())
            emit({
                "event": "token",
                "step": step,
                "token_id": next_id,
                "text": self._decode(generated),
                "kl": kl,
                "top_token_changed": base_top != int(torch.argmax(logits).item()),
                "baseline_top_token_text": decode_tokens(self.tokenizer, [base_top]),
                "controls_version": version,
            })
            position += current.shape[1]
            current = torch.tensor([[next_id]])

        summary = {
            "event": "done",
            "text": self._decode(generated),
            "token_count": len(generated),
            "reason": reason,
            "mean_kl": (sum(kls) / len(kls)) if kls else 0.0,
            "max_kl": max(kls) if kls else 0.0,
        }
        emit(summary)
        return summary

    # --- スイープ -------------------------------------------------------------

    def sweep(self, input_text: str, target: dict, scales: list[float], base_controls: Controls) -> list[dict]:
        """targetの倍率だけをscalesの各値に変えて(他のスライダーは現在の値のまま)、入力の最後の位置の
        次トークン分布を計算し、コントロールなし(通常時)とのKLダイバージェンスを返す。"""
        ids = self.encode(input_text)
        base_logits = self._forward(ids, {}, 0, None)
        base_probs = torch.softmax(base_logits.float(), dim=-1)
        base_top = int(torch.argmax(base_logits).item())
        points = []
        for s in scales:
            controls = override_target(base_controls, target, float(s))
            logits = self._forward(ids, {}, 0, controls)
            probs = torch.softmax(logits.float(), dim=-1)
            top = int(torch.argmax(logits).item())
            points.append({
                "scale": float(s),
                "kl": float(compute_kl_divergence(base_logits, logits)),
                "top_token_text": decode_tokens(self.tokenizer, [top]),
                "top_prob": float(probs[top]),
                "baseline_top_prob": float(probs[base_top]),
            })
        return points, {"baseline_top_token_text": decode_tokens(self.tokenizer, [base_top]),
                        "baseline_top_prob": float(base_probs[base_top])}
