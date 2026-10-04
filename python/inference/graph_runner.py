"""
graph_runner.py

Phase 0/Reactの解剖ビューアが生成する「ノード+エッジ」のグラフ定義を受け取り、
実際にPyTorchで推論を実行する。

グラフの検証・トポロジカルソート(torch非依存)は graph_validation.py を参照。
このファイルは実際のテンソル計算(torch依存)を担当する。

Phase 1のスコープ(合意事項):
- 対象は単一モデル内のみ(複数モデル間の接続はPhase2以降)
- ユーザーはレイヤーの順序を入れ替えたり、スキップ接続を作ったりできる
- ノード同士のshapeが合わない接続はエラーで拒否する(未学習アダプターの自動挿入はしない)
- 出力は「エラーなく計算が通ったか」までを保証する。出力の意味的な妥当性は保証しない
  (ユーザーが目視で確認する)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .graph_validation import (
    GraphNode,
    GraphValidationError,
    build_input_map_multi,
    validate_shape_compatibility,
)
from .layers import apply_rope, attention, build_rope_cache, linear, merge_tensors, rms_norm, swiglu_mlp
from .load_weights import list_available_keys, load_model_weights
from .sampling import SamplingConfig, sample_next_token


@dataclass
class ExecutionResult:
    logits: torch.Tensor  # [batch, seq_len, vocab_size]
    node_order: list[str] = field(default_factory=list)  # 実際に実行された順序(デバッグ用)
    # Phase2の内部状態可視化用。capture_internals=Trueで実行した場合のみ値が入る。
    # キーは "{node_id}.attn_weights" (Attentionブロックのみ) / "{node_id}.output" (全ブロック共通)。
    internals: dict[str, torch.Tensor] = field(default_factory=dict)


class LlamaGraphExecutor:
    """Llama系(Llama/Qwen/Mistral)モデルの重みを使って、ユーザーが組んだ
    ノードグラフの通りに推論を実行するエンジン。

    設計上の割り切り(Phase1合意事項):
    - 対応は単一モデル内のみ
    - Attention/MLPは「ブロック単位」(Q/K/V/Oをまとめて1つの計算単位、
      gate/up/downをまとめて1つの計算単位)として扱う。個々のprojection単位まで
      バラバラに繋ぎ変えることは、Attention機構の内部整合性(QとKの次元関係等)を
      壊しかねないため、Phase1ではブロック単位の入れ替え・スキップに限定する
    """

    def __init__(self, model_dir: Path, config: dict[str, Any]):
        self.model_dir = model_dir
        self.config = config
        self.hidden_size: int = config["hidden_size"]
        self.num_heads: int = config["num_attention_heads"]
        self.num_kv_heads: int = config.get("num_key_value_heads", self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.rope_base: float = config.get("rope_theta", 10000.0)
        self.norm_eps: float = config.get("rms_norm_eps", 1e-6)
        # tie_word_embeddings: 埋め込み層とLM Headで重みを共有する設計かどうか。
        # Qwen2.5の小型モデル(0.5B/1.5B等)を含む多くの省メモリモデルでTrueになっており、
        # その場合safetensorsに独立した"lm_head.weight"キーは存在しない
        # ("model.embed_tokens.weight"をそのまま使い回す)。
        self.tie_word_embeddings: bool = bool(config.get("tie_word_embeddings", False))

        self._weights: dict[str, torch.Tensor] | None = None

    def load_weights_for_nodes(self, node_ids: list[str]) -> None:
        """実行に必要なノードの重みだけを読み込む(全体をロードしない)。

        embed_tokensはexecute()が常に最初に使う(全ての実行で埋め込みは必須)ため、
        node_idsに明示的に含まれていなくても必ずロード対象に加える。
        tie_word_embeddings(lm_head.weightキーが独立して存在しない設計)の判定は
        _expand_node_to_weight_keys側で行う。

        _expand_node_to_weight_keysが返すキーには、Q/K/Vのbiasキーのように
        「アーキテクチャによっては存在しない」候補キーが含まれることがあるため、
        実際のモデルファイルに存在するキー一覧との積集合を取ってから読み込む
        (存在しないキーをload_model_weightsに渡すとKeyErrorになるため)。
        """
        required_keys: set[str] = {"model.embed_tokens.weight"}
        for node_id in node_ids:
            required_keys |= self._expand_node_to_weight_keys(node_id)

        available_keys = list_available_keys(self.model_dir)
        required_keys &= available_keys

        self._weights = load_model_weights(self.model_dir, required_keys)

    def _expand_node_to_weight_keys(self, node_id: str) -> set[str]:
        """ノードid(ブロック代表キー)から、実際に必要なsafetensorsキー一式を導出する。

        Q/K/V projectionのbiasキー(例: "...q_proj.bias")は、Qwen2のように
        存在するアーキテクチャと、Llamaのように存在しないアーキテクチャがあるため、
        ここでは「あれば使う」候補として追加しておき、実際に存在するかどうかは
        load_weights_for_nodes側でモデルの実ファイルと突き合わせて判定する。
        """
        if node_id.endswith(".self_attn"):
            base = node_id
            layer_prefix = base.rsplit(".self_attn", 1)[0]
            return {
                f"{base}.q_proj.weight",
                f"{base}.q_proj.bias",  # Qwen2等、存在する場合のみ後段でフィルタされる
                f"{base}.k_proj.weight",
                f"{base}.k_proj.bias",
                f"{base}.v_proj.weight",
                f"{base}.v_proj.bias",
                f"{base}.o_proj.weight",
                # input_layernormはこのブロックの内部処理として自動適用されるため
                # (Phase1合意事項: 正規化層はブロックに内包し独立ノード化しない)
                f"{layer_prefix}.input_layernorm.weight",
            }
        if node_id.endswith(".mlp"):
            base = node_id
            layer_prefix = base.rsplit(".mlp", 1)[0]
            return {
                f"{base}.gate_proj.weight",
                f"{base}.up_proj.weight",
                f"{base}.down_proj.weight",
                # post_attention_layernormも同様にブロック内部で自動適用
                f"{layer_prefix}.post_attention_layernorm.weight",
            }
        if node_id.endswith("_layernorm") or node_id.endswith(".norm"):
            return {f"{node_id}.weight"}
        if node_id == "model.embed_tokens":
            return {f"{node_id}.weight"}
        if node_id == "lm_head":
            # lm_head直前の最終正規化(model.norm)もこのノードの内部処理として
            # 自動適用するため、重みロード対象に含める
            keys = {"model.norm.weight"}
            if not self.tie_word_embeddings:
                keys.add("lm_head.weight")
            return keys
        # layer_blockロール: "model.layers.N" 形式で、Attention+MLP両方のキーが必要
        # (GUIでLayerを折りたたんだまま実行する場合の経路)
        if node_id.startswith("model.layers.") and node_id.count(".") == 2:
            attn_keys = self._expand_node_to_weight_keys(f"{node_id}.self_attn")
            mlp_keys = self._expand_node_to_weight_keys(f"{node_id}.mlp")
            return attn_keys | mlp_keys
        raise GraphValidationError(f"Unrecognized node id pattern: {node_id}")

    def _w(self, key: str) -> torch.Tensor:
        assert self._weights is not None, "call load_weights_for_nodes() first"
        return self._weights[key]

    def _w_optional(self, key: str) -> torch.Tensor | None:
        """重みが存在すれば返し、存在しなければNoneを返す。

        Qwen2のQ/K/V projectionバイアスのように、アーキテクチャによって
        存在したりしなかったりする重み(この場合バイアス)を扱うために使う。
        """
        assert self._weights is not None, "call load_weights_for_nodes() first"
        return self._weights.get(key)

    def run_attention_block(
        self,
        x: torch.Tensor,
        layer_prefix: str,
        cos: torch.Tensor,
        sin: torch.Tensor,
        node_id: str,
        kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_offset: int,
        capture: dict[str, torch.Tensor] | None = None,
        neuron_capture: dict[str, torch.Tensor] | None = None,
        head_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """1つのAttentionブロックを実行する。

        head_scale: Live Slider(2026-09-24追加)用。[num_heads]の倍率ベクトル(layers.attention参照)。

        Llama系アーキテクチャの標準構成に従い、以下を自動的に適用する
        (ユーザーには個別ノードとして見せず、ブロックの内部処理として扱う。
        Phase1合意事項: 正規化層は独立ノード化せず、対応するブロックに内包する):
          residual = x
          x = input_layernorm(x)
          x = Attention(x)
          x = residual + x   (残差接続)

        KVキャッシュ対応: node_id をキーとして kv_cache 辞書から過去のK/Vを取得し、
        今回計算した分と連結した結果を同じ辞書に書き戻す。これにより、逐次生成の
        各ステップで「そのノード固有」の過去K/Vが正しく引き継がれる
        (Phase1合意事項: 生成中はグラフ構造を固定するため、ノードとキャッシュの
        対応関係も生成が終わるまで一貫している)。
        position_offset: このステップの新しいトークンの、生成開始位置からの絶対位置。
        RoPEは絶対位置に依存するため、キャッシュ利用時は正しい位置のcos/sinを使う必要がある。

        capture: Noneでなければ、このノードのAttention重み(キー: "{node_id}.attn_weights")
        とブロック出力(キー: "{node_id}.output")をこの辞書に書き込む(Phase2の内部状態
        可視化機能用)。Noneの場合は一切記録せず、Phase1までの実行に余計なコストを
        かけない。

        neuron_capture: Noneでなければ、各ヘッドの出力ノルム(キー: "{node_id}.head_norms")を
        この辞書に書き込む(Neuron Search、2026-09-12追加用)。captureとは別の辞書として扱う
        (内部状態可視化とNeuron Searchは目的が異なるため、呼び出し側で必要な方だけを
        指定できるようにしている)。
        """
        residual = x
        normed = rms_norm(x, self._w(f"{layer_prefix}.input_layernorm.weight"), eps=self.norm_eps)

        # Qwen2はQ/K/V projectionにバイアス項を持つ(Llamaにはない、アーキテクチャ差異)。
        # 存在すれば適用し、存在しなければNoneのまま(バイアスなし)で計算する。
        q = linear(
            normed,
            self._w(f"{layer_prefix}.self_attn.q_proj.weight"),
            self._w_optional(f"{layer_prefix}.self_attn.q_proj.bias"),
        )
        k = linear(
            normed,
            self._w(f"{layer_prefix}.self_attn.k_proj.weight"),
            self._w_optional(f"{layer_prefix}.self_attn.k_proj.bias"),
        )
        v = linear(
            normed,
            self._w(f"{layer_prefix}.self_attn.v_proj.weight"),
            self._w_optional(f"{layer_prefix}.self_attn.v_proj.bias"),
        )

        seq_len = normed.shape[1]
        q_heads = q.view(normed.shape[0], seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = k.view(normed.shape[0], seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos_slice = cos[position_offset : position_offset + seq_len]
        sin_slice = sin[position_offset : position_offset + seq_len]
        q_heads, k_heads = apply_rope(q_heads, k_heads, cos_slice, sin_slice)
        q = q_heads.transpose(1, 2).reshape(normed.shape[0], seq_len, -1)
        k = k_heads.transpose(1, 2).reshape(normed.shape[0], seq_len, -1)

        past_k, past_v = kv_cache.get(node_id, (None, None))
        attn_out, attn_weights, new_k_cache, new_v_cache = attention(
            q, k, v, self.num_heads, self.num_kv_heads, causal=True, past_k=past_k, past_v=past_v,
            head_norm_capture=neuron_capture,
            head_norm_capture_key=(f"{node_id}.head_norms" if neuron_capture is not None else None),
            head_scale=head_scale,
        )
        kv_cache[node_id] = (new_k_cache, new_v_cache)

        attn_out = linear(attn_out, self._w(f"{layer_prefix}.self_attn.o_proj.weight"))
        block_output = residual + attn_out

        if capture is not None:
            # detach()で計算グラフから切り離す(推論専用なので勾配は不要、メモリ節約)。
            capture[f"{node_id}.attn_weights"] = attn_weights.detach()
            capture[f"{node_id}.output"] = block_output.detach()

        return block_output

    def run_mlp_block(
        self,
        x: torch.Tensor,
        layer_prefix: str,
        node_id: str | None = None,
        capture: dict[str, torch.Tensor] | None = None,
        neuron_capture: dict[str, torch.Tensor] | None = None,
        neuron_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """1つのMLPブロックを実行する。

        neuron_scale: Live Slider(2026-09-24追加)用。[intermediate_size]の倍率ベクトル。

        Attentionブロックと同様、post_attention_layernormと残差接続を
        ブロックの内部処理として自動的に適用する:
          residual = x
          x = post_attention_layernorm(x)
          x = MLP(x)  (SwiGLU)
          x = residual + x

        capture: Noneでなければ、このブロックの出力(キー: "{node_id}.output")を
        この辞書に書き込む(Phase2の内部状態可視化機能用)。node_idも同時に
        指定する必要がある。

        neuron_capture: Noneでなければ、MLP中間層の各次元の活性化値(キー:
        "{node_id}.neurons")をこの辞書に書き込む(Neuron Search、2026-09-12追加用)。
        """
        residual = x
        normed = rms_norm(
            x, self._w(f"{layer_prefix}.post_attention_layernorm.weight"), eps=self.norm_eps
        )
        mlp_out = swiglu_mlp(
            normed,
            self._w(f"{layer_prefix}.mlp.gate_proj.weight"),
            self._w(f"{layer_prefix}.mlp.up_proj.weight"),
            self._w(f"{layer_prefix}.mlp.down_proj.weight"),
            neuron_capture=neuron_capture,
            neuron_capture_key=(f"{node_id}.neurons" if (neuron_capture is not None and node_id is not None) else None),
            neuron_scale=neuron_scale,
        )
        block_output = residual + mlp_out

        if capture is not None and node_id is not None:
            capture[f"{node_id}.output"] = block_output.detach()

        return block_output

    def run_norm(self, x: torch.Tensor, node_id: str) -> torch.Tensor:
        return rms_norm(x, self._w(f"{node_id}.weight"), eps=self.norm_eps)

    def execute(
        self,
        input_ids: torch.Tensor,
        nodes: list[GraphNode],
        edges: list[tuple[str, str]],
        entry_node_id: str,
        exit_node_id: str,
        kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_offset: int,
        max_position: int = 2048,
        capture_internals: bool = False,
        neuron_capture: dict[str, torch.Tensor] | None = None,
        steering_vectors: dict[str, torch.Tensor] | None = None,
        steering_scale: float = 1.0,
        ablation_directions: dict[str, torch.Tensor] | None = None,
        node_scales: dict[str, float] | None = None,
        neuron_scales: dict[str, torch.Tensor] | None = None,
    ) -> ExecutionResult:
        """エッジで実際に接続されている経路に従って、順番に計算を適用していく(1ステップ分)。

        【重要な修正】以前の実装は、ノードの「実行順序リスト」だけを見て機械的に
        直前の出力を次のノードに渡していたため、エッジを削除してグラフを分断しても
        それが無視され、分断前と同じ結果が出てしまう不具合があった。
        (削除したはずの接続が実行結果に反映されない、というバグとして発覚)

        この実装では、各ノードの出力を node_id -> tensor の辞書に保存し、
        各ノードを実行する際は build_input_map() で解決した「実際の入力元ノード」
        から値を取得する。これにより、エッジを削除して分断されたノードは
        (build_input_mapの時点でGraphValidationErrorとして)確実に検出される。

        entry_node_id: グラフの起点となるノード(通常はembeddingノード)のID。
        exit_node_id: 最終出力として返すノードのID(通常はlm_headノード)。
        kv_cache: ノードID -> (K cache, V cache) の辞書。呼び出し元が生成ループ全体を
                  通して同じ辞書インスタンスを使い回すことで、各Attentionノードが
                  自分自身の過去K/Vを正しく蓄積できる(複数ステップの生成に対応)。
                  1回のforward pass(生成しない用途)なら空の辞書を渡せばよい。
        position_offset: このステップで処理する新しいトークンの、生成開始位置からの
                  絶対位置(0始まり)。RoPEに使う。1ステップ目は0、以降は
                  それまでに処理したトークン数を渡す。
        capture_internals: Trueの場合、各Attention/MLPブロックのAttention重み・
                  出力活性化値を ExecutionResult.internals に記録する
                  (Phase2の内部状態可視化機能用)。Falseの場合は一切記録せず、
                  Phase1までの実行と同じパフォーマンス特性を維持する。
        neuron_capture: Noneでなければ、各Attentionヘッドの出力ノルム・MLP中間層の
                  各次元の活性化値をこの辞書に書き込む(Neuron Search、2026-09-12追加用。
                  capture_internalsとは独立に指定できる)。
        steering_vectors: Noneでなければ、Activation Steering Studio(2026-09-15追加)用。
                  {node_id: ベクトル([hidden_size])} の辞書。対象ノードの出力が計算された
                  直後(残差接続・正規化を含む、そのノードの最終出力に対して)、
                  steering_scale倍したベクトルを全トークン位置・バッチに加算する
                  (ブロードキャスト加算)。無効化(ablated)されたノードには適用しない
                  (ゼロ化された出力にベクトルを足す意味がないため)。layer_blockロール
                  (層を折りたたんだまま)の場合、内部でAttention→MLPの順に実行されるが、
                  ベクトルは層全体の最終出力(MLPブロック通過後)に対して1回だけ加算される
                  (Attention出力の時点では加算しない。ノードの「最終出力」に加算する、
                  という定義を単純に保つため)。
        steering_scale: steering_vectorsに指定した各ベクトルに掛けるスケール係数
                  (正の値で「その方向へ強める」、負の値で「逆方向へ強める」、
                  0で無効化(ベクトルを足さない)と同じ効果になる)。デフォルト1.0。
        ablation_directions: Noneでなければ、Refusal Behavior Tuner(2026-09-22追加)の
                  「方向除去」用。{node_id: 方向ベクトル([hidden_size])} の辞書。対象ノードの
                  最終出力(steering_vectorsと同じ位置)から、その方向の成分を射影で取り除く
                  (x ← x - (x・r̂) r̂、r̂は単位ベクトル)。加算と違い、その方向の成分を
                  「ちょうどゼロ」にするだけで他の成分には触れないため、出力が崩壊しにくい
                  (Arditi et al. 2024 "Refusal in Language Models Is Mediated by a Single
                  Direction"の方向除去と同じ考え方)。steering_vectorsと同時に指定した場合は
                  除去→加算の順に適用する。ablatedなノードには適用しない。
        node_scales: Live Slider(2026-09-24追加)用。{node_id: 倍率}。attention_block/mlp_block/
                  layer_blockの「そのブロックが足した分」(出力-入力=残差接続で加わる寄与)に倍率を掛ける:
                  x ← x_in + s × (x_out − x_in)。s=1で通常と完全に同じ、s=0でそのブロックを素通り
                  (寄与なし)、s=2で寄与を2倍に強調。Layer Ablation(出力全体をゼロ化)とは定義が異なる
                  点に注意(倍率を連続的に動かしても出力が壊れにくいよう、寄与分だけに掛ける)。
                  方向除去・ステアリング加算より先に適用する。
        neuron_scales: Live Slider(2026-09-24追加)用。キーは "{layer_prefix}.mlp"(値は
                  [intermediate_size]の倍率ベクトル)または "{layer_prefix}.self_attn"([num_heads])。
                  layer_prefixは "model.layers.N"。展開済み・折りたたみ(layer_block)どちらのノードにも効く。
        """
        assert self._weights is not None, "call load_weights_for_nodes() first"

        node_by_id = {n.id: n for n in nodes}
        if entry_node_id not in node_by_id:
            raise GraphValidationError(f"entry_node_id '{entry_node_id}' not found in nodes")
        if exit_node_id not in node_by_id:
            raise GraphValidationError(f"exit_node_id '{exit_node_id}' not found in nodes")

        # 各ノードの実行順序(トポロジカル順)と、各ノードの入力元(実際のエッジ接続)を
        # それぞれ別に検証・解決する。両方が揃って初めて「エッジに忠実な実行」が保証される。
        from .graph_validation import topological_sort  # noqa: PLC0415 (循環import回避)

        order_ids = topological_sort(nodes, edges)
        # build_input_map_multi()は合流(複数入力)を許可し、各ノードの入力元を
        # リストで返す。単一入力のノードも要素数1のリストとして扱う。
        input_map = build_input_map_multi(nodes, edges, entry_node_id=entry_node_id)

        internals: dict[str, torch.Tensor] = {}
        capture_dict = internals if capture_internals else None

        cos, sin = None, None  # 埋め込み後、実際のdtypeが分かってから初期化する
        outputs: dict[str, torch.Tensor] = {}

        executed_ids: list[str] = []
        for node_id in order_ids:
            node = node_by_id[node_id]

            if node.role == "embedding":
                x = torch.nn.functional.embedding(input_ids, self._w("model.embed_tokens.weight"))
                cos, sin = build_rope_cache(
                    self.head_dim, max_position, base=self.rope_base, dtype=x.dtype
                )
                outputs[node_id] = x
                executed_ids.append(node_id)
                continue

            # entry_node以外は、必ずinput_mapで解決された実際の接続元から入力を取る。
            # 複数の入力元がある場合(合流)は、node.merge_modeに従って1つのテンソルに
            # 組み合わせる。組み合わせる前に、全ての入力のshapeが一致していることを
            # 検証する(異なるhidden_sizeのノード同士を合流させることはできない)。
            source_ids = input_map[node_id]
            source_tensors = [outputs[sid] for sid in source_ids]
            if len(source_tensors) > 1:
                first_dim = int(source_tensors[0].shape[-1])
                for sid, t in zip(source_ids[1:], source_tensors[1:]):
                    if int(t.shape[-1]) != first_dim:
                        raise GraphValidationError(
                            f"合流先ノード '{node_id}' への入力のshapeが一致していません: "
                            f"'{source_ids[0]}' は次元{first_dim}、'{sid}' は次元{int(t.shape[-1])}。"
                            f"合流させるノード同士は同じhidden_sizeを持つ必要があります。"
                        )
            x = merge_tensors(source_tensors, node.merge_mode)
            x_in = x
            neuron_scales_get = (neuron_scales or {}).get

            if node.ablated:
                # Layer Ablation Wizard(2026-09-12追加): このノードの実際の計算を
                # 一切行わず、出力をゼロテンソルにする(残差接続も含めて完全に寄与を断つ、
                # という漣さんとの確認事項に基づく定義)。embedding/norm/lm_headノードを
                # 無効化対象に指定した場合もここで同様にゼロ化される(GUI側は
                # attention_block/mlp_block/layer_blockのみを選択肢として提示する想定)。
                x = torch.zeros_like(x)
            elif node.role == "attention_block":
                layer_prefix = node.id.rsplit(".self_attn", 1)[0]
                x = self.run_attention_block(
                    x, layer_prefix, cos, sin, node_id, kv_cache, position_offset,
                    capture=capture_dict, neuron_capture=neuron_capture,
                    head_scale=neuron_scales_get(f"{layer_prefix}.self_attn"),
                )
            elif node.role == "mlp_block":
                layer_prefix = node.id.rsplit(".mlp", 1)[0]
                x = self.run_mlp_block(
                    x, layer_prefix, node_id=node_id, capture=capture_dict, neuron_capture=neuron_capture,
                    neuron_scale=neuron_scales_get(f"{layer_prefix}.mlp"),
                )
            elif node.role == "layer_block":
                # 折りたたまれたLayer全体を1ノードとして扱うモード。
                # 内部でAttentionブロック→MLPブロックの順に自動実行する
                # (GUIでLayerをAttention/MLPに展開していない場合の実行経路)。
                # キャッシュキーは "{layer_prefix}.self_attn" として、展開時の
                # Attentionノードと同じ命名規則に揃える(挙動の一貫性のため)。
                layer_prefix = node.id  # 例: "model.layers.3"
                x = self.run_attention_block(
                    x, layer_prefix, cos, sin, f"{layer_prefix}.self_attn", kv_cache, position_offset,
                    capture=capture_dict, neuron_capture=neuron_capture,
                    head_scale=neuron_scales_get(f"{layer_prefix}.self_attn"),
                )
                x = self.run_mlp_block(
                    x, layer_prefix, node_id=f"{layer_prefix}.mlp", capture=capture_dict,
                    neuron_capture=neuron_capture,
                    neuron_scale=neuron_scales_get(f"{layer_prefix}.mlp"),
                )
            elif node.role == "norm":
                x = self.run_norm(x, node.id)
            elif node.role == "lm_head":
                # LM Head直前には最終正規化層(model.norm)を必ず適用する
                x = rms_norm(x, self._w("model.norm.weight"), eps=self.norm_eps)
                lm_head_weight = (
                    self._w("model.embed_tokens.weight")
                    if self.tie_word_embeddings
                    else self._w("lm_head.weight")
                )
                x = linear(x, lm_head_weight)
            else:
                raise GraphValidationError(f"Unknown node role: {node.role}")

            if (
                node_scales is not None
                and node_id in node_scales
                and not node.ablated
                and node.role in ("attention_block", "mlp_block", "layer_block")
            ):
                # Live Slider(2026-09-24追加): ブロックが足した分(x - x_in)だけに倍率を掛ける
                s = float(node_scales[node_id])
                if s != 1.0:
                    x = x_in + s * (x - x_in)

            if ablation_directions is not None and node_id in ablation_directions and not node.ablated:
                # Refusal Behavior Tuner(2026-09-22追加): 方向除去。数値誤差を避けるため
                # float32で射影を計算してから元のdtypeに戻す。
                direction = ablation_directions[node_id].to(dtype=torch.float32, device=x.device)
                unit = direction / direction.norm().clamp_min(1e-12)
                x_f32 = x.float()
                x = (x_f32 - (x_f32 @ unit).unsqueeze(-1) * unit).to(x.dtype)

            if steering_vectors is not None and node_id in steering_vectors and not node.ablated:
                # Activation Steering Studio(2026-09-15追加): このノードの最終出力に
                # steering_scale倍したベクトルを加算する(全トークン位置・バッチに
                # ブロードキャスト)。node.ablatedがTrueの場合は適用しない(ゼロ化された
                # 出力にベクトルを足しても意味がないため)。
                steering_vec = steering_vectors[node_id].to(dtype=x.dtype, device=x.device)
                x = x + steering_scale * steering_vec

            if node.role != "lm_head":
                validate_shape_compatibility(int(x.shape[-1]), self.hidden_size, node.id, "next node")

            outputs[node_id] = x
            executed_ids.append(node_id)

        if exit_node_id not in outputs:
            raise GraphValidationError(
                f"exit_node_id '{exit_node_id}' was never executed. "
                f"embeddingからexit_nodeまでの経路が繋がっているか確認してください。"
            )

        return ExecutionResult(logits=outputs[exit_node_id], node_order=executed_ids, internals=internals)

    def generate(
        self,
        input_ids: torch.Tensor,
        nodes: list[GraphNode],
        edges: list[tuple[str, str]],
        entry_node_id: str,
        exit_node_id: str,
        max_new_tokens: int,
        sampling_config: "SamplingConfig | None" = None,
        max_position: int = 2048,
        steering_vectors: dict[str, torch.Tensor] | None = None,
        steering_scale: float = 1.0,
        ablation_directions: dict[str, torch.Tensor] | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> "GenerationResult":
        """KVキャッシュを使って複数トークンを逐次生成する。

        Phase1合意事項: 生成中はグラフ構造(繋ぎ方)を固定する。最初にトポロジカル
        ソート・入力元解決を行ったグラフのまま、最後まで生成し続ける。これにより、
        各Attentionノードのキャッシュとの対応関係が生成の最初から最後まで一貫する。

        手順:
          1. 入力プロンプト全体(input_ids)を1回のforward passで処理し、
             各AttentionノードのKVキャッシュを構築する(position_offset=0)。
          2. その出力から次の1トークンを選ぶ(sampling_configに従う。デフォルトは
             貪欲デコード=常に最大確率のトークン)。
          3. 新しく生成した1トークンだけを次のforward passの入力にし、
             position_offsetを増やしながらKVキャッシュを使って続きを生成する。
          4. max_new_tokens回繰り返す。

        sampling_config: Noneの場合は貪欲デコード(temperature=0.0相当)。
        temperature/top-k/top-p/repetition penaltyを指定すると、それらを
        適用したサンプリングで次トークンを選ぶ(sampling.pyのsample_next_token参照)。
        steering_vectors/steering_scale: Activation Steering Studio(2026-09-15追加)用。
        execute()にそのまま引き継ぎ、生成の各ステップ(新しいトークンごと)で同じベクトルを
        加算し続ける(グラフ構造と同様、生成中は一貫して適用する)。
        ablation_directions: Refusal Behavior Tuner(2026-09-22追加)の方向除去用。
        steering_vectorsと同様にexecute()へ引き継ぎ、生成の全ステップで適用する。
        stop_token_ids: 指定した場合(2026-09-22追加、チャットテンプレート適用時)、
        これらのトークンが生成された時点で生成を打ち切る(停止トークン自体は結果に含めない)。
        Noneなら従来どおりmax_new_tokens分を最後まで生成する。
        """
        assert self._weights is not None, "call load_weights_for_nodes() first"

        if sampling_config is None:
            sampling_config = SamplingConfig(temperature=0.0)

        kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        generated_ids: list[int] = []
        executed_order: list[str] = []

        current_input = input_ids
        position_offset = 0

        for step in range(max_new_tokens):
            result = self.execute(
                current_input,
                nodes,
                edges,
                entry_node_id=entry_node_id,
                exit_node_id=exit_node_id,
                kv_cache=kv_cache,
                position_offset=position_offset,
                max_position=max_position,
                steering_vectors=steering_vectors,
                steering_scale=steering_scale,
                ablation_directions=ablation_directions,
            )
            if step == 0:
                # 実行順序はどのステップでも同じグラフ構造から来るため、最初のステップの
                # ものだけ記録すれば十分(Phase1合意事項: 生成中はグラフ構造を固定する)
                executed_order = result.node_order

            next_token_logits = result.logits[0, -1, :]  # [vocab_size]
            next_token_id = sample_next_token(next_token_logits, sampling_config, generated_ids)
            if stop_token_ids and next_token_id in stop_token_ids:
                break
            generated_ids.append(next_token_id)

            position_offset += current_input.shape[1]
            current_input = torch.tensor([[next_token_id]], device=input_ids.device)

        return GenerationResult(generated_token_ids=generated_ids, executed_order=executed_order)


@dataclass
class GenerationResult:
    generated_token_ids: list[int]
    executed_order: list[str]


def load_graph_definition(graph_json_path: Path) -> tuple[list[GraphNode], list[tuple[str, str]]]:
    """React側から渡されるノード+エッジ定義のJSONを読み込む。"""
    data = json.loads(graph_json_path.read_text(encoding="utf-8"))
    nodes = [
        GraphNode(id=n["id"], role=n["role"], layer_index=n.get("layer_index"))
        for n in data["nodes"]
    ]
    edges = [(e["source"], e["target"]) for e in data["edges"]]
    return nodes, edges
