"""
steering.py

Phase3: Activation Steering Studio(活性化操作による挙動誘導)の中核ロジック(torch依存)。

2026-09-15、漣さんとの確認事項:
- ステアリングベクトルの作り方: 対照的な2つのプロンプト群(positive/negative)の
  最後のトークン位置の活性化値の平均を取り、その差分(positive - negative)を
  ベクトルとして使う("Contrastive Activation Addition"系の一般的な手法)
- 適用対象: ユーザーがGUI上で選んだ1つ以上のノード(Layer Ablation Wizardと同様、
  attention_block/mlp_block/layer_blockのチェックボックス選択)。ノードごとに、
  そのノード自身で計算した差分ベクトルを、共通のscaleで加算する
- 効果の確認: 生成テキストの変化(目視)に加え、KLダイバージェンス(Prompt Sensitivity
  Map・Layer Ablation Wizardと同じcompute_kl_divergenceを流用)を定量指標として提示する

ベクトルの計算(compute_steering_vectors)自体は1回のforward pass(capture_internals=True)を
プロンプトの数だけ繰り返すだけで、既存のgraph_runner.LlamaGraphExecutor.execute()を
そのまま使い回している(新しい推論ロジックを追加しているわけではない)。ベクトルの
「適用」(生成時に加算する部分)はgraph_runner.pyのexecute()/generate()に
steering_vectors/steering_scale引数として実装した(そちらもLayer Ablation Wizardの
node.ablatedと同じ形の、既存ループへの小さな追加)。
"""

from __future__ import annotations

import torch

from .graph_runner import LlamaGraphExecutor
from .graph_validation import GraphNode, GraphValidationError


def resolve_output_key(internals: dict[str, torch.Tensor], node_id: str) -> str:
    """internals辞書から、指定ノードの最終出力に対応するキーを解決する。

    通常のattention_block/mlp_blockノードは "{node_id}.output" だが、layer_block
    ロール(層を折りたたんだまま)の場合は内部で"{node_id}.self_attn.output"/
    "{node_id}.mlp.output"の2つに分かれて記録される(graph_runner.py参照、
    inspect_internals.pyの既存のリネーム処理と同じ考え方)。層全体の最終出力を
    代表させたいので、mlp側(Attentionの後に実行される方)を優先し、なければ
    self_attn側にフォールバックする。
    """
    for key in (f"{node_id}.output", f"{node_id}.mlp.output", f"{node_id}.self_attn.output"):
        if key in internals:
            return key
    raise GraphValidationError(
        f"ノード '{node_id}' の内部出力が見つかりませんでした。"
        f"target_node_idsに指定したノードがグラフ内に実在するか確認してください。"
    )


def compute_steering_vectors(
    executor: LlamaGraphExecutor,
    nodes: list[GraphNode],
    edges: list[tuple[str, str]],
    entry_node_id: str,
    exit_node_id: str,
    positive_input_ids: list[torch.Tensor],
    negative_input_ids: list[torch.Tensor],
    target_node_ids: list[str],
) -> dict[str, torch.Tensor]:
    """対照的なプロンプト群それぞれについて1回ずつforward passし(capture_internals=True)、
    各対象ノードの最後のトークン位置の活性化値を集め、正群の平均 - 負群の平均を
    差分ベクトル(ステアリングベクトル)として返す。

    最後のトークン位置を代表値として使う理由: プロンプト全体の意味を集約した表現であり、
    かつ次トークン予測に直接使われる位置であるため(Prompt Sensitivity Map等、他の
    Phase2機能でも「最後の位置の次トークン分布」を基準にしている慣習と揃えている)。

    戻り値: {node_id: ベクトル([hidden_size]、torch.Tensor)}
    """
    if not positive_input_ids:
        raise GraphValidationError("positive_prompts を1つ以上指定してください。")
    if not negative_input_ids:
        raise GraphValidationError("negative_prompts を1つ以上指定してください。")
    if not target_node_ids:
        raise GraphValidationError("target_node_ids を1つ以上指定してください。")

    def collect(prompt_input_ids: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        collected: dict[str, list[torch.Tensor]] = {nid: [] for nid in target_node_ids}
        for input_ids in prompt_input_ids:
            result = executor.execute(
                input_ids,
                nodes,
                edges,
                entry_node_id=entry_node_id,
                exit_node_id=exit_node_id,
                kv_cache={},
                position_offset=0,
                capture_internals=True,
            )
            for nid in target_node_ids:
                key = resolve_output_key(result.internals, nid)
                vec = result.internals[key][0, -1, :].float()  # 最後のトークン位置、[hidden_size]
                collected[nid].append(vec)
        return collected

    positive_collected = collect(positive_input_ids)
    negative_collected = collect(negative_input_ids)

    steering_vectors: dict[str, torch.Tensor] = {}
    for nid in target_node_ids:
        pos_mean = torch.stack(positive_collected[nid], dim=0).mean(dim=0)
        neg_mean = torch.stack(negative_collected[nid], dim=0).mean(dim=0)
        steering_vectors[nid] = (pos_mean - neg_mean).detach()

    return steering_vectors
