import type { Node, Edge } from "@xyflow/react";
import type { ModelInspectResult, ModelNode } from "../electron.d";
import type { NormalNodeData } from "../components/NormalNode";
import type { LayerSummaryNodeData } from "../components/LayerSummaryNode";

const ROLE_LABELS_JA: Record<string, string> = {
  embedding: "埋め込み層",
  attention: "Attention",
  mlp: "MLP",
  norm: "正規化層",
  lm_head: "LM Head",
  other: "その他",
};

// フルラベル("Attention (Q projection) — Layer 3")から、ノード内表示用の短い名前を作る。
// 「— Layer N」部分はレイヤー列自体で分かるため冗長なので省く。
function shortenLabel(fullLabel: string): string {
  return fullLabel.replace(/\s*—\s*Layer\s*\d+\s*$/, "");
}

// shapeの各次元の積からパラメータ数を計算する。
function paramCountFromShape(shape: number[] | null): number | null {
  if (!shape || shape.length === 0) return null;
  return shape.reduce((a, b) => a * b, 1);
}

const LAYER_COLUMN_WIDTH = 280;
const ROW_HEIGHT = 76;
const NON_LAYER_X = -340; // embedding/lm_head等、layer_indexを持たないノードの列
const HEADER_ROW_Y = -ROW_HEIGHT; // 展開/折りたたみヘッダーの縦位置

// レイヤー内でのノードの「縦位置(行)」を、パイプライン順に沿って個別に割り当てる。
// 同じroleでもQ/K/V/Oのような複数ノードが重ならず縦に並ぶようにするため。
const ROW_ORDER_FRAGMENTS = [
  "input_layernorm",
  "self_attn.q_proj",
  "self_attn.k_proj",
  "self_attn.v_proj",
  "self_attn.o_proj",
  "post_attention_layernorm",
  "pre_feedforward_layernorm",
  "mlp.gate_proj",
  "mlp.up_proj",
  "mlp.down_proj",
  "post_feedforward_layernorm",
];

function rowIndexForKey(id: string): number {
  for (let i = 0; i < ROW_ORDER_FRAGMENTS.length; i++) {
    if (id.includes(ROW_ORDER_FRAGMENTS[i])) return i;
  }
  return ROW_ORDER_FRAGMENTS.length; // 未知のキーは末尾に配置
}

function toNormalNode(n: ModelNode, x: number, y: number): Node<NormalNodeData> {
  return {
    id: n.id,
    type: "normal",
    position: { x, y },
    data: {
      kind: "normal",
      displayName: shortenLabel(n.label),
      role: n.role,
      roleLabelJa: ROLE_LABELS_JA[n.role] ?? n.role,
      shape: n.shape,
      paramKey: n.param_key,
      paramCount: paramCountFromShape(n.shape),
    },
  };
}

/**
 * バックエンド(Python)から受け取ったモデル構造を、
 * React Flowが描画できるノード/エッジ配列に変換する。
 *
 * expandedLayers に含まれないレイヤー番号は「折りたたみ」表示となり、
 * そのレイヤーの全ノードの代わりに1つの代表ノード(layer summary)だけを描画する。
 * ノード数が多いモデル(数十層)でも初期表示が重くならないようにするための対応。
 */
export function toFlowGraph(
  result: ModelInspectResult,
  expandedLayers: Set<number>
): {
  nodes: Node[];
  edges: Edge[];
} {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  // レイヤーごとにノードをグルーピング
  const nodesByLayer = new Map<number, ModelNode[]>();
  const nonLayerNodes: ModelNode[] = [];
  for (const n of result.nodes) {
    if (n.layer_index === null) {
      nonLayerNodes.push(n);
    } else {
      const arr = nodesByLayer.get(n.layer_index) ?? [];
      arr.push(n);
      nodesByLayer.set(n.layer_index, arr);
    }
  }

  // layer_index を持たないノード(embedding, final norm, lm_head)
  for (const n of nonLayerNodes) {
    const fixedOrder: Record<string, number> = { embedding: 0, lm_head: 2 };
    const row = fixedOrder[n.role] ?? 1;
    nodes.push(toNormalNode(n, NON_LAYER_X, row * ROW_HEIGHT * 2));
  }

  const sortedLayers = Array.from(nodesByLayer.keys()).sort((a, b) => a - b);

  for (const layerIndex of sortedLayers) {
    const layerNodes = nodesByLayer.get(layerIndex)!;
    const isExpanded = expandedLayers.has(layerIndex);
    const x = layerIndex * LAYER_COLUMN_WIDTH;

    const summaryData: LayerSummaryNodeData = {
      kind: "layerSummary",
      layerIndex,
      nodeCount: layerNodes.length,
      isExpanded,
    };
    nodes.push({
      id: `layer-summary-${layerIndex}`,
      type: "layerSummary",
      position: { x, y: HEADER_ROW_Y },
      data: summaryData,
    });

    if (!isExpanded) continue;

    // 展開: レイヤー内の全ノードをパイプライン順で縦に並べる
    const sorted = [...layerNodes].sort((a, b) => rowIndexForKey(a.id) - rowIndexForKey(b.id));
    for (const n of sorted) {
      const row = rowIndexForKey(n.id);
      nodes.push(toNormalNode(n, x, row * ROW_HEIGHT));
    }

    // レイヤー内のエッジも展開時のみ描画
    for (const e of result.edges) {
      if (sorted.some((n) => n.id === e.source) && sorted.some((n) => n.id === e.target)) {
        edges.push({
          id: e.id,
          source: e.source,
          target: e.target,
          animated: false,
          style: { stroke: "var(--border-strong)" },
        });
      }
    }
  }

  return { nodes, edges };
}
