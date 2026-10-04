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

// 列幅は展開状態によって動的に決まる。折りたたまれた層は代表ノード1つぶんの
// 幅しか使わず、展開されるほど(ブロック単位 -> 個別ノードまで)幅が広がる。
// これにより、隣接する層のノードが重ならず、かつ折りたたんだ層は無駄に
// 横幅を占有しない。
const COLLAPSED_COLUMN_WIDTH = 260; // 折りたたみ状態(代表ノード1つ)の列幅
const SUB_COLUMN_WIDTH = 240; // 展開時、Attention/MLPそれぞれが占める列幅
const COLUMN_GAP = 40; // 層と層の間の余白
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

// ノードが Attention系(正規化含む) か MLP系(正規化含む) かを判定する。
// pre/post_feedforward_layernormはMLP側、input/post_attention_layernormはAttention側に属する。
function subBlockOf(id: string): "attention" | "mlp" {
  if (id.includes("mlp") || id.includes("feedforward")) return "mlp";
  return "attention";
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

function summaryNode(
  id: string,
  x: number,
  y: number,
  label: string,
  nodeCount: number,
  isExpanded: boolean,
  groupKey: string
): Node<LayerSummaryNodeData> {
  return {
    id,
    type: "layerSummary",
    position: { x, y },
    data: { kind: "layerSummary", label, nodeCount, isExpanded, groupKey },
  };
}

/**
 * バックエンド(Python)から受け取ったモデル構造を、
 * React Flowが描画できるノード/エッジ配列に変換する。
 *
 * 3段階の折りたたみに対応:
 *   段階1(デフォルト): "layer-N" が expandedGroups になければ、Layer全体を1ノードで表示
 *   段階2: "layer-N" が展開済み、かつ "layer-N-attention"/"layer-N-mlp" が未展開なら、
 *          Attentionブロック1ノード・MLPブロック1ノードの計2ノードで表示
 *   段階3: さらに "layer-N-attention" (または "-mlp") も展開済みなら、
 *          そのブロック内の個別ノード(Q/K/V/O等)まで展開する
 *
 * 列幅(x座標)は各層の展開状態に応じて動的に計算する。折りたたまれた層は
 * 代表ノード1つぶんの幅しか使わないため、多くの層を折りたたんだままにしておけば
 * 横幅を大きく節約できる。個別ノードまでの展開は列幅を追加で広げない
 * (Q/K/V/O等は同じ列内で縦に並ぶため)。
 */
export function toFlowGraph(
  result: ModelInspectResult,
  expandedGroups: Set<string>
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

  // 各層が実際に占める幅(展開状態に応じて動的)を先に計算し、
  // 累積してx座標(列の開始位置)を決める。
  let cursorX = 0;

  for (const layerIndex of sortedLayers) {
    const layerNodes = nodesByLayer.get(layerIndex)!;
    const layerKey = `layer-${layerIndex}`;
    const layerExpanded = expandedGroups.has(layerKey);
    const x = cursorX;

    // --- 段階1: Layer全体の代表ノード ---
    nodes.push(
      summaryNode(
        `layer-summary-${layerIndex}`,
        x,
        HEADER_ROW_Y,
        `Layer ${layerIndex}`,
        layerNodes.length,
        layerExpanded,
        layerKey
      )
    );

    if (!layerExpanded) {
      cursorX += COLLAPSED_COLUMN_WIDTH + COLUMN_GAP;
      continue;
    }

    // --- 段階2: Attention/MLPブロック単位に分割 ---
    const attentionNodes = layerNodes.filter((n) => subBlockOf(n.id) === "attention");
    const mlpNodes = layerNodes.filter((n) => subBlockOf(n.id) === "mlp");

    const blocks: {
      key: "attention" | "mlp";
      label: string;
      blockNodes: ModelNode[];
    }[] = [
      { key: "attention", label: `Attention (Layer ${layerIndex})`, blockNodes: attentionNodes },
      { key: "mlp", label: `MLP (Layer ${layerIndex})`, blockNodes: mlpNodes },
    ];

    let subX = x;
    for (const block of blocks) {
      const blockGroupKey = `${layerKey}-${block.key}`;
      const blockExpanded = expandedGroups.has(blockGroupKey);

      nodes.push(
        summaryNode(
          `block-summary-${layerIndex}-${block.key}`,
          subX,
          0,
          block.label,
          block.blockNodes.length,
          blockExpanded,
          blockGroupKey
        )
      );

      if (blockExpanded) {
        // --- 段階3: 個別ノード(Q/K/V/O等)まで展開 ---
        const sorted = [...block.blockNodes].sort(
          (a, b) => rowIndexForKey(a.id) - rowIndexForKey(b.id)
        );
        for (const n of sorted) {
          const row = rowIndexForKey(n.id);
          nodes.push(toNormalNode(n, subX, (row + 1) * ROW_HEIGHT));
        }

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

      subX += SUB_COLUMN_WIDTH;
    }

    cursorX = subX + COLUMN_GAP;
  }

  return { nodes, edges };
}
