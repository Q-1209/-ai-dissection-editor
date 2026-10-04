import type { Node, Edge } from "@xyflow/react";
import type { ModelInspectResult, ExecutionNode, ExecutionEdge } from "../electron.d";

// Phase0の可視化グラフ(Q/K/V/O個別ノード)とは異なり、Phase1の実行グラフは
// Attentionブロック1つ・MLPブロック1つを単位として扱う。
// (合意事項: 個々のprojection単位でバラバラに繋ぎ変えると、Attention機構内部の
//  整合性が壊れるため、Phase1ではブロック単位の入れ替え・スキップに限定する)
//
// さらに、デフォルトではLayer全体を1つの折りたたみノード(layer_block)として表示し、
// クリックした層だけAttention/MLPの2ノードに展開できる。折りたたまれたままの
// layer_blockノードも、内部でAttention→MLPを自動実行する形でそのまま実行可能。

export const EXECUTION_ROLE_COLORS: Record<string, string> = {
  embedding: "#1e3a52",
  attention_block: "#4a3a1a",
  mlp_block: "#1c3826",
  layer_block: "#262b2f",
  norm: "#2e2540",
  lm_head: "#3d1f2c",
};

export const EXECUTION_ROLE_BORDERS: Record<string, string> = {
  embedding: "#5b9bd5",
  attention_block: "#d4a24e",
  mlp_block: "#5fae7a",
  layer_block: "#5a656c",
  norm: "#9b7fc4",
  lm_head: "#c4587e",
};

const EXECUTION_ROLE_LABELS: Record<string, string> = {
  embedding: "Embedding",
  attention_block: "Attention",
  mlp_block: "MLP",
  layer_block: "Layer",
  norm: "Norm",
  lm_head: "LM Head",
};

const COLUMN_WIDTH = 220;
const NODE_WIDTH = 180;

/**
 * Phase0の解析結果から、実行用のブロック単位ノード/エッジを生成する。
 * デフォルトのエッジ配置は「オリジナルのモデル構造通りの順序」——
 * ユーザーがここから自由に繋ぎ変えることを想定した初期状態。
 *
 * expandedLayers: 展開する(Attention/MLPの2ノードに分ける)層番号の集合。
 * 含まれない層は "model.layers.N" という単一のlayer_blockノードになる。
 */
export function toExecutionGraph(
  result: ModelInspectResult,
  expandedLayers: Set<number>
): { nodes: Node[]; edges: Edge[]; executionNodes: ExecutionNode[] } {
  const executionNodes: ExecutionNode[] = [];

  executionNodes.push({ id: "model.embed_tokens", role: "embedding", layer_index: null });

  const numLayers = result.num_layers ?? 0;
  for (let i = 0; i < numLayers; i++) {
    if (expandedLayers.has(i)) {
      executionNodes.push({
        id: `model.layers.${i}.self_attn`,
        role: "attention_block",
        layer_index: i,
      });
      executionNodes.push({
        id: `model.layers.${i}.mlp`,
        role: "mlp_block",
        layer_index: i,
      });
    } else {
      executionNodes.push({
        id: `model.layers.${i}`,
        role: "layer_block",
        layer_index: i,
      });
    }
  }

  executionNodes.push({ id: "lm_head", role: "lm_head", layer_index: null });

  // デフォルトエッジ: オリジナル構造通りの直列接続
  const executionEdges: ExecutionEdge[] = [];
  for (let i = 0; i < executionNodes.length - 1; i++) {
    executionEdges.push({ source: executionNodes[i].id, target: executionNodes[i + 1].id });
  }

  const nodes: Node[] = executionNodes.map((n, index) => {
    const bg = EXECUTION_ROLE_COLORS[n.role] ?? "#262b2f";
    const border = EXECUTION_ROLE_BORDERS[n.role] ?? "#5a656c";
    const label =
      n.layer_index !== null
        ? n.role === "layer_block"
          ? `Layer ${n.layer_index}（クリックで展開）`
          : `${EXECUTION_ROLE_LABELS[n.role]} (Layer ${n.layer_index})`
        : EXECUTION_ROLE_LABELS[n.role];
    return {
      id: n.id,
      position: { x: index * COLUMN_WIDTH, y: 0 },
      data: {
        label,
        role: n.role,
        layerIndex: n.layer_index,
      },
      style: {
        background: bg,
        border: n.role === "layer_block" ? `1px dashed ${border}` : `1px solid ${border}`,
        color: "#e4e8ea",
        borderRadius: 8,
        padding: "8px 10px",
        width: NODE_WIDTH,
        fontSize: 12,
        cursor: n.role === "layer_block" ? "pointer" : "default",
      },
    };
  });

  const edges: Edge[] = executionEdges.map((e) => ({
    id: `${e.source}->${e.target}`,
    source: e.source,
    target: e.target,
    style: { stroke: "var(--border-strong)" },
  }));

  return { nodes, edges, executionNodes };
}

/** React Flowの現在のedges状態から、Python側に送るExecutionEdge[]を作る。 */
export function flowEdgesToExecutionEdges(edges: Edge[]): ExecutionEdge[] {
  return edges.map((e) => ({ source: e.source, target: e.target }));
}
