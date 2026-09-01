import { useCallback, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  SelectionMode,
  type Node,
  type Edge,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { toFlowGraph } from "./lib/toFlowGraph";
import { Legend } from "./components/Legend";
import { NormalNode, type NormalNodeData } from "./components/NormalNode";
import { LayerSummaryNode, type LayerSummaryNodeData } from "./components/LayerSummaryNode";
import type { ModelInspectResult } from "./electron.d";

const nodeTypes: NodeTypes = {
  normal: NormalNode,
  layerSummary: LayerSummaryNode,
};

type LoadState =
  | { status: "idle" }
  | { status: "loading"; modelDir: string }
  | { status: "error"; message: string }
  | { status: "loaded"; modelDir: string; summary: ModelInspectResult };

function formatParamCount(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${n}`;
}

export default function App() {
  const [rawResult, setRawResult] = useState<ModelInspectResult | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [loadState, setLoadState] = useState<LoadState>({ status: "idle" });
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [selectedCount, setSelectedCount] = useState(0);

  // 展開中のレイヤー番号の集合。ここに含まれない層は折りたたみ表示(代表ノード1つ)になる。
  const [expandedLayers, setExpandedLayers] = useState<Set<number>>(new Set());

  const applyGraph = useCallback(
    (result: ModelInspectResult, expanded: Set<number>) => {
      const { nodes: flowNodes, edges: flowEdges } = toFlowGraph(result, expanded);
      setNodes(flowNodes);
      setEdges(flowEdges);
    },
    [setNodes, setEdges]
  );

  const handleSelectModel = useCallback(async () => {
    const dir = await window.electronAPI.selectModelDir();
    if (!dir) return;

    setLoadState({ status: "loading", modelDir: dir });
    try {
      const result = await window.electronAPI.inspectModel(dir);
      setRawResult(result);
      const initialExpanded = new Set<number>(); // 初期状態は全層折りたたみ
      setExpandedLayers(initialExpanded);
      applyGraph(result, initialExpanded);
      setLoadState({ status: "loaded", modelDir: dir, summary: result });
    } catch (err) {
      setLoadState({
        status: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, [applyGraph]);

  const toggleLayer = useCallback(
    (layerIndex: number) => {
      if (!rawResult) return;
      setExpandedLayers((prev) => {
        const next = new Set(prev);
        if (next.has(layerIndex)) {
          next.delete(layerIndex);
        } else {
          next.add(layerIndex);
        }
        applyGraph(rawResult, next);
        return next;
      });
    },
    [rawResult, applyGraph]
  );

  const handleNodeClick = useCallback(
    (_: unknown, node: Node) => {
      setSelectedNode(node);
      const data = node.data as unknown as LayerSummaryNodeData | NormalNodeData;
      if (data.kind === "layerSummary") {
        toggleLayer(data.layerIndex);
      }
    },
    [toggleLayer]
  );

  const handleSelectionChange = useCallback(
    ({ nodes: selected }: { nodes: Node[]; edges: Edge[] }) => {
      setSelectedCount(selected.length);
    },
    []
  );

  const totalLayers = rawResult?.num_layers ?? 0;
  const layerToggleLabel = useMemo(() => {
    if (!totalLayers) return "";
    return `展開中 ${expandedLayers.size} / ${totalLayers} 層`;
  }, [expandedLayers, totalLayers]);

  const totalParams = useMemo(() => {
    if (!rawResult) return null;
    let sum = 0;
    for (const n of rawResult.nodes) {
      if (!n.shape) continue;
      sum += n.shape.reduce((a, b) => a * b, 1);
    }
    return sum;
  }, [rawResult]);

  const expandAll = useCallback(() => {
    if (!rawResult || !totalLayers) return;
    const all = new Set<number>(Array.from({ length: totalLayers }, (_, i) => i));
    setExpandedLayers(all);
    applyGraph(rawResult, all);
  }, [rawResult, totalLayers, applyGraph]);

  const collapseAll = useCallback(() => {
    if (!rawResult) return;
    const none = new Set<number>();
    setExpandedLayers(none);
    applyGraph(rawResult, none);
  }, [rawResult, applyGraph]);

  const selectedData =
    selectedNode && selectedCount <= 1
      ? (selectedNode.data as unknown as LayerSummaryNodeData | NormalNodeData)
      : null;

  return (
    <div style={{ width: "100vw", height: "100vh", display: "flex", flexDirection: "column" }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "10px 16px",
          background: "var(--bg-panel)",
          borderBottom: "1px solid var(--border-default)",
        }}
      >
        <strong style={{ fontSize: 14, letterSpacing: 0.2 }}>解剖ビューア</strong>

        <button onClick={handleSelectModel} style={buttonStyle}>
          モデルフォルダを開く
        </button>

        {loadState.status === "loaded" && (
          <>
            <div style={{ width: 1, height: 20, background: "var(--border-default)" }} />
            <button onClick={expandAll} style={buttonStyle}>
              全層展開
            </button>
            <button onClick={collapseAll} style={buttonStyle}>
              全層折りたたみ
            </button>
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{layerToggleLabel}</span>
          </>
        )}

        {loadState.status === "loading" && (
          <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>
            解析中: {loadState.modelDir}
          </span>
        )}
        {loadState.status === "error" && (
          <span style={{ fontSize: 12, color: "#e08a8a" }}>エラー: {loadState.message}</span>
        )}
        {loadState.status === "loaded" && (
          <span
            style={{
              fontSize: 12,
              color: "var(--text-secondary)",
              fontFamily: "var(--font-mono)",
            }}
          >
            {loadState.summary.model_type} · {loadState.summary.num_layers ?? "?"} layers ·{" "}
            hidden={loadState.summary.hidden_size ?? "?"}
            {totalParams !== null && <> · {formatParamCount(totalParams)} params</>}
          </span>
        )}

        {selectedCount > 1 && (
          <span style={{ fontSize: 12, color: "var(--accent-text)", marginLeft: "auto" }}>
            {selectedCount} 個選択中
          </span>
        )}
      </header>

      <div style={{ flex: 1, position: "relative", background: "var(--bg-app)" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          onSelectionChange={handleSelectionChange}
          fitView
          minZoom={0.05}
          // 複数範囲選択: 左ドラッグで矩形選択。パンは右/中ボタンドラッグに割り当て。
          selectionOnDrag
          selectionMode={SelectionMode.Partial}
          panOnDrag={[1, 2]}
          multiSelectionKeyCode="Shift"
        >
          <Background color="var(--border-default)" gap={24} size={1} />
          <Controls />
          <MiniMap
            nodeColor={() => "var(--border-strong)"}
            style={{ background: "var(--bg-panel)" }}
          />
        </ReactFlow>

        <Legend />

        {selectedData && (
          <div
            style={{
              position: "absolute",
              top: 16,
              right: 16,
              width: 280,
              background: "var(--bg-panel)",
              border: "1px solid var(--border-default)",
              color: "var(--text-primary)",
              borderRadius: "var(--radius-md)",
              padding: 14,
              fontSize: 12,
              zIndex: 10,
            }}
          >
            {selectedData.kind === "layerSummary" ? (
              <>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>Layer {selectedData.layerIndex}</div>
                <div style={{ color: "var(--text-secondary)" }}>
                  {selectedData.nodeCount} ノード ·{" "}
                  {selectedData.isExpanded ? "展開中" : "折りたたみ中"}
                </div>
                <div style={{ color: "var(--text-muted)", marginTop: 6 }}>
                  クリックで{selectedData.isExpanded ? "折りたたみ" : "展開"}
                </div>
              </>
            ) : (
              <>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>{selectedData.displayName}</div>
                <div style={{ color: "var(--text-secondary)", marginBottom: 2 }}>
                  役割: {selectedData.roleLabelJa}
                </div>
                {selectedData.shape && (
                  <div style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                    shape: [{selectedData.shape.join(", ")}]
                  </div>
                )}
                {selectedData.paramCount !== null && (
                  <div style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                    {formatParamCount(selectedData.paramCount)} params
                  </div>
                )}
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    wordBreak: "break-all",
                    marginTop: 8,
                    color: "var(--text-muted)",
                    fontSize: 10.5,
                  }}
                >
                  {selectedData.paramKey}
                </div>
              </>
            )}
          </div>
        )}

        {loadState.status === "idle" && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--text-muted)",
              fontSize: 14,
              pointerEvents: "none",
            }}
          >
            「モデルフォルダを開く」からHuggingFace形式のモデルを選択してください
          </div>
        )}
      </div>
    </div>
  );
}

const buttonStyle = {
  padding: "6px 12px",
  fontSize: 12,
  borderRadius: "var(--radius-sm)",
  border: "1px solid var(--border-strong)",
  background: "var(--bg-panel-raised)",
  color: "var(--text-primary)",
  cursor: "pointer",
} as const;
