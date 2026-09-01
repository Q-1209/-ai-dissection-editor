import { Handle, Position } from "@xyflow/react";
import type { CSSProperties } from "react";

export interface LayerSummaryNodeData extends Record<string, unknown> {
  kind: "layerSummary";
  layerIndex: number;
  nodeCount: number;
  isExpanded: boolean;
}

export function LayerSummaryNode({ data }: { data: LayerSummaryNodeData }) {
  const style: CSSProperties = {
    background: "var(--role-summary-bg)",
    border: "1px dashed var(--role-summary-border)",
    color: "var(--role-summary-text)",
    borderRadius: "var(--radius-sm)",
    padding: "8px 10px",
    width: 220,
    cursor: "pointer",
    fontFamily: "var(--font-ui)",
  };

  return (
    <div style={style}>
      <Handle type="target" position={Position.Left} style={handleStyle} />
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontSize: 11, opacity: 0.8 }} aria-hidden="true">
          {data.isExpanded ? "▾" : "▸"}
        </span>
        <span style={{ fontSize: 12, fontWeight: 600 }}>Layer {data.layerIndex}</span>
      </div>
      <div style={{ fontSize: 10.5, opacity: 0.75, marginTop: 4 }}>
        {data.isExpanded ? "クリックで折りたたむ" : `クリックで展開 · ${data.nodeCount}ノード`}
      </div>
      <Handle type="source" position={Position.Right} style={handleStyle} />
    </div>
  );
}

const handleStyle: CSSProperties = {
  width: 6,
  height: 6,
  background: "var(--border-strong)",
  border: "none",
};
