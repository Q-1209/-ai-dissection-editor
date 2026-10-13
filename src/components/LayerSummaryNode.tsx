import { Handle, Position } from "@xyflow/react";
import type { CSSProperties } from "react";
import { useLang } from "../app/i18n";

export interface LayerSummaryNodeData extends Record<string, unknown> {
  kind: "layerSummary";
  // 表示ラベル(例: "Layer 3", "Attention (Layer 3)", "MLP (Layer 3)")
  label: string;
  nodeCount: number;
  isExpanded: boolean;
  // このノードをクリックしたときに呼ぶべきグループの識別子。
  // 折りたたみ状態の管理キーとしてApp.tsx側で使う
  // (例: "layer-3"はLayer全体、"layer-3-attention"はAttentionブロックのみ)
  groupKey: string;
}

export function LayerSummaryNode({ data }: { data: LayerSummaryNodeData }) {
  const { t } = useLang();
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
        <span style={{ fontSize: 12, fontWeight: 600 }}>{data.label}</span>
      </div>
      <div style={{ fontSize: 10.5, opacity: 0.75, marginTop: 4 }}>
        {data.isExpanded
          ? t("クリックで折りたたむ", "Click to collapse")
          : t(`クリックで展開 · ${data.nodeCount}ノード`, `Click to expand · ${data.nodeCount} nodes`)}
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
