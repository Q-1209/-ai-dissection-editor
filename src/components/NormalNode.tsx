import { Handle, Position } from "@xyflow/react";
import type { CSSProperties } from "react";

export interface NormalNodeData extends Record<string, unknown> {
  kind: "normal";
  displayName: string; // 短縮された表示名(例: "Q projection")
  role: string;
  roleLabelJa: string;
  shape: number[] | null;
  paramKey: string;
  paramCount: number | null;
}

const ROLE_ICONS: Record<string, string> = {
  embedding: "◧", // 入力
  attention: "◈", // 注意機構
  mlp: "▦", // 変換
  norm: "≈", // 正規化
  lm_head: "◨", // 出力
  other: "◌",
};

function formatParamCount(n: number | null): string {
  if (n === null) return "";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M params`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K params`;
  return `${n} params`;
}

export function NormalNode({ data }: { data: NormalNodeData }) {
  const roleColorVar = `--role-${data.role}`;
  const style: CSSProperties = {
    background: `var(${roleColorVar}-bg, var(--role-other-bg))`,
    border: `1px solid var(${roleColorVar}-border, var(--role-other-border))`,
    color: `var(${roleColorVar}-text, var(--role-other-text))`,
    borderRadius: "var(--radius-sm)",
    padding: "8px 10px",
    width: 220,
    fontFamily: "var(--font-ui)",
  };

  return (
    <div style={style}>
      <Handle type="target" position={Position.Left} style={handleStyle} />
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontSize: 13, opacity: 0.85 }} aria-hidden="true">
          {ROLE_ICONS[data.role] ?? ROLE_ICONS.other}
        </span>
        <span style={{ fontSize: 12, fontWeight: 600, lineHeight: 1.3 }}>
          {data.displayName}
        </span>
      </div>
      {data.shape && (
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10.5,
            opacity: 0.75,
            marginTop: 4,
          }}
        >
          shape: [{data.shape.join(", ")}]
        </div>
      )}
      {data.paramCount !== null && (
        <div style={{ fontFamily: "var(--font-mono)", fontSize: 10.5, opacity: 0.6, marginTop: 2 }}>
          {formatParamCount(data.paramCount)}
        </div>
      )}
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
