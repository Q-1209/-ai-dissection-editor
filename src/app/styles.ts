// App.tsx共通のインラインスタイル(2026-09-22、App.tsx分割の第1段階で切り出し)。

export const buttonStyle = {
  padding: "6px 12px",
  fontSize: 12,
  borderRadius: "var(--radius-sm)",
  border: "1px solid var(--border-strong)",
  background: "var(--bg-panel-raised)",
  color: "var(--text-primary)",
  cursor: "pointer",
} as const;

export const tabButtonStyle = {
  ...buttonStyle,
  border: "1px solid transparent",
  background: "transparent",
  color: "var(--text-secondary)",
} as const;

export const tabButtonActiveStyle = {
  ...buttonStyle,
  background: "var(--bg-hover)",
  color: "var(--text-primary)",
  border: "1px solid var(--border-strong)",
} as const;

export const infoPanelStyle = {
  position: "absolute",
  top: 16,
  right: 16,
  width: 300,
  background: "var(--bg-panel)",
  border: "1px solid var(--border-default)",
  color: "var(--text-primary)",
  borderRadius: "var(--radius-md)",
  padding: 14,
  fontSize: 12,
  zIndex: 10,
} as const;

export const emptyStateStyle = {
  position: "absolute",
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  color: "var(--text-muted)",
  fontSize: 14,
  pointerEvents: "none",
} as const;
