const ROWS: { role: string; label: string; icon: string }[] = [
  { role: "embedding", label: "埋め込み層", icon: "◧" },
  { role: "attention", label: "Attention", icon: "◈" },
  { role: "mlp", label: "MLP", icon: "▦" },
  { role: "norm", label: "正規化層", icon: "≈" },
  { role: "lm_head", label: "LM Head", icon: "◨" },
];

export function Legend() {
  return (
    <div
      style={{
        position: "absolute",
        bottom: 16,
        left: 16,
        background: "var(--bg-panel)",
        border: "1px solid var(--border-default)",
        borderRadius: "var(--radius-md)",
        padding: "10px 14px",
        display: "flex",
        flexDirection: "column",
        gap: 6,
        zIndex: 10,
        fontSize: 12,
        color: "var(--text-primary)",
      }}
    >
      {ROWS.map(({ role, label, icon }) => (
        <div key={role} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              width: 18,
              height: 18,
              borderRadius: 4,
              background: `var(--role-${role}-bg)`,
              border: `1px solid var(--role-${role}-border)`,
              color: `var(--role-${role}-text)`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 11,
            }}
            aria-hidden="true"
          >
            {icon}
          </span>
          <span style={{ color: "var(--text-secondary)" }}>{label}</span>
        </div>
      ))}
    </div>
  );
}
