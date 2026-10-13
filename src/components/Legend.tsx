import { useLang } from "../app/i18n";

const ROWS: { role: string; label: string; labelEn: string; icon: string }[] = [
  { role: "embedding", label: "埋め込み層", labelEn: "Embedding", icon: "◧" },
  { role: "attention", label: "Attention", labelEn: "Attention", icon: "◈" },
  { role: "mlp", label: "MLP", labelEn: "MLP", icon: "▦" },
  { role: "norm", label: "正規化層", labelEn: "Norm", icon: "≈" },
  { role: "lm_head", label: "LM Head", labelEn: "LM Head", icon: "◨" },
];

export function Legend() {
  const { t } = useLang();
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
      {ROWS.map(({ role, label, labelEn, icon }) => (
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
          <span style={{ color: "var(--text-secondary)" }}>{t(label, labelEn)}</span>
        </div>
      ))}
    </div>
  );
}
