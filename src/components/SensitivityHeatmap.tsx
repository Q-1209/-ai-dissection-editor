import type { SensitivityTokenResult } from "../electron.d";

interface SensitivityHeatmapProps {
  results: SensitivityTokenResult[];
  tokenLabels: string[];
}

// KLダイバージェンス値を色に変換する。0=背景色、maxKl=アクセント色(オレンジ寄り)に近づく。
function valueToColor(value: number, maxValue: number): string {
  const clamped = maxValue > 0 ? Math.max(0, Math.min(1, value / maxValue)) : 0;
  // 背景(var(--bg-app) #0f1418)から、影響度が大きい語ほど暖色(#d4a24e、Attention色と同系統)に近づける
  const r = Math.round(0x0f + (0xd4 - 0x0f) * clamped);
  const g = Math.round(0x14 + (0xa2 - 0x14) * clamped);
  const b = Math.round(0x18 + (0x4e - 0x18) * clamped);
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Prompt Sensitivity Mapの結果を、トークンごとの重要度バーとして横に並べて表示する。
 * KLダイバージェンスが大きいトークン(=そのトークンを変更すると出力が大きく変わる)ほど
 * 濃い色で強調される。最上位予測トークンが変わった位置には印を付ける。
 */
export function SensitivityHeatmap({ results, tokenLabels }: SensitivityHeatmapProps) {
  if (results.length === 0) return null;

  const maxKl = Math.max(...results.map((r) => r.kl_divergence), 1e-6);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
        {results.map((r) => {
          const label = tokenLabels[r.position] ?? String(r.position);
          return (
            <div
              key={r.position}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                minWidth: 40,
              }}
            >
              <div
                style={{
                  padding: "8px 10px",
                  background: valueToColor(r.kl_divergence, maxKl),
                  borderRadius: "var(--radius-sm)",
                  border: r.top_token_changed
                    ? "2px solid var(--accent)"
                    : "1px solid var(--border-default)",
                  fontSize: 13,
                  color: "#111",
                  fontWeight: 600,
                  cursor: "default",
                }}
                title={
                  `position: ${r.position}\n` +
                  `KL divergence: ${r.kl_divergence.toFixed(4)}\n` +
                  `top token changed: ${r.top_token_changed ? "yes" : "no"}` +
                  (r.original_top_token_text
                    ? `\noriginal next: ${r.original_top_token_text} -> modified next: ${r.modified_top_token_text}`
                    : "")
                }
              >
                {label}
              </div>
              <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 2 }}>
                {r.kl_divergence.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 4 }}>
        色が濃い（オレンジ寄り）ほど、そのトークンを変更したときの出力への影響が大きいことを示す。
        枠が太い（アクセント色）トークンは、変更により次トークンの最上位予測が変わったことを示す。
      </div>
    </div>
  );
}
