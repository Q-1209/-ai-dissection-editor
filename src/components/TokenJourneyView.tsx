import type { NearestWordMatch } from "../electron.d";
import { useLang } from "../app/i18n";

interface TokenJourneyViewProps {
  blockOrder: string[];
  journey: Record<string, NearestWordMatch[][]>;
  tokenLabels: string[];
  // ブロックIDを短縮表示するための関数(例: "model.layers.3.self_attn" -> "L3-Attn")
  shortenBlockId: (blockId: string) => string;
}

// 類似度(-1〜1程度、実質0〜1)を色に変換する。低い=背景色、高い=アクセント色。
function similarityToColor(similarity: number): string {
  const clamped = Math.max(0, Math.min(1, similarity));
  const r = Math.round(0x17 + (0x5b - 0x17) * clamped);
  const g = Math.round(0x1d + (0x9b - 0x1d) * clamped);
  const b = Math.round(0x22 + (0xd5 - 0x22) * clamped);
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Token Journeyの結果を、トークン位置(行)×ブロック(列)の表として表示する。
 * 各セルには、そのブロック通過後にそのトークン位置が最も近いと判定された単語
 * (top-1)を表示する。セルの背景色はコサイン類似度の高さを表す。
 */
export function TokenJourneyView({
  blockOrder,
  journey,
  tokenLabels,
  shortenBlockId,
}: TokenJourneyViewProps) {
  const { t } = useLang();
  if (blockOrder.length === 0 || tokenLabels.length === 0) return null;

  return (
    <div style={{ overflow: "auto" }}>
      <table style={{ borderCollapse: "collapse", fontSize: 11 }}>
        <thead>
          <tr>
            <th
              style={{
                position: "sticky",
                left: 0,
                background: "var(--bg-panel)",
                padding: "6px 10px",
                textAlign: "left",
                color: "var(--text-secondary)",
                borderBottom: "1px solid var(--border-default)",
              }}
            >
              {t("トークン", "Token")}
            </th>
            {blockOrder.map((blockId) => (
              <th
                key={blockId}
                style={{
                  padding: "6px 8px",
                  textAlign: "center",
                  color: "var(--text-secondary)",
                  borderBottom: "1px solid var(--border-default)",
                  fontFamily: "var(--font-mono)",
                  whiteSpace: "nowrap",
                }}
              >
                {shortenBlockId(blockId)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {tokenLabels.map((label, tokenPos) => (
            <tr key={tokenPos}>
              <td
                style={{
                  position: "sticky",
                  left: 0,
                  background: "var(--bg-panel)",
                  padding: "6px 10px",
                  fontWeight: 600,
                  color: "var(--text-primary)",
                  borderBottom: "1px solid var(--border-default)",
                }}
              >
                {label}
              </td>
              {blockOrder.map((blockId) => {
                const matches = journey[blockId]?.[tokenPos];
                const top = matches?.[0];
                return (
                  <td
                    key={blockId}
                    style={{
                      padding: "6px 8px",
                      textAlign: "center",
                      borderBottom: "1px solid var(--border-default)",
                      background: top ? similarityToColor(top.similarity) : "transparent",
                      color: "#111",
                      minWidth: 60,
                    }}
                    title={
                      top
                        ? `similarity: ${top.similarity.toFixed(4)}\ntoken_id: ${top.token_id}`
                        : undefined
                    }
                  >
                    {top?.token_text ?? (top ? String(top.token_id) : "—")}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
