import type { LayerAblationResult } from "../electron.d";
import { useLang } from "../app/i18n";

interface LayerAblationViewProps {
  result: LayerAblationResult;
}

/**
 * Layer Ablation Wizard(2026-09-12実装)の結果表示。
 *
 * 2026-09-12確認事項に基づく設計:
 * - 「無効化」= ノードの出力をゼロにする(残差接続も含めて寄与を完全に断つ)
 * - 「全層自動バッチ処理」ではなく、ユーザーが選んだノード(1つ以上)を無効化した
 *   状態と通常時とを1回だけ比較する対話的ツール(おすすめで、と一任された部分)
 * - 測定指標: KLダイバージェンス + 生成テキストの変化(両方)
 */
export function LayerAblationView({ result }: LayerAblationViewProps) {
  const { t } = useLang();
  if (!result.success) {
    return (
      <div style={{ color: "#e08a8a", fontSize: 12 }}>
        {t("エラー", "Error")} ({result.error_type}): {result.message}
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
        {t("無効化したノード", "Ablated nodes")}: {result.ablated_node_ids.join(", ")}
      </div>

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <div
          style={{
            background: "var(--bg-panel-raised)",
            borderRadius: "var(--radius-md)",
            padding: 12,
            minWidth: 220,
          }}
        >
          <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 4 }}>
            {t("次トークン分布のKLダイバージェンス(通常時→無効化時)", "KL divergence of the next-token distribution (normal → ablated)")}
          </div>
          <div style={{ fontSize: 22, fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
            {result.kl_divergence.toFixed(4)}
          </div>
        </div>

        <div
          style={{
            background: "var(--bg-panel-raised)",
            borderRadius: "var(--radius-md)",
            padding: 12,
            minWidth: 220,
          }}
        >
          <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 4 }}>
            {t("最有力トークンの変化", "Top token")}
          </div>
          <div
            style={{
              fontSize: 14,
              fontWeight: 600,
              color: result.top_token_changed ? "#e0b45a" : "var(--text-primary)",
            }}
          >
            {result.top_token_changed ? t("変化あり", "Changed") : t("変化なし", "Unchanged")}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 6 }}>
            {t("通常時", "Normal")}: {result.baseline_top_token_text ?? `ID ${result.baseline_top_token_id}`}
            {" → "}
            {t("無効化時", "Ablated")}: {result.ablated_top_token_text ?? `ID ${result.ablated_top_token_id}`}
          </div>
        </div>
      </div>

      {(result.generated_text_baseline !== null || result.generated_text_ablated !== null) && (
        <div>
          <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 6 }}>
            {t("生成テキストの比較", "Generated text")}
          </div>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <div
              style={{
                flex: "1 1 260px",
                background: "var(--bg-panel-raised)",
                borderRadius: "var(--radius-sm)",
                padding: "8px 10px",
              }}
            >
              <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>{t("通常時", "Normal")}</div>
              <div style={{ fontSize: 12.5, color: "var(--text-primary)", whiteSpace: "pre-wrap" }}>
                {result.generated_text_baseline || t("(空)", "(empty)")}
              </div>
            </div>
            <div
              style={{
                flex: "1 1 260px",
                background: "var(--bg-panel-raised)",
                borderRadius: "var(--radius-sm)",
                padding: "8px 10px",
                border: "1px solid var(--accent)",
              }}
            >
              <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>
                {t("無効化時", "Ablated")}
              </div>
              <div style={{ fontSize: 12.5, color: "var(--text-primary)", whiteSpace: "pre-wrap" }}>
                {result.generated_text_ablated || t("(空)", "(empty)")}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
