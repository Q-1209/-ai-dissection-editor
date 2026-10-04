import type { LayerAblationResult } from "../electron.d";

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
  if (!result.success) {
    return (
      <div style={{ color: "#e08a8a", fontSize: 12 }}>
        エラー ({result.error_type}): {result.message}
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
        無効化したノード: {result.ablated_node_ids.join(", ")}
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
            次トークン分布のKLダイバージェンス(通常時→無効化時)
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
            最有力トークンの変化
          </div>
          <div
            style={{
              fontSize: 14,
              fontWeight: 600,
              color: result.top_token_changed ? "#e0b45a" : "var(--text-primary)",
            }}
          >
            {result.top_token_changed ? "変化あり" : "変化なし"}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 6 }}>
            通常時: {result.baseline_top_token_text ?? `ID ${result.baseline_top_token_id}`}
            {" → "}
            無効化時: {result.ablated_top_token_text ?? `ID ${result.ablated_top_token_id}`}
          </div>
        </div>
      </div>

      {(result.generated_text_baseline !== null || result.generated_text_ablated !== null) && (
        <div>
          <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 6 }}>
            生成テキストの比較
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
              <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>通常時</div>
              <div style={{ fontSize: 12.5, color: "var(--text-primary)", whiteSpace: "pre-wrap" }}>
                {result.generated_text_baseline || "(空)"}
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
                無効化時
              </div>
              <div style={{ fontSize: 12.5, color: "var(--text-primary)", whiteSpace: "pre-wrap" }}>
                {result.generated_text_ablated || "(空)"}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
