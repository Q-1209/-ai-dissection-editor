import type { BattleModelResult } from "../electron.d";
import { ActivationStatsList } from "./ActivationStatsList";
import { useLang } from "../app/i18n";

interface BattleViewProps {
  results: BattleModelResult[];
  activationScale: "linear" | "log";
}

/**
 * Multi-Model Battle View の結果を、モデルごとに縦カラムで横並び表示する。
 *
 * 2026-09-10確認事項により、異なるアーキテクチャ・異なるhidden_sizeのモデル
 * 同士を比較することも想定しているため、モデル間で活性化値統計やAttention重みの
 * 値をそのまま重ね合わせたり同じ軸で比較したりはしない(スケールも行列サイズも
 * 揃わないため)。各モデルの結果は完全に独立したカラムとして表示し、
 * 「目視で見比べる」ことに徹する設計。
 *
 * Attentionヒートマップは1モデルぶんでも情報量が多いため、Battle View(MVP)では
 * 生成結果と層ごとの活性化値統計(std)の比較に絞り、Attention重みそのものの
 * 可視化は含めない(データはattention_weightsとして結果に含まれているので、
 * 将来的にモデルごとのヒートマップ表示を追加する余地はある)。
 */
export function BattleView({ results, activationScale }: BattleViewProps) {
  const { t } = useLang();
  if (results.length === 0) return null;

  return (
    <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-start" }}>
      {results.map((r, index) => (
        <div
          key={`${r.model_dir}-${index}`}
          style={{
            background: "var(--bg-panel)",
            border: "1px solid var(--border-default)",
            borderRadius: "var(--radius-md)",
            padding: 14,
            width: 320,
            flex: "0 0 auto",
          }}
        >
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontWeight: 600, fontSize: 13, color: "var(--text-primary)" }}>
              {r.label}
            </div>
            <div
              style={{
                fontSize: 10,
                color: "var(--text-muted)",
                fontFamily: "var(--font-mono)",
                wordBreak: "break-all",
              }}
            >
              {r.model_dir}
            </div>
          </div>

          {!r.success && (
            <div style={{ color: "#e08a8a", fontSize: 12 }}>
              {t("エラー", "Error")} ({r.error_type}): {r.message}
            </div>
          )}

          {r.success && (
            <>
              {r.generated_text !== null && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 4 }}>
                    {t("生成結果", "Generation")}
                  </div>
                  <div
                    style={{
                      background: "var(--bg-panel-raised)",
                      borderRadius: "var(--radius-sm)",
                      padding: "8px 10px",
                      fontSize: 12.5,
                      color: "var(--text-primary)",
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                    }}
                  >
                    {r.generated_text || t("(空)", "(empty)")}
                  </div>
                </div>
              )}

              {r.generated_token_ids !== null && (
                <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 12 }}>
                  {t("生成トークン数", "Generated tokens")}: {r.generated_token_ids.length} / {t("ブロック数", "Blocks")}:{" "}
                  {r.executed_order.length}
                </div>
              )}

              {r.activation_stats !== null && (
                <div>
                  <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 6 }}>
                    {t("層ごとの活性化値統計(std)", "Activation stats per layer (std)")}
                  </div>
                  <ActivationStatsList
                    stats={r.activation_stats}
                    order={r.executed_order}
                    scale={activationScale}
                  />
                </div>
              )}
            </>
          )}
        </div>
      ))}
    </div>
  );
}
