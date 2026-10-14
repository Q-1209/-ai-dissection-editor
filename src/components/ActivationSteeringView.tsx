import type { ActivationSteeringResult, SycophancyPhraseCount } from "../electron.d";
import { useLang } from "../app/i18n";

interface ActivationSteeringViewProps {
  result: ActivationSteeringResult;
}

/**
 * Activation Steering Studio(Phase3、2026-09-15実装)の結果表示。
 *
 * 2026-09-15確認事項に基づく設計:
 * - ステアリングベクトルは対照プロンプト群(positive/negative)の平均活性化値の
 *   差分方式("Contrastive Activation Addition"系)
 * - 適用対象はユーザーが選んだノード(1つ以上、Layer Ablation Wizardと同じ
 *   チェックボックス選択UX)
 * - 効果の確認: KLダイバージェンス(定量指標)+ 生成テキストの変化(目視)の両方
 *
 * レイアウトはLayerAblationViewを踏襲しつつ、ステアリング固有の情報
 * (各対象ノードのベクトルのノルム、適用したscale)を追加している。
 *
 * 2026-09-22追加(Refusal Behavior Tuner): 介入方式(加算/方向除去)とチャットテンプレート
 * 適用有無を上部に表示し、拒否定型句の出現回数カードを追加(迎合語彙カードと共通の
 * PhraseCountCardで表示)。
 */

function PhraseCountCard({
  title,
  baseline,
  steered,
}: {
  title: string;
  baseline: SycophancyPhraseCount | null;
  steered: SycophancyPhraseCount | null;
}) {
  const { t } = useLang();
  const detail = (c: SycophancyPhraseCount | null) =>
    c && Object.keys(c.by_phrase).length > 0
      ? Object.entries(c.by_phrase)
          .map(([k, v]) => `${k}×${v}`)
          .join(" / ")
      : "";
  return (
    <div
      style={{
        background: "var(--bg-panel-raised)",
        borderRadius: "var(--radius-md)",
        padding: 12,
        maxWidth: 460,
      }}
    >
      <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 6 }}>{title}</div>
      <div style={{ display: "flex", gap: 20, fontSize: 12.5 }}>
        <div>
          <span style={{ color: "var(--text-secondary)" }}>{t("通常時", "Normal")}: </span>
          <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
            {baseline?.total ?? "-"}
          </span>
        </div>
        <div>
          <span style={{ color: "var(--text-secondary)" }}>{t("ステアリング適用時", "Steered")}: </span>
          <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
            {steered?.total ?? "-"}
          </span>
        </div>
      </div>
      {(detail(baseline) || detail(steered)) && (
        <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 6, lineHeight: 1.6 }}>
          {detail(baseline) && <div>{t("通常時の内訳", "Normal breakdown")}: {detail(baseline)}</div>}
          {detail(steered) && <div>{t("適用時の内訳", "Steered breakdown")}: {detail(steered)}</div>}
        </div>
      )}
    </div>
  );
}
export function ActivationSteeringView({ result }: ActivationSteeringViewProps) {
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
      {result.warnings?.map((w) => (
        <div key={w} style={{ fontSize: 12, color: "#e0b45a", border: "1px solid #e0b45a55", borderRadius: "var(--radius-sm)", padding: "8px 10px" }}>
          ⚠ {w}
        </div>
      ))}
      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
        {t("ステアリング対象ノード", "Target nodes")}: {result.target_node_ids.join(", ")} (
        {result.intervention_mode === "ablate_direction" ? t("方向除去", "ablate direction") : `${t("加算", "add")} scale: ${result.scale}`}
        ) · {t("チャットテンプレート", "chat template")}: {result.used_chat_template ? t("適用", "on") : t("なし", "off")}
      </div>

      <div>
        <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 6 }}>
          {t("ステアリングベクトルのノルム（ベクトルの大きさの目安）", "Steering vector norm (rough size of the vector)")}
        </div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {Object.entries(result.vector_norms).map(([nodeId, norm]) => (
            <div
              key={nodeId}
              style={{
                background: "var(--bg-panel-raised)",
                borderRadius: "var(--radius-sm)",
                padding: "6px 10px",
                fontSize: 11.5,
              }}
            >
              <span style={{ color: "var(--text-secondary)" }}>{nodeId}</span>
              {"  "}
              <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
                {norm.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
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
            {t("次トークン分布のKLダイバージェンス(通常時→ステアリング適用時)", "KL divergence of the next-token distribution (normal → steered)")}
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
            {t("ステアリング適用時", "Steered")}: {result.steered_top_token_text ?? `ID ${result.steered_top_token_id}`}
          </div>
        </div>
      </div>

      {(result.sycophancy_phrase_count_baseline !== null ||
        result.sycophancy_phrase_count_steered !== null) && (
        <PhraseCountCard
          title={t("迎合的定型語彙の出現回数(機械的指標、Sycophancy Reducer)", "Count of agreeing stock phrases (simple metric, Sycophancy Reducer)")}
          baseline={result.sycophancy_phrase_count_baseline}
          steered={result.sycophancy_phrase_count_steered}
        />
      )}

      {(result.refusal_phrase_count_baseline !== null || result.refusal_phrase_count_steered !== null) && (
        <PhraseCountCard
          title={t("拒否定型句の出現回数(機械的指標、Refusal Behavior Tuner)", "Count of refusal stock phrases (simple metric, Refusal Tuner)")}
          baseline={result.refusal_phrase_count_baseline}
          steered={result.refusal_phrase_count_steered}
        />
      )}

      {(result.generated_text_baseline !== null || result.generated_text_steered !== null) && (
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
                {t("ステアリング適用時", "Steered")}
              </div>
              <div style={{ fontSize: 12.5, color: "var(--text-primary)", whiteSpace: "pre-wrap" }}>
                {result.generated_text_steered || t("(空)", "(empty)")}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
