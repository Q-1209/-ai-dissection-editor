import type { ActivationSteeringResult, SycophancyPhraseCount } from "../electron.d";

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
          <span style={{ color: "var(--text-secondary)" }}>通常時: </span>
          <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
            {baseline?.total ?? "-"}
          </span>
        </div>
        <div>
          <span style={{ color: "var(--text-secondary)" }}>ステアリング適用時: </span>
          <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
            {steered?.total ?? "-"}
          </span>
        </div>
      </div>
      {(detail(baseline) || detail(steered)) && (
        <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 6, lineHeight: 1.6 }}>
          {detail(baseline) && <div>通常時の内訳: {detail(baseline)}</div>}
          {detail(steered) && <div>適用時の内訳: {detail(steered)}</div>}
        </div>
      )}
    </div>
  );
}
export function ActivationSteeringView({ result }: ActivationSteeringViewProps) {
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
        ステアリング対象ノード: {result.target_node_ids.join(", ")}（
        {result.intervention_mode === "ablate_direction" ? "方向除去" : `加算 scale: ${result.scale}`}
        ）・チャットテンプレート: {result.used_chat_template ? "適用" : "なし"}
      </div>

      <div>
        <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 6 }}>
          ステアリングベクトルのノルム（ベクトルの大きさの目安）
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
            次トークン分布のKLダイバージェンス(通常時→ステアリング適用時)
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
            ステアリング適用時: {result.steered_top_token_text ?? `ID ${result.steered_top_token_id}`}
          </div>
        </div>
      </div>

      {(result.sycophancy_phrase_count_baseline !== null ||
        result.sycophancy_phrase_count_steered !== null) && (
        <PhraseCountCard
          title="迎合的定型語彙の出現回数(機械的指標、Sycophancy Reducer)"
          baseline={result.sycophancy_phrase_count_baseline}
          steered={result.sycophancy_phrase_count_steered}
        />
      )}

      {(result.refusal_phrase_count_baseline !== null || result.refusal_phrase_count_steered !== null) && (
        <PhraseCountCard
          title="拒否定型句の出現回数(機械的指標、Refusal Behavior Tuner)"
          baseline={result.refusal_phrase_count_baseline}
          steered={result.refusal_phrase_count_steered}
        />
      )}

      {(result.generated_text_baseline !== null || result.generated_text_steered !== null) && (
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
                ステアリング適用時
              </div>
              <div style={{ fontSize: 12.5, color: "var(--text-primary)", whiteSpace: "pre-wrap" }}>
                {result.generated_text_steered || "(空)"}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
