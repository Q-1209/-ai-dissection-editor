import type { NeuronSearchResult } from "../electron.d";
import { useLang } from "../app/i18n";

interface NeuronSearchViewProps {
  result: NeuronSearchResult;
  // rank結果の行をクリックしたときに、そのニューロンをprobeモードへ引き継ぐためのコールバック
  // (「このニューロンを他の入力でも試す」というワークフローを想定)
  onProbeThisNeuron?: (layerIndex: number, kind: "mlp" | "attention_head", neuronIndex: number) => void;
}

/**
 * Neuron Search(2026-09-12実装)の結果表示。
 *
 * 2026-09-12確認事項: 「ニューロン」はMLP中間次元・Attentionヘッドの両方に対応、
 * 探索方向は(a)入力に対するランキング(rank)・(b)特定ニューロンの複数入力への
 * 反応比較(probe)の両方に対応する。このコンポーネントは両モードの結果表示を
 * まとめて担当する(resultのmodeフィールドで分岐)。
 */
export function NeuronSearchView({ result, onProbeThisNeuron }: NeuronSearchViewProps) {
  const { t } = useLang();
  if (!result.success) {
    return (
      <div style={{ color: "#e08a8a", fontSize: 12 }}>
        {t("エラー", "Error")} ({result.error_type}): {result.message}
      </div>
    );
  }

  if (result.mode === "rank") {
    if (result.ranked_neurons.length === 0) {
      return (
        <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
          {t("該当するニューロンが見つかりませんでした(対象の層・種類を見直してください)", "No matching neurons (check the layers and type)")}
        </div>
      );
    }
    return (
      <div>
        <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>
          {t(`入力に強く反応した上位${result.ranked_neurons.length}件(スコア=系列内での活性化値の絶対値集約)`, `Top ${result.ranked_neurons.length} neurons for this input (score = aggregated |activation| over the sequence)`)}
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ textAlign: "left", color: "var(--text-muted)", fontSize: 10.5 }}>
              <th style={thStyle}>#</th>
              <th style={thStyle}>{t("層", "Layer")}</th>
              <th style={thStyle}>{t("種類", "Type")}</th>
              <th style={thStyle}>{t("ニューロン番号", "Neuron")}</th>
              <th style={thStyle}>{t("スコア", "Score")}</th>
              {onProbeThisNeuron && <th style={thStyle} />}
            </tr>
          </thead>
          <tbody>
            {result.ranked_neurons.map((entry, i) => (
              <tr
                key={`${entry.layer_index}-${entry.kind}-${entry.neuron_index}`}
                style={{ borderTop: "1px solid var(--border-default)" }}
              >
                <td style={tdStyle}>{i + 1}</td>
                <td style={tdStyle}>{entry.layer_index}</td>
                <td style={tdStyle}>{entry.kind === "mlp" ? "MLP" : t("Attentionヘッド", "Attention head")}</td>
                <td style={tdStyle}>{entry.neuron_index}</td>
                <td style={{ ...tdStyle, fontFamily: "var(--font-mono)" }}>
                  {entry.score.toFixed(4)}
                </td>
                {onProbeThisNeuron && (
                  <td style={tdStyle}>
                    <button
                      onClick={() => onProbeThisNeuron(entry.layer_index, entry.kind, entry.neuron_index)}
                      style={{
                        fontSize: 10.5,
                        padding: "2px 6px",
                        borderRadius: "var(--radius-sm)",
                        border: "1px solid var(--border-strong)",
                        background: "var(--bg-panel-raised)",
                        color: "var(--text-primary)",
                        cursor: "pointer",
                      }}
                    >
                      {t("Probeへ", "Probe")}
                    </button>
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  // mode === "probe"
  const maxScore = Math.max(1e-9, ...result.results.map((r) => Math.abs(r.score)));
  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>
        {t("層", "Layer ")}{result.layer_index} / {result.kind === "mlp" ? "MLP" : t("Attentionヘッド", "Attention head")} /{" "}
        {t("ニューロン番号", "neuron")} {result.neuron_index} — {t("各入力文への反応の強さ", "response to each input")}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {result.results.map((r, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div
              style={{
                width: 160,
                fontSize: 12,
                color: "var(--text-primary)",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
              title={r.input_text}
            >
              {r.input_text}
            </div>
            <div
              style={{
                flex: 1,
                height: 14,
                background: "var(--bg-panel-raised)",
                borderRadius: "var(--radius-sm)",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${(Math.abs(r.score) / maxScore) * 100}%`,
                  height: "100%",
                  background: "var(--accent)",
                }}
              />
            </div>
            <div style={{ width: 60, fontSize: 11, fontFamily: "var(--font-mono)", textAlign: "right" }}>
              {r.score.toFixed(4)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const thStyle = { padding: "4px 8px" } as const;
const tdStyle = { padding: "4px 8px" } as const;
