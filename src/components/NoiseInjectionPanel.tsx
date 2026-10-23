import { useCallback, useState } from "react";
import type {
  ExecutionEdge,
  ExecutionNode,
  LiveSweepPoint,
  NoiseInjectionMode,
  NoiseInjectionResult,
} from "../electron.d";
import { buttonStyle, tabButtonActiveStyle, tabButtonStyle } from "../app/styles";
import { SweepChart } from "./SweepChart";
import { useLang } from "../app/i18n";

/**
 * Noise Injection(2026-10-24追加、長編動画M4「AIを酔っぱらわせろ」用)。
 *
 * 2026-10-24、漣さんとの確認事項:
 * - 観察タブの新パネル。Layer Ablation Wizardと同じ「層のチェックボックス→実行→通常時との比較」
 * - 強さ = その位置の内部状態の大きさに対する割合(0.1 = 10%)
 * - スイープ: 強さを段階的に上げる比較と、同じ強さで各層を1つずつ試す比較の両方
 * - 乱数: 1語ごとに新しい雑音。乱数の種(seed)を固定するので、同じ設定なら同じ結果
 *
 * python/run_noise_injection.py を呼ぶ。状態はこのコンポーネントの中で完結させている(App.tsxの肥大化を避けるため)。
 */

interface NoiseInjectionPanelProps {
  modelDir: string;
  config: Record<string, unknown>;
  nodes: ExecutionNode[];
  edges: ExecutionEdge[];
  useChatTemplate: boolean;
  selectableNodes: ExecutionNode[]; // attention_block/mlp_block/layer_block(Layer Ablationと同じ)
}

type RunState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "done"; result: NoiseInjectionResult };

const DEFAULT_STRENGTHS_TEXT = "0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1";

export function NoiseInjectionPanel({ modelDir, config, nodes, edges, useChatTemplate, selectableNodes }: NoiseInjectionPanelProps) {
  const { t, lang } = useLang();
  const [inputText, setInputText] = useState(() => (lang === "en" ? "Introduce yourself." : "自己紹介してください。"));
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [mode, setMode] = useState<NoiseInjectionMode>("single");
  const [strength, setStrength] = useState(0.3);
  const [strengthsText, setStrengthsText] = useState(DEFAULT_STRENGTHS_TEXT);
  const [seed, setSeed] = useState(0);
  const [compareGeneration, setCompareGeneration] = useState(true);
  const [maxNewTokens, setMaxNewTokens] = useState(30);
  const [runState, setRunState] = useState<RunState>({ status: "idle" });

  const toggle = useCallback((id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const parsedStrengths = strengthsText
    .split(/[,、\s]+/)
    .filter(Boolean)
    .map(Number);
  const strengthsValid = parsedStrengths.length > 0 && parsedStrengths.every((v) => Number.isFinite(v) && v >= 0 && v <= 10);

  const handleRun = useCallback(async () => {
    if (selected.size === 0) return;
    setRunState({ status: "running" });
    try {
      // 選んだ順ではなくグラフの順(層の順)に並べる。層スイープの横軸がそのまま層の順になる
      const targetNodeIds = selectableNodes.filter((n) => selected.has(n.id)).map((n) => n.id);
      const result = await window.electronAPI.runNoiseInjection({
        modelDir,
        config,
        nodes,
        edges,
        useChatTemplate,
        inputText: inputText.trim() || undefined,
        targetNodeIds,
        mode,
        strength,
        strengths: mode === "strength_sweep" ? parsedStrengths : undefined,
        seed,
        compareGeneration,
        maxNewTokens,
      });
      setRunState({ status: "done", result });
    } catch (err) {
      setRunState({
        status: "done",
        result: { success: false, error_type: "UnexpectedError", message: err instanceof Error ? err.message : String(err) },
      });
    }
  }, [selected, selectableNodes, modelDir, config, nodes, edges, useChatTemplate, inputText, mode, strength,
    parsedStrengths, seed, compareGeneration, maxNewTokens]);

  const inputStyle = { ...buttonStyle, cursor: "text", background: "var(--bg-app)" } as const;
  const labelStyle = { display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--text-secondary)" } as const;
  const modes: Array<[NoiseInjectionMode, string]> = [
    ["single", t("1回比較", "Single")],
    ["strength_sweep", t("強さスイープ", "Strength sweep")],
    ["layer_sweep", t("層スイープ", "Layer sweep")],
  ];

  return (
    <div style={{ width: "100%", height: "100%", overflow: "auto", padding: 20, display: "flex", gap: 20 }}>
      <div style={{ flex: "0 0 260px" }}>
        <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>
          {t("雑音を足すノードを選択(「実行」タブで組んだ現在のグラフ構造から選べます)", "Pick nodes to add noise to (from the graph you built in the Run tab)")}
        </div>
        <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
          <button style={buttonStyle} onClick={() => setSelected(new Set(selectableNodes.map((n) => n.id)))}>
            {t("全選択", "Select all")}
          </button>
          <button style={buttonStyle} onClick={() => setSelected(new Set())}>
            {t("全解除", "Clear")}
          </button>
        </div>
        {selectableNodes.length === 0 ? (
          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
            {t("選べるノードがありません(モデルを読み込んでください)", "No nodes available (load a model first)")}
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: "70vh", overflow: "auto",
            background: "var(--bg-panel)", border: "1px solid var(--border-default)", borderRadius: "var(--radius-md)", padding: 10 }}>
            {selectableNodes.map((n) => (
              <label key={n.id} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-primary)", cursor: "pointer" }}>
                <input type="checkbox" checked={selected.has(n.id)} onChange={() => toggle(n.id)} />
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{n.id}</span>
              </label>
            ))}
          </div>
        )}
      </div>

      <div style={{ flex: "1 1 auto", minWidth: 320, display: "flex", flexDirection: "column", gap: 14 }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
          <input type="text" value={inputText} onChange={(e) => setInputText(e.target.value)}
            placeholder={t("入力テキスト", "Input text")} style={{ ...inputStyle, width: 220 }} />
          <div style={{ display: "flex", gap: 2 }}>
            {modes.map(([m, label]) => (
              <button key={m} onClick={() => setMode(m)} style={mode === m ? tabButtonActiveStyle : tabButtonStyle}>
                {label}
              </button>
            ))}
          </div>
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 14, alignItems: "center" }}>
          {mode === "strength_sweep" ? (
            <label style={labelStyle}>
              {t("強さの並び(カンマ区切り)", "Strengths (comma-separated)")}
              <input type="text" value={strengthsText} onChange={(e) => setStrengthsText(e.target.value)}
                style={{ ...inputStyle, width: 240, borderColor: strengthsValid ? undefined : "#e08a8a" }} />
            </label>
          ) : (
            <label style={labelStyle} title={t("内部状態の大きさに対する雑音の割合(0.1 = 10%)", "Noise size relative to the hidden state (0.1 = 10%)")}>
              {t("雑音の強さ", "Noise strength")}
              <input type="range" min={0} max={2} step={0.05} value={strength} onChange={(e) => setStrength(Number(e.target.value))} />
              <input type="number" min={0} max={10} step={0.05} value={strength} onChange={(e) => setStrength(Number(e.target.value))}
                style={{ ...inputStyle, width: 70 }} />
              <span style={{ color: "var(--text-muted)" }}>({Math.round(strength * 100)}%)</span>
            </label>
          )}
          <label style={labelStyle} title={t("乱数の種。同じ値なら同じ雑音になる", "Random seed. The same value gives the same noise")}>
            seed
            <input type="number" value={seed} onChange={(e) => setSeed(Math.trunc(Number(e.target.value)))} style={{ ...inputStyle, width: 70 }} />
          </label>
          <label style={{ ...labelStyle, cursor: "pointer" }}>
            <input type="checkbox" checked={compareGeneration} onChange={(e) => setCompareGeneration(e.target.checked)} />
            {t("生成テキストも比較", "Also compare generated text")}
          </label>
          {compareGeneration && (
            <input type="number" value={maxNewTokens} min={1} onChange={(e) => setMaxNewTokens(Number(e.target.value))}
              title={t("生成トークン数", "Tokens to generate")} style={{ ...inputStyle, width: 60 }} />
          )}
          <button onClick={handleRun}
            disabled={runState.status === "running" || selected.size === 0 || (mode === "strength_sweep" && !strengthsValid)}
            style={{ ...buttonStyle, background: "var(--accent)", color: "#0f1418", fontWeight: 600 }}>
            {runState.status === "running" ? t("実行中...", "Running...") : t("雑音を足して比較", "Add noise and compare")}
          </button>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {t(`${selected.size}個選択中`, `${selected.size} selected`)}
            {mode === "layer_sweep" && compareGeneration && selected.size > 4 &&
              t("(層スイープ+生成は時間がかかります)", " (layer sweep with generation takes a while)")}
          </span>
        </div>

        {runState.status === "idle" && (
          <div style={{ color: "var(--text-muted)", fontSize: 13 }}>
            {t("ノードを選び、強さを決めて「雑音を足して比較」を押すと、通常時との違いが表示されます。強さスイープはどこで文章が崩れるか、層スイープはどの層が雑音に弱いかを見られます",
              "Pick nodes, set a strength and press “Add noise and compare”. The strength sweep shows where the text falls apart; the layer sweep shows which layers are most sensitive to noise")}
          </div>
        )}
        {runState.status === "running" && <div style={{ color: "var(--text-secondary)" }}>{t("実行中...", "Running...")}</div>}
        {runState.status === "done" && <NoiseResultView result={runState.result} selectableNodes={selectableNodes} />}
      </div>
    </div>
  );
}

function NoiseResultView({ result, selectableNodes }: { result: NoiseInjectionResult; selectableNodes: ExecutionNode[] }) {
  const { t } = useLang();
  if (!result.success) {
    return (
      <div style={{ color: "#e08a8a", fontSize: 12 }}>
        {t("エラー", "Error")} ({result.error_type}): {result.message}
      </div>
    );
  }
  const card = { background: "var(--bg-panel-raised)", borderRadius: "var(--radius-md)", padding: 12, minWidth: 220 } as const;
  const textBox = { flex: "1 1 260px", background: "var(--bg-panel-raised)", borderRadius: "var(--radius-sm)", padding: "8px 10px" } as const;

  if (result.mode === "single") {
    const p = result.points[0];
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
          {t("雑音を足したノード", "Noisy nodes")}: {result.target_node_ids.join(", ")} / {t("強さ", "strength")} {p.strength} / seed {result.seed}
        </div>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <div style={card}>
            <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 4 }}>
              {t("次トークン分布のKLダイバージェンス(通常時→雑音あり)", "KL divergence of the next-token distribution (normal → noisy)")}
            </div>
            <div style={{ fontSize: 22, fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>{p.kl_divergence.toFixed(4)}</div>
          </div>
          <div style={card}>
            <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 4 }}>{t("最有力トークンの変化", "Top token")}</div>
            <div style={{ fontSize: 14, fontWeight: 600, color: p.top_token_changed ? "#e0b45a" : "var(--text-primary)" }}>
              {p.top_token_changed ? t("変化あり", "Changed") : t("変化なし", "Unchanged")}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 6 }}>
              {t("通常時", "Normal")}: {JSON.stringify(result.baseline_top_token_text)} ({(result.baseline_top_prob * 100).toFixed(1)}%)
              {" → "}
              {t("雑音あり", "Noisy")}: {JSON.stringify(p.top_token_text)} ({(p.top_prob * 100).toFixed(1)}%)
            </div>
          </div>
        </div>
        {(result.generated_text_baseline !== null || p.generated_text !== null) && (
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <div style={textBox}>
              <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>{t("通常時", "Normal")}</div>
              <div style={{ fontSize: 12.5, color: "var(--text-primary)", whiteSpace: "pre-wrap" }}>
                {result.generated_text_baseline || t("(空)", "(empty)")}
              </div>
            </div>
            <div style={{ ...textBox, border: "1px solid var(--accent)" }}>
              <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>{t("雑音あり", "Noisy")}</div>
              <div style={{ fontSize: 12.5, color: "var(--text-primary)", whiteSpace: "pre-wrap" }}>
                {p.generated_text || t("(空)", "(empty)")}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // スイープ: 横軸は強さ、または層番号(層番号が重複する展開済みノードでは選んだノードの順番)
  const isLayer = result.mode === "layer_sweep";
  const layerIndexOf = new Map(selectableNodes.map((n) => [n.id, n.layer_index]));
  const layerIdx = result.points.map((p) => (p.node_id ? layerIndexOf.get(p.node_id) ?? null : null));
  const useLayerIndex = isLayer && layerIdx.every((v) => v !== null) && new Set(layerIdx).size === layerIdx.length;
  const chartPoints: LiveSweepPoint[] = result.points.map((p, i) => ({
    scale: isLayer ? (useLayerIndex ? (layerIdx[i] as number) : i) : p.strength,
    kl: p.kl_divergence,
    top_token_text: p.top_token_text ?? "",
    top_prob: p.top_prob,
    baseline_top_prob: 0,
  }));
  const xLabel = isLayer
    ? useLayerIndex ? t("雑音を足した層", "Noisy layer") : t("選んだノードの順番", "Selected node #")
    : t("雑音の強さ", "Noise strength");
  const most = isLayer ? result.points.reduce((a, b) => (b.kl_divergence > a.kl_divergence ? b : a)) : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
        {isLayer
          ? t(`各ノードに1つずつ強さ${result.strength}の雑音`, `Noise of strength ${result.strength} on each node, one at a time`)
          : t(`選んだ${result.target_node_ids.length}個のノードすべてに同時に雑音`, `Noise on all ${result.target_node_ids.length} selected nodes at once`)}
        {" / seed "}{result.seed}
        {most && t(` / いちばん雑音に弱い: ${most.node_id}(KL ${most.kl_divergence.toFixed(3)})`,
          ` / most sensitive: ${most.node_id} (KL ${most.kl_divergence.toFixed(3)})`)}
      </div>
      <SweepChart points={chartPoints} currentScale={null} xLabel={xLabel}
        pointName={isLayer ? t("層", "Layer") : t("強さ", "Strength")} />
      {result.generated_text_baseline !== null && (
        <div style={{ ...textBox, flex: "none" }}>
          <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>{t("通常時", "Normal")}</div>
          <div style={{ fontSize: 12.5, color: "var(--text-primary)", whiteSpace: "pre-wrap" }}>{result.generated_text_baseline || t("(空)", "(empty)")}</div>
        </div>
      )}
      <table style={{ borderCollapse: "collapse", fontSize: 12, width: "100%" }}>
        <thead>
          <tr style={{ color: "var(--text-muted)", textAlign: "left" }}>
            <th style={{ padding: "4px 8px" }}>{isLayer ? t("ノード", "Node") : t("強さ", "Strength")}</th>
            <th style={{ padding: "4px 8px" }}>KL</th>
            <th style={{ padding: "4px 8px" }}>{t("最有力", "Top")}</th>
            <th style={{ padding: "4px 8px" }}>{t("生成テキスト", "Generated text")}</th>
          </tr>
        </thead>
        <tbody>
          {result.points.map((p, i) => (
            <tr key={i} style={{ borderTop: "1px solid var(--border-default)", verticalAlign: "top" }}>
              <td style={{ padding: "4px 8px", fontFamily: "var(--font-mono)", whiteSpace: "nowrap" }}>
                {isLayer ? p.node_id : `${p.strength} (${Math.round(p.strength * 100)}%)`}
              </td>
              <td style={{ padding: "4px 8px", fontFamily: "var(--font-mono)" }}>{p.kl_divergence.toFixed(3)}</td>
              <td style={{ padding: "4px 8px", color: p.top_token_changed ? "#e0b45a" : "var(--text-primary)" }}>
                {JSON.stringify(p.top_token_text)} ({(p.top_prob * 100).toFixed(0)}%)
              </td>
              <td style={{ padding: "4px 8px", whiteSpace: "pre-wrap", color: "var(--text-primary)" }}>
                {p.generated_text ?? "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
