import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  ExecutionEdge,
  ExecutionNode,
  LiveControls,
  LiveEvent,
  LiveNeuronKind,
  LiveSweepPoint,
  LiveSweepTarget,
} from "../electron.d";
import { buttonStyle } from "../app/styles";
import { SweepChart } from "./SweepChart";

/**
 * Live Slider(Phase3、2026-09-24追加): スライダーによるリアルタイム出力操作。
 *
 * 2026-09-24、漣さんとの確認事項:
 * - 操作対象: すべて(層・ブロックの倍率、個別ニューロン(MLP次元/Attentionヘッド)の倍率、ステアリングの強さ)
 * - 反映方式: 生成中に反映(スライダーを動かすと次のトークンから新しい値が使われる)
 * - 表示: 通常時と操作後の文章・KLの比較に加え、倍率ごとのKLの変化グラフ(スイープ)
 *
 * python/run_live_session.py を常駐させ(electron/main.ts の live:*)、モデルは開始時に1回だけ読み込む。
 * 状態はこのコンポーネントの中で完結させている(App.tsxの肥大化を避けるため)。
 */

interface LiveSliderPanelProps {
  modelDir: string;
  config: Record<string, unknown>;
  nodes: ExecutionNode[];
  edges: ExecutionEdge[];
  useChatTemplate: boolean;
  // Activation Steeringタブで入力した対照プロンプト・対象ノードを共有する
  steeringPositivePrompts: string[];
  steeringNegativePrompts: string[];
  steeringTargetNodeIds: string[];
}

type SessionStatus = "stopped" | "starting" | "ready" | "exited";

interface NeuronRow {
  layer: number;
  kind: LiveNeuronKind;
  index: number;
  scale: number;
}

interface TokenPiece {
  text: string;
  version: number;
  kl: number;
  changed: boolean;
}

const NODE_RANGE = { min: 0, max: 3, step: 0.05 };
const NEURON_RANGE = { min: -2, max: 5, step: 0.1 };
const STEERING_RANGE = { min: -4, max: 4, step: 0.1 };

function neuronKey(n: { layer: number; kind: LiveNeuronKind; index: number }) {
  return `${n.layer}:${n.kind}:${n.index}`;
}

// ここまでの全文(nextText)から、前回までの断片列を更新する。マルチバイト文字が2トークンに分かれると
// 前回の末尾が置換文字(�)で、次のトークンで確定するため、食い違った分は末尾の断片を削って付け直す
export function appendPiece(pieces: TokenPiece[], nextText: string, piece: Omit<TokenPiece, "text">): TokenPiece[] {
  const prevText = pieces.map((p) => p.text).join("");
  if (nextText.startsWith(prevText)) {
    return [...pieces, { ...piece, text: nextText.slice(prevText.length) }];
  }
  let common = 0;
  while (common < prevText.length && common < nextText.length && prevText[common] === nextText[common]) common++;
  const kept: TokenPiece[] = [];
  let len = 0;
  for (const p of pieces) {
    if (len + p.text.length > common) {
      const cut = p.text.slice(0, common - len);
      if (cut) kept.push({ ...p, text: cut });
      break;
    }
    kept.push(p);
    len += p.text.length;
  }
  return [...kept, { ...piece, text: nextText.slice(common) }];
}

export function LiveSliderPanel(props: LiveSliderPanelProps) {
  const { modelDir, config, nodes, edges, useChatTemplate } = props;

  const scalableNodes = useMemo(
    () => nodes.filter((n) => n.role === "attention_block" || n.role === "mlp_block" || n.role === "layer_block"),
    [nodes]
  );
  // グラフ・テンプレート設定が変わったら、セッションの作り直しが必要
  const graphSignature = useMemo(
    () => JSON.stringify({ modelDir, n: nodes.map((n) => [n.id, n.role, n.merge_mode]), e: edges, t: useChatTemplate }),
    [modelDir, nodes, edges, useChatTemplate]
  );

  const [status, setStatus] = useState<SessionStatus>("stopped");
  const [sessionSignature, setSessionSignature] = useState<string | null>(null);
  const [limits, setLimits] = useState<{ num_layers: number; intermediate_size: number; num_heads: number } | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);

  const [nodeScales, setNodeScales] = useState<Record<string, number>>({});
  const [nodeToAdd, setNodeToAdd] = useState("");
  const [neurons, setNeurons] = useState<NeuronRow[]>([]);
  const [neuronForm, setNeuronForm] = useState<{ layer: number; kind: LiveNeuronKind; index: number }>({
    layer: 0,
    kind: "mlp",
    index: 0,
  });
  const [steeringScale, setSteeringScale] = useState(0);
  const [steeringNorms, setSteeringNorms] = useState<Record<string, number> | null>(null);
  const [steeringBusy, setSteeringBusy] = useState(false);
  const [appliedVersion, setAppliedVersion] = useState(0);

  const [inputText, setInputText] = useState("こんにちは");
  const [maxNewTokens, setMaxNewTokens] = useState(60);
  const [compareBaseline, setCompareBaseline] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [baselineText, setBaselineText] = useState<string | null>(null);
  const [pieces, setPieces] = useState<TokenPiece[]>([]);
  const [doneInfo, setDoneInfo] = useState<{ reason: string; mean_kl: number; max_kl: number } | null>(null);
  const [lastKl, setLastKl] = useState<number | null>(null);

  const [sweepTargetKey, setSweepTargetKey] = useState("");
  const [sweepRange, setSweepRange] = useState({ start: 0, stop: 2, steps: 9 });
  const [sweepBusy, setSweepBusy] = useState(false);
  const [sweep, setSweep] = useState<{
    key: string;
    points: LiveSweepPoint[];
    baselineTop: string;
  } | null>(null);
  const sweepKeyRef = useRef("");

  // --- イベント購読 ---
  useEffect(() => {
    const off = window.electronAPI.onLiveEvent((ev: LiveEvent) => {
      switch (ev.event) {
        case "ready":
          setStatus("ready");
          setLimits(ev.limits);
          setLastError(null);
          break;
        case "steering_ready":
          setSteeringNorms(ev.vector_norms);
          setSteeringBusy(false);
          break;
        case "controls_applied":
          setAppliedVersion(ev.version);
          break;
        case "started":
          break;
        case "baseline":
          setBaselineText(ev.text);
          break;
        case "token":
          setLastKl(ev.kl);
          setPieces((prev) =>
            appendPiece(prev, ev.text, { version: ev.controls_version, kl: ev.kl, changed: ev.top_token_changed })
          );
          break;
        case "done":
          setGenerating(false);
          setDoneInfo({ reason: ev.reason, mean_kl: ev.mean_kl, max_kl: ev.max_kl });
          break;
        case "sweep_result":
          setSweepBusy(false);
          setSweep({ key: sweepKeyRef.current, points: ev.points, baselineTop: ev.baseline_top_token_text });
          break;
        case "error":
          setLastError(`${ev.error_type}: ${ev.message}`);
          if (ev.cmd === "generate") setGenerating(false);
          if (ev.cmd === "sweep") setSweepBusy(false);
          if (ev.cmd === "set_steering") setSteeringBusy(false);
          if (ev.cmd === "init") setStatus("exited");
          break;
        case "exited":
          setStatus("exited");
          setGenerating(false);
          setSweepBusy(false);
          setSteeringBusy(false);
          if (ev.code !== 0 && ev.stderr) setLastError(`セッションが終了しました(code ${ev.code}): ${ev.stderr.slice(-600)}`);
          break;
      }
    });
    return off;
  }, []);

  // 画面を離れたらセッションを終了する
  useEffect(() => () => void window.electronAPI.liveStop(), []);

  // --- コントロール(スライダー値)→ Python へ送る(間引き80ms) ---
  const controls: LiveControls = useMemo(
    () => ({
      node_scales: nodeScales,
      neurons: neurons.map((n) => ({ layer: n.layer, kind: n.kind, index: n.index, scale: n.scale })),
      steering_scale: steeringNorms ? steeringScale : 0,
    }),
    [nodeScales, neurons, steeringScale, steeringNorms]
  );
  const sendTimer = useRef<number | null>(null);
  useEffect(() => {
    if (status !== "ready") return;
    if (sendTimer.current !== null) window.clearTimeout(sendTimer.current);
    sendTimer.current = window.setTimeout(() => {
      void window.electronAPI.liveSend({ cmd: "set_controls", controls });
    }, 80);
    return () => {
      if (sendTimer.current !== null) window.clearTimeout(sendTimer.current);
    };
  }, [controls, status]);

  // --- 操作 ---
  const startSession = useCallback(async () => {
    setStatus("starting");
    setLastError(null);
    setSteeringNorms(null);
    setAppliedVersion(0);
    setSessionSignature(graphSignature);
    await window.electronAPI.liveStart({ modelDir, config, nodes, edges, useChatTemplate });
  }, [graphSignature, modelDir, config, nodes, edges, useChatTemplate]);

  const stopSession = useCallback(async () => {
    await window.electronAPI.liveStop();
    setStatus("stopped");
    setGenerating(false);
  }, []);

  const generate = useCallback(() => {
    if (status !== "ready" || !inputText.trim()) return;
    setGenerating(true);
    setBaselineText(null);
    setPieces([]);
    setDoneInfo(null);
    setLastKl(null);
    void window.electronAPI.liveSend({
      cmd: "generate",
      input_text: inputText,
      max_new_tokens: maxNewTokens,
      compare_baseline: compareBaseline,
    });
  }, [status, inputText, maxNewTokens, compareBaseline]);

  const stopGeneration = useCallback(() => void window.electronAPI.liveSend({ cmd: "stop" }), []);

  const computeSteering = useCallback(() => {
    setSteeringBusy(true);
    void window.electronAPI.liveSend({
      cmd: "set_steering",
      positive_prompts: props.steeringPositivePrompts,
      negative_prompts: props.steeringNegativePrompts,
      target_node_ids: props.steeringTargetNodeIds,
    });
  }, [props.steeringPositivePrompts, props.steeringNegativePrompts, props.steeringTargetNodeIds]);

  const resetAll = useCallback(() => {
    setNodeScales((prev) => Object.fromEntries(Object.keys(prev).map((k) => [k, 1])));
    setNeurons((prev) => prev.map((n) => ({ ...n, scale: 1 })));
    setSteeringScale(0);
  }, []);

  // スイープ対象の候補(追加済みのスライダー+ステアリング)
  const sweepTargets = useMemo(() => {
    const list: { key: string; label: string; target: LiveSweepTarget; current: number }[] = [];
    for (const id of Object.keys(nodeScales)) {
      list.push({ key: `node:${id}`, label: `層/ブロック ${id}`, target: { type: "node", id }, current: nodeScales[id] });
    }
    for (const n of neurons) {
      list.push({
        key: `neuron:${neuronKey(n)}`,
        label: `ニューロン L${n.layer} ${n.kind === "mlp" ? "MLP" : "Head"} #${n.index}`,
        target: { type: "neuron", layer: n.layer, kind: n.kind, index: n.index },
        current: n.scale,
      });
    }
    if (steeringNorms) {
      list.push({ key: "steering", label: "ステアリングの強さ", target: { type: "steering" }, current: steeringScale });
    }
    return list;
  }, [nodeScales, neurons, steeringNorms, steeringScale]);

  const runSweep = useCallback(() => {
    const t = sweepTargets.find((s) => s.key === sweepTargetKey);
    if (!t || status !== "ready") return;
    setSweepBusy(true);
    sweepKeyRef.current = t.key;
    void window.electronAPI.liveSend({
      cmd: "sweep",
      input_text: inputText,
      target: t.target,
      start: sweepRange.start,
      stop: sweepRange.stop,
      steps: sweepRange.steps,
    });
  }, [sweepTargets, sweepTargetKey, status, inputText, sweepRange]);

  // ニューロン入力の範囲チェック(範囲外を追加すると、Python側で全コントロールが拒否されてしまうため事前に防ぐ)
  const neuronFormError = useMemo(() => {
    if (!limits) return null;
    const { layer, kind, index } = neuronForm;
    if (!Number.isInteger(layer) || layer < 0 || layer >= limits.num_layers) return `層は 0〜${limits.num_layers - 1}`;
    const upper = kind === "mlp" ? limits.intermediate_size : limits.num_heads;
    if (!Number.isInteger(index) || index < 0 || index >= upper) return `${kind === "mlp" ? "MLP次元" : "ヘッド"}は 0〜${upper - 1}`;
    return null;
  }, [limits, neuronForm]);

  const ready = status === "ready";
  const stale = ready && sessionSignature !== graphSignature;
  const currentSweep = sweepTargets.find((s) => s.key === sweep?.key);

  return (
    <div style={{ width: "100%", height: "100%", overflow: "auto", padding: 20, display: "flex", gap: 20 }}>
      {/* ---------------- 左: コントロール ---------------- */}
      <div style={{ flex: "0 0 330px", display: "flex", flexDirection: "column", gap: 14 }}>
        <Section title="セッション">
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <button onClick={startSession} style={buttonStyle} disabled={status === "starting"}>
              {ready || status === "exited" ? "再開始" : "開始(モデル読み込み)"}
            </button>
            <button onClick={stopSession} style={buttonStyle} disabled={!ready && status !== "starting"}>
              終了
            </button>
            <span style={{ fontSize: 11, color: ready ? "var(--accent-text)" : "var(--text-muted)" }}>
              {status === "stopped" && "未開始"}
              {status === "starting" && "読み込み中..."}
              {ready && `準備完了(反映済み v${appliedVersion})`}
              {status === "exited" && "終了しました"}
            </span>
          </div>
          {stale && (
            <Note>グラフまたはチャットテンプレート設定が変わりました。反映するには「再開始」を押してください。</Note>
          )}
          <Note>チャットテンプレート: {useChatTemplate ? "適用(ヘッダーで切り替え)" : "なし(ヘッダーで切り替え)"}</Note>
        </Section>

        <Section title="層・ブロックの倍率" hint="ブロックが足す分(残差への寄与)に掛ける。1=通常、0=素通り、2=強調">
          <div style={{ display: "flex", gap: 6 }}>
            <select value={nodeToAdd} onChange={(e) => setNodeToAdd(e.target.value)}
              style={{ ...buttonStyle, background: "var(--bg-app)", flex: 1, minWidth: 0 }}>
              <option value="">ノードを選択</option>
              {scalableNodes.filter((n) => !(n.id in nodeScales)).map((n) => (
                <option key={n.id} value={n.id}>{n.id}</option>
              ))}
            </select>
            <button style={buttonStyle} disabled={!nodeToAdd}
              onClick={() => { setNodeScales((p) => ({ ...p, [nodeToAdd]: 1 })); setNodeToAdd(""); }}>
              追加
            </button>
          </div>
          {Object.entries(nodeScales).map(([id, v]) => (
            <SliderRow key={id} label={id} value={v} range={NODE_RANGE}
              onChange={(nv) => setNodeScales((p) => ({ ...p, [id]: nv }))}
              onRemove={() => setNodeScales((p) => Object.fromEntries(Object.entries(p).filter(([k]) => k !== id)))} />
          ))}
        </Section>

        <Section title="ニューロンの倍率" hint="MLP中間層の1次元、またはAttentionの1ヘッド。Neuron Searchで見つけた番号を入れる">
          <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
            <label style={labelStyle}>層
              <input type="number" min={0} max={(limits?.num_layers ?? 1) - 1} value={neuronForm.layer}
                onChange={(e) => setNeuronForm((f) => ({ ...f, layer: Number(e.target.value) }))}
                style={{ ...numberStyle, width: 52 }} />
            </label>
            <select value={neuronForm.kind}
              onChange={(e) => setNeuronForm((f) => ({ ...f, kind: e.target.value as LiveNeuronKind }))}
              style={{ ...buttonStyle, background: "var(--bg-app)" }}>
              <option value="mlp">MLP次元</option>
              <option value="attention_head">ヘッド</option>
            </select>
            <label style={labelStyle}>番号
              <input type="number" min={0} value={neuronForm.index}
                onChange={(e) => setNeuronForm((f) => ({ ...f, index: Number(e.target.value) }))}
                style={{ ...numberStyle, width: 64 }} />
            </label>
            <button style={buttonStyle} disabled={!limits || neuronFormError !== null}
              title={!limits ? "セッションを開始すると追加できます" : neuronFormError ?? ""}
              onClick={() => setNeurons((p) =>
                p.some((n) => neuronKey(n) === neuronKey(neuronForm)) ? p : [...p, { ...neuronForm, scale: 1 }])}>
              追加
            </button>
          </div>
          {limits && (
            <Note>
              層 0〜{limits.num_layers - 1} / MLP次元 0〜{limits.intermediate_size - 1} / ヘッド 0〜{limits.num_heads - 1}
              {neuronFormError && <span style={{ color: "#e08a8a" }}>(範囲外: {neuronFormError})</span>}
            </Note>
          )}
          {neurons.map((n) => (
            <SliderRow key={neuronKey(n)}
              label={`L${n.layer} ${n.kind === "mlp" ? "MLP" : "Head"} #${n.index}`}
              value={n.scale} range={NEURON_RANGE}
              onChange={(nv) => setNeurons((p) => p.map((x) => (neuronKey(x) === neuronKey(n) ? { ...x, scale: nv } : x)))}
              onRemove={() => setNeurons((p) => p.filter((x) => neuronKey(x) !== neuronKey(n)))} />
          ))}
        </Section>

        <Section title="ステアリングの強さ"
          hint="Activation Steeringタブの対照プロンプト・対象ノードでベクトルを作り、強さだけをここで動かす">
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <button style={buttonStyle} onClick={computeSteering}
              disabled={!ready || steeringBusy || props.steeringTargetNodeIds.length === 0 ||
                props.steeringPositivePrompts.length === 0 || props.steeringNegativePrompts.length === 0}>
              {steeringBusy ? "計算中..." : steeringNorms ? "ベクトルを再計算" : "ベクトルを計算"}
            </button>
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              対象 {props.steeringTargetNodeIds.length}ノード / positive {props.steeringPositivePrompts.length}件 /
              negative {props.steeringNegativePrompts.length}件
            </span>
          </div>
          {steeringNorms ? (
            <SliderRow label="強さ" value={steeringScale} range={STEERING_RANGE} onChange={setSteeringScale} />
          ) : (
            <Note>先にベクトルを計算してください(対象ノードはActivation Steeringタブで選択)。</Note>
          )}
        </Section>

        <button style={buttonStyle} onClick={resetAll}>すべて通常に戻す</button>

        {lastError && (
          <div style={{ color: "#e08a8a", fontSize: 11.5, whiteSpace: "pre-wrap" }}>{lastError}</div>
        )}
      </div>

      {/* ---------------- 右: 生成とスイープ ---------------- */}
      <div style={{ flex: "1 1 auto", minWidth: 320, display: "flex", flexDirection: "column", gap: 18 }}>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <input type="text" value={inputText} onChange={(e) => setInputText(e.target.value)} placeholder="入力テキスト"
            style={{ ...buttonStyle, cursor: "text", width: 260, background: "var(--bg-app)" }} />
          <label style={labelStyle}>トークン数
            <input type="number" min={1} max={512} value={maxNewTokens}
              onChange={(e) => setMaxNewTokens(Math.max(1, Number(e.target.value) || 1))}
              style={{ ...numberStyle, width: 60 }} />
          </label>
          <label style={labelStyle}>
            <input type="checkbox" checked={compareBaseline} onChange={(e) => setCompareBaseline(e.target.checked)} />
            通常時も生成して比較
          </label>
          {!generating ? (
            <button onClick={generate} disabled={!ready}
              style={{ ...buttonStyle, background: "var(--accent)", color: "#0f1418", fontWeight: 600 }}>
              生成
            </button>
          ) : (
            <button onClick={stopGeneration} style={buttonStyle}>停止</button>
          )}
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            生成中にスライダーを動かすと、次のトークンから反映されます
          </span>
        </div>

        <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
          <Stat label="直近トークンのKL(通常時→操作後)" value={lastKl === null ? "—" : lastKl.toFixed(4)} />
          <Stat label="平均KL / 最大KL"
            value={doneInfo ? `${doneInfo.mean_kl.toFixed(4)} / ${doneInfo.max_kl.toFixed(4)}` : "—"} />
          <Stat label="終了理由" value={doneInfo ? REASON_LABEL[doneInfo.reason] ?? doneInfo.reason : generating ? "生成中" : "—"} />
        </div>

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {compareBaseline && (
            <TextPane title="通常時">
              {baselineText === null ? (generating ? "生成中..." : "") : baselineText || "(空)"}
            </TextPane>
          )}
          <TextPane title="操作後(背景色の切り替わり=スライダーの値が変わった位置、点線の下線=最有力トークンが通常時と違う)" accent>
            {pieces.length === 0
              ? generating ? "生成中..." : ""
              : segmentIndexes(pieces).map((seg, i) => {
                  const p = pieces[i];
                  return (
                    <span key={i} title={`KL ${p.kl.toFixed(4)} / スライダー設定 v${p.version}`}
                      style={{
                        background: seg % 2 === 1 ? "rgba(91,155,213,0.32)" : "transparent",
                        textDecoration: p.changed ? "underline dotted" : "none",
                      }}>
                      {p.text}
                    </span>
                  );
                })}
          </TextPane>
        </div>

        <Section title="倍率スイープ(倍率ごとのKLの変化)"
          hint="選んだスライダー1つだけを刻みながら、入力の次トークン予測が通常時からどれだけ変わるかを測る(他のスライダーは今の値のまま)">
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <select value={sweepTargetKey} onChange={(e) => setSweepTargetKey(e.target.value)}
              style={{ ...buttonStyle, background: "var(--bg-app)", maxWidth: 280 }}>
              <option value="">対象を選択(先にスライダーを追加)</option>
              {sweepTargets.map((t) => <option key={t.key} value={t.key}>{t.label}</option>)}
            </select>
            <label style={labelStyle}>開始
              <input type="number" step={0.5} value={sweepRange.start}
                onChange={(e) => setSweepRange((r) => ({ ...r, start: Number(e.target.value) }))}
                style={{ ...numberStyle, width: 56 }} />
            </label>
            <label style={labelStyle}>終了
              <input type="number" step={0.5} value={sweepRange.stop}
                onChange={(e) => setSweepRange((r) => ({ ...r, stop: Number(e.target.value) }))}
                style={{ ...numberStyle, width: 56 }} />
            </label>
            <label style={labelStyle}>点数
              <input type="number" min={2} max={41} value={sweepRange.steps}
                onChange={(e) => setSweepRange((r) => ({ ...r, steps: Number(e.target.value) }))}
                style={{ ...numberStyle, width: 50 }} />
            </label>
            <button style={buttonStyle} onClick={runSweep} disabled={!ready || !sweepTargetKey || sweepBusy || generating}>
              {sweepBusy ? "計測中..." : "スイープ"}
            </button>
          </div>
          {sweep && (
            <>
              <div style={{ fontSize: 11.5, color: "var(--text-secondary)" }}>
                {sweepTargets.find((t) => t.key === sweep.key)?.label ?? sweep.key} の倍率ごとのKL(入力「{inputText}」の次トークン、
                通常時の最有力: {JSON.stringify(sweep.baselineTop)})
              </div>
              <SweepChart points={sweep.points} currentScale={currentSweep ? currentSweep.current : null} xLabel="倍率" />
              <details>
                <summary style={{ fontSize: 11, color: "var(--text-secondary)", cursor: "pointer" }}>表で見る</summary>
                <table style={{ fontSize: 11, borderCollapse: "collapse", marginTop: 6 }}>
                  <thead>
                    <tr>{["倍率", "KL", "最有力トークン", "その確率", "通常時の最有力の確率"].map((h) => (
                      <th key={h} style={thStyle}>{h}</th>))}</tr>
                  </thead>
                  <tbody>
                    {sweep.points.map((p) => (
                      <tr key={p.scale}>
                        <td style={tdStyle}>{p.scale}</td>
                        <td style={tdStyle}>{p.kl.toFixed(4)}</td>
                        <td style={tdStyle}>{JSON.stringify(p.top_token_text)}</td>
                        <td style={tdStyle}>{(p.top_prob * 100).toFixed(1)}%</td>
                        <td style={tdStyle}>{(p.baseline_top_prob * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>
            </>
          )}
        </Section>
      </div>
    </div>
  );
}

// スライダーの値(版)が変わるたびに区切りの番号を1つ進める(背景色を交互にするため)
function segmentIndexes(pieces: TokenPiece[]): number[] {
  let seg = 0;
  return pieces.map((p, i) => {
    if (i > 0 && p.version !== pieces[i - 1].version) seg += 1;
    return seg;
  });
}

const REASON_LABEL: Record<string, string> = {
  max_new_tokens: "トークン数の上限",
  end_of_turn: "発話終了",
  stopped: "停止ボタン",
};

function Section({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: "var(--bg-panel)", border: "1px solid var(--border-default)",
      borderRadius: "var(--radius-md)", padding: 12, display: "flex", flexDirection: "column", gap: 8,
    }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)" }}>{title}</div>
      {hint && <div style={{ fontSize: 10.5, color: "var(--text-muted)" }}>{hint}</div>}
      {children}
    </div>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return <div style={{ fontSize: 10.5, color: "var(--text-muted)" }}>{children}</div>;
}

function SliderRow({ label, value, range, onChange, onRemove }: {
  label: string;
  value: number;
  range: { min: number; max: number; step: number };
  onChange: (v: number) => void;
  onRemove?: () => void;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-secondary)", flex: 1,
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={label}>{label}</span>
        <input type="number" value={value} step={range.step} min={-10} max={10}
          onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) onChange(Math.max(-10, Math.min(10, v))); }}
          style={{ ...numberStyle, width: 64 }} />
        {onRemove && (
          <button onClick={onRemove} title="外す" style={{ ...buttonStyle, padding: "2px 8px" }}>×</button>
        )}
      </div>
      <input type="range" min={range.min} max={range.max} step={range.step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} style={{ width: "100%" }} />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ background: "var(--bg-panel-raised)", borderRadius: "var(--radius-md)", padding: "8px 12px", minWidth: 170 }}>
      <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 16, fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>{value}</div>
    </div>
  );
}

function TextPane({ title, accent, children }: { title: string; accent?: boolean; children: React.ReactNode }) {
  return (
    <div style={{
      flex: "1 1 280px", background: "var(--bg-panel-raised)", borderRadius: "var(--radius-sm)", padding: "8px 10px",
      border: accent ? "1px solid var(--accent)" : "1px solid transparent", minHeight: 80,
    }}>
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 13, color: "var(--text-primary)", whiteSpace: "pre-wrap", lineHeight: 1.7 }}>{children}</div>
    </div>
  );
}

const labelStyle = { display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--text-secondary)" } as const;
const numberStyle = { ...buttonStyle, cursor: "text", background: "var(--bg-app)", padding: "4px 6px" } as const;
const thStyle = { textAlign: "left", padding: "3px 10px 3px 0", color: "var(--text-muted)", fontWeight: 500 } as const;
const tdStyle = { padding: "2px 10px 2px 0", fontFamily: "var(--font-mono)", color: "var(--text-primary)" } as const;
