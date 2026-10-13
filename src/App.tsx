import { useCallback, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  SelectionMode,
  type Node,
  type Edge,
  type NodeTypes,
  type Connection,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { toFlowGraph } from "./lib/toFlowGraph";
import { toExecutionGraph, flowEdgesToExecutionEdges } from "./lib/toExecutionGraph";
import { Legend } from "./components/Legend";
import { NormalNode, type NormalNodeData } from "./components/NormalNode";
import { LayerSummaryNode, type LayerSummaryNodeData } from "./components/LayerSummaryNode";
import { AttentionHeatmap } from "./components/AttentionHeatmap";
import { ActivationStatsList } from "./components/ActivationStatsList";
import { SensitivityHeatmap } from "./components/SensitivityHeatmap";
import { TokenJourneyView } from "./components/TokenJourneyView";
import { BattleView } from "./components/BattleView";
import { NeuronSearchView } from "./components/NeuronSearchView";
import { LayerAblationView } from "./components/LayerAblationView";
import { ActivationSteeringView } from "./components/ActivationSteeringView";
import { SteeringToolbar } from "./components/SteeringToolbar";
import { LiveSliderPanel } from "./components/LiveSliderPanel";
import type {
  ModelInspectResult,
  ExecutionNode,
  ReplacementMode,
  NeuronKind,
  NeuronAggregation,
  InterventionMode,
} from "./electron.d";

import {
  SYCOPHANCY_POSITIVE_PROMPTS,
  SYCOPHANCY_NEGATIVE_PROMPTS,
  REFUSAL_POSITIVE_PROMPTS,
  REFUSAL_NEGATIVE_PROMPTS,
} from "./app/presets";
import type {
  LoadState,
  Tab,
  RunState,
  InternalsRunState,
  SensitivityRunState,
  TokenJourneyRunState,
  BattleRunState,
  NeuronSearchRunState,
  LayerAblationRunState,
  ActivationSteeringRunState,
  BattleModelEntry,
} from "./app/types";
import { formatParamCount, shortenBlockId } from "./app/format";
import { buttonStyle, tabButtonStyle, tabButtonActiveStyle, infoPanelStyle, emptyStateStyle } from "./app/styles";

// 2026-09-22: App.tsx分割の第1段階。プリセット(src/app/presets.ts、実体はpython/presets/
// steering_presets.json)・状態型(src/app/types.ts)・整形関数(src/app/format.ts)・
// スタイル(src/app/styles.ts)を別ファイルへ切り出した。

const nodeTypes: NodeTypes = {
  normal: NormalNode,
  layerSummary: LayerSummaryNode,
};

export default function App() {
  const [tab, setTab] = useState<Tab>("dissect");
  const [rawResult, setRawResult] = useState<ModelInspectResult | null>(null);
  const [loadState, setLoadState] = useState<LoadState>({ status: "idle" });

  // --- 解剖タブ(Phase0)の状態 ---
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [selectedCount, setSelectedCount] = useState(0);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());

  // --- 実行タブ(Phase1)の状態 ---
  const [execNodes, setExecNodes, onExecNodesChange] = useNodesState<Node>([]);
  const [execEdges, setExecEdges, onExecEdgesChange] = useEdgesState<Edge>([]);
  const [executionNodeDefs, setExecutionNodeDefs] = useState<ExecutionNode[]>([]);
  const [runState, setRunState] = useState<RunState>({ status: "idle" });
  // 実行タブでAttention/MLPに展開済みのLayer番号。展開操作をすると、そのグラフは
  // 再生成される(繋ぎ替えていたエッジはリセットされる、という割り切り)。
  const [expandedExecLayers, setExpandedExecLayers] = useState<Set<number>>(new Set());
  const [inputText, setInputText] = useState("こんにちは");
  // 2026-09-22追加(全タブ共通、漣さんとの確認事項): 入力文をモデルのチャットテンプレート
  // (tokenizer_config.jsonのchat_template)で包んでから実行するかどうか。Instructモデルは
  // テンプレート無しだと「応答」ではなく「文章の続き書き」モードで動くため(Sycophancy Reducer
  // 実機検証で判明)。ヘッダーのチェックボックスで切り替え、実行・観察の全機能に共通で渡す。
  const [useChatTemplate, setUseChatTemplate] = useState(false);
  const [maxNewTokens, setMaxNewTokens] = useState(20);
  const [temperature, setTemperature] = useState(0); // 0 = 貪欲デコード
  const [topK, setTopK] = useState<number | "">("");
  const [topP, setTopP] = useState<number | "">("");
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.0);
  const [noRepeatNgramSize, setNoRepeatNgramSize] = useState<number | "">("");
  const [showSamplingOptions, setShowSamplingOptions] = useState(false);

  // --- 観察タブ(Phase2)の状態 ---
  const [observeSubTab, setObserveSubTab] = useState<
    "internals" | "sensitivity" | "journey" | "battle" | "neuronSearch" | "ablation" | "steering" | "live"
  >("internals");
  const [observeInputText, setObserveInputText] = useState("こんにちは");
  const [aggregateHeads, setAggregateHeads] = useState(true);
  const [internalsRunState, setInternalsRunState] = useState<InternalsRunState>({ status: "idle" });
  const [selectedAttentionNodeId, setSelectedAttentionNodeId] = useState<string | null>(null);
  const [activationScale, setActivationScale] = useState<"linear" | "log">("linear");

  // --- Prompt Sensitivity Map(Phase2)の状態 ---
  const [sensitivityInputText, setSensitivityInputText] = useState("こんにちは、元気ですか？");
  const [replacementMode, setReplacementMode] = useState<ReplacementMode>("pad");
  const [customTokenId, setCustomTokenId] = useState<number | "">("");
  const [sensitivityRunState, setSensitivityRunState] = useState<SensitivityRunState>({
    status: "idle",
  });

  // --- Token Journey(Phase2)の状態 ---
  const [journeyInputText, setJourneyInputText] = useState("こんにちは");
  const [journeyRunState, setJourneyRunState] = useState<TokenJourneyRunState>({ status: "idle" });

  // --- Multi-Model Battle View(Phase2)の状態 ---
  // 現在「解剖」タブで開いているモデルとは独立に、比較対象モデルを個別に選ぶ
  // (Battle Viewでは異なるアーキテクチャのモデル同士も比較対象になりうるため)。
  const [battleModels, setBattleModels] = useState<BattleModelEntry[]>([]);
  const [battleAddingModel, setBattleAddingModel] = useState(false);
  const [battleInputText, setBattleInputText] = useState("こんにちは");
  const [battleCompareGeneration, setBattleCompareGeneration] = useState(true);
  const [battleCompareInternals, setBattleCompareInternals] = useState(true);
  const [battleMaxNewTokens, setBattleMaxNewTokens] = useState(20);
  const [battleRunState, setBattleRunState] = useState<BattleRunState>({ status: "idle" });

  // --- Neuron Search(Phase2)の状態 ---
  // 2026-09-12確認事項: rank(入力に対するランキング)・probe(特定ニューロンの
  // 複数入力への反応比較)の両方に対応する。既存の観察タブと同様、実行タブで
  // 組んだ現在のグラフ構造(execEdges/executionNodeDefs)をそのまま使う。
  const [neuronSearchMode, setNeuronSearchMode] = useState<"rank" | "probe">("rank");
  const [neuronSearchInputText, setNeuronSearchInputText] = useState("こんにちは");
  const [neuronKind, setNeuronKind] = useState<NeuronKind>("both");
  const [neuronTargetLayersText, setNeuronTargetLayersText] = useState(""); // 空=全層
  const [neuronTopK, setNeuronTopK] = useState(20);
  const [neuronAggregation, setNeuronAggregation] = useState<NeuronAggregation>("max_abs");
  const [neuronSearchRunState, setNeuronSearchRunState] = useState<NeuronSearchRunState>({
    status: "idle",
  });
  // probeモード用の入力(改行区切りで複数文を指定する)
  const [probeLayerIndex, setProbeLayerIndex] = useState(0);
  const [probeKind, setProbeKind] = useState<"mlp" | "attention_head">("mlp");
  const [probeNeuronIndex, setProbeNeuronIndex] = useState(0);
  const [probeInputTextsRaw, setProbeInputTextsRaw] = useState("こんにちは\nさようなら");

  // --- Layer Ablation Wizard(Phase2)の状態 ---
  // 2026-09-12確認事項: 「無効化」=出力をゼロに(残差も断つ)。全層自動バッチ処理
  // ではなく、ユーザーがGUI上で選んだノード(1つ以上)を無効化した状態と通常時とを
  // 1回の実行で比較する対話的ツール(具体的なUXは「おすすめで」と一任された部分)。
  const [ablationInputText, setAblationInputText] = useState("こんにちは");
  const [ablationSelectedNodeIds, setAblationSelectedNodeIds] = useState<Set<string>>(new Set());
  const [ablationCompareGeneration, setAblationCompareGeneration] = useState(true);
  const [ablationMaxNewTokens, setAblationMaxNewTokens] = useState(20);
  const [ablationRunState, setAblationRunState] = useState<LayerAblationRunState>({ status: "idle" });

  // --- Activation Steering Studio(Phase3、2026-09-15着手)の状態 ---
  // 2026-09-15確認事項: ステアリングベクトルは対照プロンプト群(positive/negative)の
  // 平均活性化値の差分方式("Contrastive Activation Addition"系)。適用対象は
  // ユーザーが選んだノード(Layer Ablation Wizardと同じチェックボックス選択、
  // attention_block/mlp_block/layer_blockに限る。ablatableNodesを再利用する)。
  // 効果の確認はKLダイバージェンス(定量指標)+生成テキストの変化(目視)の両方。
  const [steeringPositivePromptsRaw, setSteeringPositivePromptsRaw] = useState(
    "ありがとう、助かりました\nとても嬉しいです"
  );
  const [steeringNegativePromptsRaw, setSteeringNegativePromptsRaw] = useState(
    "最悪だ\nもう嫌になった"
  );
  const [steeringTargetNodeIds, setSteeringTargetNodeIds] = useState<Set<string>>(new Set());
  const [steeringScale, setSteeringScale] = useState(4.0);
  const [steeringInputText, setSteeringInputText] = useState("こんにちは");
  const [steeringCompareGeneration, setSteeringCompareGeneration] = useState(true);
  const [steeringMaxNewTokens, setSteeringMaxNewTokens] = useState(20);
  const [steeringRunState, setSteeringRunState] = useState<ActivationSteeringRunState>({
    status: "idle",
  });
  // Sycophancy Reducer(2026-09-15追加): 迎合的定型語彙の出現回数(機械的指標)も
  // 併せて計測するかどうか。「Sycophancy Reducerプリセット」ボタンでtrueにする
  // (Activation Steering Studio本体の挙動には影響しないオプトイン)。
  const [steeringMeasureSycophancy, setSteeringMeasureSycophancy] = useState(false);
  // Refusal Behavior Tuner(2026-09-22追加): 拒否定型句の出現回数も計測するか、
  // 介入方式("add"=scale倍して加算、"ablate_direction"=その方向の成分を射影で除去)
  const [steeringMeasureRefusal, setSteeringMeasureRefusal] = useState(false);
  const [steeringInterventionMode, setSteeringInterventionMode] = useState<InterventionMode>("add");

  const applyGraph = useCallback(
    (result: ModelInspectResult, expanded: Set<string>) => {
      const { nodes: flowNodes, edges: flowEdges } = toFlowGraph(result, expanded);
      setNodes(flowNodes);
      setEdges(flowEdges);
    },
    [setNodes, setEdges]
  );

  const applyExecutionGraph = useCallback(
    (result: ModelInspectResult, expandedLayers: Set<number>) => {
      const { nodes: eNodes, edges: eEdges, executionNodes } = toExecutionGraph(result, expandedLayers);
      setExecNodes(eNodes);
      setExecEdges(eEdges);
      setExecutionNodeDefs(executionNodes);
      setRunState({ status: "idle" });
    },
    [setExecNodes, setExecEdges]
  );

  const handleSelectModel = useCallback(async () => {
    const dir = await window.electronAPI.selectModelDir();
    if (!dir) return;

    setLoadState({ status: "loading", modelDir: dir });
    try {
      const result = await window.electronAPI.inspectModel(dir);
      setRawResult(result);
      const initialExpanded = new Set<string>();
      setExpandedGroups(initialExpanded);
      applyGraph(result, initialExpanded);

      const initialExecExpanded = new Set<number>();
      setExpandedExecLayers(initialExecExpanded);
      applyExecutionGraph(result, initialExecExpanded);

      setLoadState({ status: "loaded", modelDir: dir, summary: result });
    } catch (err) {
      setLoadState({
        status: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, [applyGraph, applyExecutionGraph]);

  const toggleGroup = useCallback(
    (groupKey: string) => {
      if (!rawResult) return;
      setExpandedGroups((prev) => {
        const next = new Set(prev);
        if (next.has(groupKey)) {
          // 折りたたむ時は、そのグループより下位の展開状態も一緒に畳む
          // (例: "layer-3"を畳んだら"layer-3-attention"の展開状態も無意味になるため削除)
          next.delete(groupKey);
          for (const key of Array.from(next)) {
            if (key.startsWith(`${groupKey}-`)) next.delete(key);
          }
        } else {
          next.add(groupKey);
        }
        applyGraph(rawResult, next);
        return next;
      });
    },
    [rawResult, applyGraph]
  );

  const handleNodeClick = useCallback(
    (_: unknown, node: Node) => {
      setSelectedNode(node);
      const data = node.data as unknown as LayerSummaryNodeData | NormalNodeData;
      if (data.kind === "layerSummary") {
        toggleGroup(data.groupKey);
      }
    },
    [toggleGroup]
  );

  const handleSelectionChange = useCallback(
    ({ nodes: selected }: { nodes: Node[]; edges: Edge[] }) => {
      setSelectedCount(selected.length);
    },
    []
  );

  // 実行タブ: ユーザーがノード同士をドラッグして繋いだときのハンドラ。
  // これがPhase1のコア操作(自由な繋ぎ替え)。
  const handleExecConnect = useCallback(
    (connection: Connection) => {
      setExecEdges((eds) => addEdge({ ...connection, style: { stroke: "var(--accent)" } }, eds));
      setRunState({ status: "idle" }); // グラフが変わったら前回の実行結果は無効化
    },
    [setExecEdges]
  );

  // 2026-10-13修正: ノードを削除(選択してBackspace)しても実行用のノード定義に残り、
  // 「どこにも接続されていないノード」として実行エラーになっていた。削除をノード定義にも反映する
  const handleExecNodesChange = useCallback(
    (changes: Parameters<typeof onExecNodesChange>[0]) => {
      onExecNodesChange(changes);
      const removed = new Set(changes.filter((c) => c.type === "remove").map((c) => c.id));
      if (removed.size > 0) {
        setExecutionNodeDefs((prev) => prev.filter((n) => !removed.has(n.id)));
        setRunState({ status: "idle" });
      }
    },
    [onExecNodesChange]
  );

  const handleExecEdgesChange = useCallback(
    (changes: Parameters<typeof onExecEdgesChange>[0]) => {
      onExecEdgesChange(changes);
      setRunState({ status: "idle" });
    },
    [onExecEdgesChange]
  );

  // 実行タブ: layer_blockノード(折りたたまれたLayer)をクリックしたら、
  // そのLayerだけAttention/MLPの2ノードに展開する。
  // 注意: 展開するとグラフを再生成するため、それまでの繋ぎ替えはリセットされる。
  const toggleExecLayer = useCallback(
    (layerIndex: number) => {
      if (!rawResult) return;
      setExpandedExecLayers((prev) => {
        const next = new Set(prev);
        if (next.has(layerIndex)) {
          next.delete(layerIndex);
        } else {
          next.add(layerIndex);
        }
        applyExecutionGraph(rawResult, next);
        return next;
      });
    },
    [rawResult, applyExecutionGraph]
  );

  const handleExecNodeClick = useCallback(
    (_: unknown, node: Node) => {
      const data = node.data as { role?: string; layerIndex?: number | null };
      if (data.role === "layer_block" && typeof data.layerIndex === "number") {
        toggleExecLayer(data.layerIndex);
        return;
      }

      // 合流ノード(入力エッジが2本以上)なら、クリックでmerge_mode(add/average)を
      // トグルする。合流でないノードをクリックしても何も起きない。
      const incomingCount = execEdges.filter((e) => e.target === node.id).length;
      if (incomingCount < 2) return;

      setExecutionNodeDefs((prev) =>
        prev.map((n) =>
          n.id === node.id
            ? { ...n, merge_mode: n.merge_mode === "average" ? "add" : "average" }
            : n
        )
      );
      setRunState({ status: "idle" }); // 設定が変わったら前回の実行結果は無効化
    },
    [toggleExecLayer, execEdges]
  );

  // 実行タブのノード表示に、合流状態(入力エッジ数)とmerge_modeを反映する。
  // execEdgesが変わる(繋ぎ替え・合流の追加削除)たびに再計算される。
  const displayExecNodes = useMemo(() => {
    const incomingCountByTarget = new Map<string, number>();
    for (const e of execEdges) {
      incomingCountByTarget.set(e.target, (incomingCountByTarget.get(e.target) ?? 0) + 1);
    }
    const mergeModeById = new Map(executionNodeDefs.map((n) => [n.id, n.merge_mode ?? "add"]));

    return execNodes.map((n) => {
      const incoming = incomingCountByTarget.get(n.id) ?? 0;
      if (incoming < 2) return n;
      const mode = mergeModeById.get(n.id) ?? "add";
      const baseLabel = typeof n.data.label === "string" ? n.data.label : "";
      // 既に付与済みの合流バッジを剥がしてから付け直す(重複防止)
      const cleanLabel = baseLabel.replace(/\s*\[合流:(add|average)\]$/, "");
      return {
        ...n,
        data: {
          ...n.data,
          label: `${cleanLabel} [合流:${mode}]`,
        },
        style: {
          ...(n.style as Record<string, unknown>),
          border: "2px solid var(--accent)",
        },
      };
    });
  }, [execNodes, execEdges, executionNodeDefs]);

  const handleRunExecution = useCallback(async () => {
    if (!rawResult || loadState.status !== "loaded") return;
    setRunState({ status: "running" });
    try {
      const executionEdges = flowEdgesToExecutionEdges(execEdges);
      const result = await window.electronAPI.executeModel({
        useChatTemplate,
        modelDir: loadState.modelDir,
        config: rawResult.raw_config,
        nodes: executionNodeDefs,
        edges: executionEdges,
        inputText: inputText.trim() || undefined,
        maxNewTokens,
        temperature,
        topK: topK === "" ? undefined : topK,
        topP: topP === "" ? undefined : topP,
        repetitionPenalty,
        noRepeatNgramSize: noRepeatNgramSize === "" ? undefined : noRepeatNgramSize,
      });
      setRunState({ status: "done", result });
    } catch (err) {
      setRunState({
        status: "done",
        result: {
          success: false,
          error_type: "UnexpectedError",
          message: err instanceof Error ? err.message : String(err),
        },
      });
    }
  }, [
    useChatTemplate,
    rawResult,
    loadState,
    execEdges,
    executionNodeDefs,
    inputText,
    maxNewTokens,
    temperature,
    topK,
    topP,
    repetitionPenalty,
    noRepeatNgramSize,
  ]);

  // 観察タブ(Phase2): 実行タブと同じグラフ構造(execNodes/execEdges)を使い回して、
  // 内部状態(Attention重み・活性化値)を取得する。グラフの繋ぎ替えは実行タブで行う
  // 前提とし、観察タブでは「今のグラフ構造でどう振る舞うか」を覗く役割に絞る。
  const handleRunInternals = useCallback(async () => {
    if (!rawResult || loadState.status !== "loaded") return;
    setInternalsRunState({ status: "running" });
    try {
      const executionEdges = flowEdgesToExecutionEdges(execEdges);
      const result = await window.electronAPI.inspectInternals({
        useChatTemplate,
        modelDir: loadState.modelDir,
        config: rawResult.raw_config,
        nodes: executionNodeDefs,
        edges: executionEdges,
        inputText: observeInputText.trim() || undefined,
        aggregateHeads,
      });
      setInternalsRunState({ status: "done", result });
      if (result.success) {
        const firstAttnKey = Object.keys(result.attention_weights)[0] ?? null;
        setSelectedAttentionNodeId(firstAttnKey);
      }
    } catch (err) {
      setInternalsRunState({
        status: "done",
        result: {
          success: false,
          error_type: "UnexpectedError",
          message: err instanceof Error ? err.message : String(err),
        },
      });
    }
  }, [useChatTemplate, rawResult, loadState, execEdges, executionNodeDefs, observeInputText, aggregateHeads]);

  const handleRunSensitivity = useCallback(async () => {
    if (!rawResult || loadState.status !== "loaded") return;
    setSensitivityRunState({ status: "running" });
    try {
      const executionEdges = flowEdgesToExecutionEdges(execEdges);
      const result = await window.electronAPI.analyzeSensitivity({
        useChatTemplate,
        modelDir: loadState.modelDir,
        config: rawResult.raw_config,
        nodes: executionNodeDefs,
        edges: executionEdges,
        inputText: sensitivityInputText.trim() || undefined,
        replacementMode,
        customTokenId: customTokenId === "" ? undefined : customTokenId,
      });
      setSensitivityRunState({ status: "done", result });
    } catch (err) {
      setSensitivityRunState({
        status: "done",
        result: {
          success: false,
          error_type: "UnexpectedError",
          message: err instanceof Error ? err.message : String(err),
        },
      });
    }
  }, [
    useChatTemplate,
    rawResult,
    loadState,
    execEdges,
    executionNodeDefs,
    sensitivityInputText,
    replacementMode,
    customTokenId,
  ]);

  const handleRunTokenJourney = useCallback(async () => {
    if (!rawResult || loadState.status !== "loaded") return;
    setJourneyRunState({ status: "running" });
    try {
      const executionEdges = flowEdgesToExecutionEdges(execEdges);
      const result = await window.electronAPI.computeTokenJourney({
        useChatTemplate,
        modelDir: loadState.modelDir,
        config: rawResult.raw_config,
        nodes: executionNodeDefs,
        edges: executionEdges,
        inputText: journeyInputText.trim() || undefined,
        topK: 1,
      });
      setJourneyRunState({ status: "done", result });
    } catch (err) {
      setJourneyRunState({
        status: "done",
        result: {
          success: false,
          error_type: "UnexpectedError",
          message: err instanceof Error ? err.message : String(err),
        },
      });
    }
  }, [useChatTemplate, rawResult, loadState, execEdges, executionNodeDefs, journeyInputText]);

  // Multi-Model Battle View(Phase2): 比較対象モデルをモデル選択ダイアログで追加する。
  // 現在「解剖」タブで開いているモデルとは別に、inspectModel()を使って
  // そのモデル自身のconfig(raw_config)を取得しておく(実行時にそのまま使う)。
  const handleAddBattleModel = useCallback(async () => {
    const dir = await window.electronAPI.selectModelDir();
    if (!dir) return;
    if (battleModels.some((m) => m.modelDir === dir)) return; // 同じモデルの重複追加を防ぐ

    setBattleAddingModel(true);
    try {
      const result = await window.electronAPI.inspectModel(dir);
      setBattleModels((prev) => [
        ...prev,
        { modelDir: dir, label: `${result.model_type} (${dir.split(/[\\/]/).pop()})`, config: result.raw_config },
      ]);
    } catch (err) {
      alert(`モデルの読み込みに失敗しました: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBattleAddingModel(false);
    }
  }, [battleModels]);

  const handleRemoveBattleModel = useCallback((modelDir: string) => {
    setBattleModels((prev) => prev.filter((m) => m.modelDir !== modelDir));
  }, []);

  const handleRunBattle = useCallback(async () => {
    if (battleModels.length === 0) return;
    setBattleRunState({ status: "running" });
    try {
      const result = await window.electronAPI.battleCompare({
        useChatTemplate,
        models: battleModels.map((m) => ({ modelDir: m.modelDir, config: m.config, label: m.label })),
        inputText: battleInputText.trim() || undefined,
        compareGeneration: battleCompareGeneration,
        compareInternals: battleCompareInternals,
        maxNewTokens: battleMaxNewTokens,
      });
      setBattleRunState({ status: "done", result });
    } catch (err) {
      setBattleRunState({
        status: "done",
        result: {
          success: false,
          error_type: "UnexpectedError",
          message: err instanceof Error ? err.message : String(err),
        },
      });
    }
  }, [useChatTemplate, battleModels, battleInputText, battleCompareGeneration, battleCompareInternals, battleMaxNewTokens]);

  // Neuron Search(Phase2): 「実行」タブと同じグラフ構造(execEdges/executionNodeDefs)を
  // 使い回す(既存の観察タブ系の機能と同じ考え方)。
  const parsedTargetLayers = useMemo(() => {
    const trimmed = neuronTargetLayersText.trim();
    if (!trimmed) return undefined;
    const parts = trimmed
      .split(/[,\s]+/)
      .filter((s) => s.length > 0)
      .map((s) => Number(s));
    if (parts.some((n) => Number.isNaN(n))) return undefined;
    return parts;
  }, [neuronTargetLayersText]);

  const handleRunNeuronRank = useCallback(async () => {
    if (!rawResult || loadState.status !== "loaded") return;
    setNeuronSearchRunState({ status: "running" });
    try {
      const executionEdges = flowEdgesToExecutionEdges(execEdges);
      const result = await window.electronAPI.neuronSearchRank({
        useChatTemplate,
        modelDir: loadState.modelDir,
        config: rawResult.raw_config,
        nodes: executionNodeDefs,
        edges: executionEdges,
        inputText: neuronSearchInputText.trim() || undefined,
        neuronKind,
        targetLayers: parsedTargetLayers,
        topK: neuronTopK,
        aggregation: neuronAggregation,
      });
      setNeuronSearchRunState({ status: "done", result });
    } catch (err) {
      setNeuronSearchRunState({
        status: "done",
        result: {
          success: false,
          error_type: "UnexpectedError",
          message: err instanceof Error ? err.message : String(err),
        },
      });
    }
  }, [
    useChatTemplate,
    rawResult,
    loadState,
    execEdges,
    executionNodeDefs,
    neuronSearchInputText,
    neuronKind,
    parsedTargetLayers,
    neuronTopK,
    neuronAggregation,
  ]);

  const handleRunNeuronProbe = useCallback(async () => {
    if (!rawResult || loadState.status !== "loaded") return;
    const inputTexts = probeInputTextsRaw
      .split("\n")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    if (inputTexts.length === 0) return;
    setNeuronSearchRunState({ status: "running" });
    try {
      const executionEdges = flowEdgesToExecutionEdges(execEdges);
      const result = await window.electronAPI.neuronSearchProbe({
        useChatTemplate,
        modelDir: loadState.modelDir,
        config: rawResult.raw_config,
        nodes: executionNodeDefs,
        edges: executionEdges,
        layerIndex: probeLayerIndex,
        kind: probeKind,
        neuronIndex: probeNeuronIndex,
        inputTexts,
        aggregation: neuronAggregation,
      });
      setNeuronSearchRunState({ status: "done", result });
    } catch (err) {
      setNeuronSearchRunState({
        status: "done",
        result: {
          success: false,
          error_type: "UnexpectedError",
          message: err instanceof Error ? err.message : String(err),
        },
      });
    }
  }, [
    useChatTemplate,
    rawResult,
    loadState,
    execEdges,
    executionNodeDefs,
    probeLayerIndex,
    probeKind,
    probeNeuronIndex,
    probeInputTextsRaw,
    neuronAggregation,
  ]);

  const handleRunNeuronSearch = useCallback(() => {
    if (neuronSearchMode === "rank") {
      void handleRunNeuronRank();
    } else {
      void handleRunNeuronProbe();
    }
  }, [neuronSearchMode, handleRunNeuronRank, handleRunNeuronProbe]);

  // Neuron Searchのrank結果の行から「このニューロンをprobeで試す」を選んだときに、
  // probeモードへ切り替えて該当ニューロンの指定を引き継ぐ。
  const handleProbeThisNeuron = useCallback(
    (layerIndex: number, kind: "mlp" | "attention_head", neuronIndex: number) => {
      setNeuronSearchMode("probe");
      setProbeLayerIndex(layerIndex);
      setProbeKind(kind);
      setProbeNeuronIndex(neuronIndex);
    },
    []
  );

  // Layer Ablation Wizard(Phase2): チェックボックスで選択できるノードは
  // attention_block/mlp_block/layer_blockに限る(embedding/norm/lm_headを
  // ゼロ化しても意味のある比較にならないため、GUI側の選択肢からは除外する)。
  const ablatableNodes = useMemo(
    () =>
      executionNodeDefs.filter(
        (n) => n.role === "attention_block" || n.role === "mlp_block" || n.role === "layer_block"
      ),
    [executionNodeDefs]
  );

  const toggleAblationNode = useCallback((nodeId: string) => {
    setAblationSelectedNodeIds((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) next.delete(nodeId);
      else next.add(nodeId);
      return next;
    });
  }, []);

  const handleRunAblation = useCallback(async () => {
    if (!rawResult || loadState.status !== "loaded") return;
    if (ablationSelectedNodeIds.size === 0) return;
    setAblationRunState({ status: "running" });
    try {
      const executionEdges = flowEdgesToExecutionEdges(execEdges);
      const result = await window.electronAPI.runLayerAblation({
        useChatTemplate,
        modelDir: loadState.modelDir,
        config: rawResult.raw_config,
        nodes: executionNodeDefs,
        edges: executionEdges,
        inputText: ablationInputText.trim() || undefined,
        ablateNodeIds: Array.from(ablationSelectedNodeIds),
        compareGeneration: ablationCompareGeneration,
        maxNewTokens: ablationMaxNewTokens,
      });
      setAblationRunState({ status: "done", result });
    } catch (err) {
      setAblationRunState({
        status: "done",
        result: {
          success: false,
          error_type: "UnexpectedError",
          message: err instanceof Error ? err.message : String(err),
        },
      });
    }
  }, [
    useChatTemplate,
    rawResult,
    loadState,
    execEdges,
    executionNodeDefs,
    ablationInputText,
    ablationSelectedNodeIds,
    ablationCompareGeneration,
    ablationMaxNewTokens,
  ]);

  // Activation Steering Studio: 対象ノードの選択もLayer Ablation Wizardと同じ
  // ablatableNodes(attention_block/mlp_block/layer_blockに限る)を再利用する。
  const toggleSteeringNode = useCallback((nodeId: string) => {
    setSteeringTargetNodeIds((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) next.delete(nodeId);
      else next.add(nodeId);
      return next;
    });
  }, []);

  const parsedSteeringPositivePrompts = useMemo(
    () =>
      steeringPositivePromptsRaw
        .split("\n")
        .map((s) => s.trim())
        .filter((s) => s.length > 0),
    [steeringPositivePromptsRaw]
  );
  const parsedSteeringNegativePrompts = useMemo(
    () =>
      steeringNegativePromptsRaw
        .split("\n")
        .map((s) => s.trim())
        .filter((s) => s.length > 0),
    [steeringNegativePromptsRaw]
  );

  const handleRunSteering = useCallback(async () => {
    if (!rawResult || loadState.status !== "loaded") return;
    if (steeringTargetNodeIds.size === 0) return;
    if (parsedSteeringPositivePrompts.length === 0 || parsedSteeringNegativePrompts.length === 0) return;
    setSteeringRunState({ status: "running" });
    try {
      const executionEdges = flowEdgesToExecutionEdges(execEdges);
      const result = await window.electronAPI.activationSteering({
        useChatTemplate,
        modelDir: loadState.modelDir,
        config: rawResult.raw_config,
        nodes: executionNodeDefs,
        edges: executionEdges,
        positivePrompts: parsedSteeringPositivePrompts,
        negativePrompts: parsedSteeringNegativePrompts,
        targetNodeIds: Array.from(steeringTargetNodeIds),
        scale: steeringScale,
        inputText: steeringInputText.trim() || undefined,
        compareGeneration: steeringCompareGeneration,
        maxNewTokens: steeringMaxNewTokens,
        measureSycophancy: steeringMeasureSycophancy,
        measureRefusal: steeringMeasureRefusal,
        interventionMode: steeringInterventionMode,
      });
      setSteeringRunState({ status: "done", result });
    } catch (err) {
      setSteeringRunState({
        status: "done",
        result: {
          success: false,
          error_type: "UnexpectedError",
          message: err instanceof Error ? err.message : String(err),
        },
      });
    }
  }, [
    useChatTemplate,
    rawResult,
    loadState,
    execEdges,
    executionNodeDefs,
    parsedSteeringPositivePrompts,
    parsedSteeringNegativePrompts,
    steeringTargetNodeIds,
    steeringScale,
    steeringInputText,
    steeringCompareGeneration,
    steeringMaxNewTokens,
    steeringMeasureSycophancy,
    steeringMeasureRefusal,
    steeringInterventionMode,
  ]);

  // Sycophancy Reducer(2026-09-15追加): プリセットボタン1つで対照プロンプト欄・scale・
  // 迎合語彙カウント計測フラグをまとめて設定する。scaleを負の値にする理由:
  // positive=迎合的例文/negative=中立的例文という向きで差分ベクトルを計算するため、
  // 「迎合を減らす」には逆方向(負のscale)で適用する必要がある(2026-09-15確認事項)。
  const applySycophancyReducerPreset = useCallback(() => {
    setSteeringPositivePromptsRaw(SYCOPHANCY_POSITIVE_PROMPTS.join("\n"));
    setSteeringNegativePromptsRaw(SYCOPHANCY_NEGATIVE_PROMPTS.join("\n"));
    setSteeringScale(-2);
    setSteeringMeasureSycophancy(true);
  }, []);

  // Refusal Behavior Tuner(2026-09-22追加): プリセットボタン1つで対照プロンプト欄・介入方式・
  // 拒否定型句カウント計測フラグ・チャットテンプレート適用をまとめて設定する(漣さんとの確認事項:
  // Sycophancy Reducerと同じ評価方式・固定プリセット)。
  // - 介入方式は「方向除去」を初期値にする(拒否を減らす方向で、加算より出力が崩壊しにくい)。
  //   拒否を増やしたい場合は「加算」に切り替えてscaleを正の値(まず1程度)にする
  // - 拒否は「応答」モードでしか出ないため、チャットテンプレート適用もオンにする
  const applyRefusalTunerPreset = useCallback(() => {
    setSteeringPositivePromptsRaw(REFUSAL_POSITIVE_PROMPTS.join("\n"));
    setSteeringNegativePromptsRaw(REFUSAL_NEGATIVE_PROMPTS.join("\n"));
    setSteeringInterventionMode("ablate_direction");
    setSteeringScale(1);
    setSteeringMeasureRefusal(true);
    setSteeringInputText(REFUSAL_POSITIVE_PROMPTS[0]);
    setUseChatTemplate(true);
  }, []);

  const totalLayers = rawResult?.num_layers ?? 0;
  const layerToggleLabel = useMemo(() => {
    if (!totalLayers) return "";
    const layerLevelExpanded = Array.from(expandedGroups).filter(
      (k) => k.startsWith("layer-") && !k.includes("-attention") && !k.includes("-mlp")
    ).length;
    return `展開中 ${layerLevelExpanded} / ${totalLayers} 層`;
  }, [expandedGroups, totalLayers]);

  const totalParams = useMemo(() => {
    if (!rawResult) return null;
    let sum = 0;
    for (const n of rawResult.nodes) {
      if (!n.shape) continue;
      sum += n.shape.reduce((a, b) => a * b, 1);
    }
    return sum;
  }, [rawResult]);

  // 全層を完全展開(個別ノードまで): layer-N, layer-N-attention, layer-N-mlp を全て追加
  const expandAll = useCallback(() => {
    if (!rawResult || !totalLayers) return;
    const all = new Set<string>();
    for (let i = 0; i < totalLayers; i++) {
      all.add(`layer-${i}`);
      all.add(`layer-${i}-attention`);
      all.add(`layer-${i}-mlp`);
    }
    setExpandedGroups(all);
    applyGraph(rawResult, all);
  }, [rawResult, totalLayers, applyGraph]);

  const collapseAll = useCallback(() => {
    if (!rawResult) return;
    const none = new Set<string>();
    setExpandedGroups(none);
    applyGraph(rawResult, none);
  }, [rawResult, applyGraph]);

  const resetExecutionGraph = useCallback(() => {
    if (!rawResult) return;
    const none = new Set<number>();
    setExpandedExecLayers(none);
    applyExecutionGraph(rawResult, none);
  }, [rawResult, applyExecutionGraph]);

  const selectedData =
    selectedNode && selectedCount <= 1
      ? (selectedNode.data as unknown as LayerSummaryNodeData | NormalNodeData)
      : null;

  return (
    <div style={{ width: "100vw", height: "100vh", display: "flex", flexDirection: "column" }}>
      <header
        className="app-header"
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: 10,
          padding: "10px 16px",
          background: "var(--bg-panel)",
          borderBottom: "1px solid var(--border-default)",
        }}
      >
        <strong style={{ fontSize: 14, letterSpacing: 0.2 }}>解剖ビューア</strong>

        <button onClick={handleSelectModel} style={buttonStyle}>
          モデルフォルダを開く
        </button>

        {loadState.status === "loaded" && (
          <div style={{ display: "flex", gap: 4, marginLeft: 4 }}>
            <button
              onClick={() => setTab("dissect")}
              style={tab === "dissect" ? tabButtonActiveStyle : tabButtonStyle}
            >
              解剖
            </button>
            <button
              onClick={() => setTab("execute")}
              style={tab === "execute" ? tabButtonActiveStyle : tabButtonStyle}
            >
              実行
            </button>
            <button
              onClick={() => setTab("observe")}
              style={tab === "observe" ? tabButtonActiveStyle : tabButtonStyle}
            >
              観察
            </button>
          </div>
        )}

        {loadState.status === "loaded" && tab !== "dissect" && (
          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              fontSize: 11,
              color: useChatTemplate ? "var(--accent-text)" : "var(--text-secondary)",
              cursor: "pointer",
              whiteSpace: "nowrap",
            }}
            title="入力文をモデルのチャットテンプレート(tokenizer_config.jsonのchat_template)で包んでから実行する。Instructモデルで「応答」させたい場合にオン。実行・観察の全機能に共通で適用される"
          >
            <input
              type="checkbox"
              checked={useChatTemplate}
              onChange={(e) => setUseChatTemplate(e.target.checked)}
            />
            チャットテンプレート適用
          </label>
        )}

        {loadState.status === "loaded" && tab === "dissect" && (
          <>
            <div style={{ width: 1, height: 20, background: "var(--border-default)" }} />
            <button onClick={expandAll} style={buttonStyle}>
              全層展開
            </button>
            <button onClick={collapseAll} style={buttonStyle}>
              全層折りたたみ
            </button>
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{layerToggleLabel}</span>
          </>
        )}

        {loadState.status === "loaded" && tab === "execute" && (
          <>
            <div style={{ width: 1, height: 20, background: "var(--border-default)" }} />
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="入力テキスト"
              style={{
                ...buttonStyle,
                cursor: "text",
                width: 180,
                background: "var(--bg-app)",
              }}
            />
            <input
              type="number"
              min={1}
              max={200}
              value={maxNewTokens}
              onChange={(e) => setMaxNewTokens(Math.max(1, Number(e.target.value) || 1))}
              title="生成するトークン数"
              style={{
                ...buttonStyle,
                cursor: "text",
                width: 56,
                background: "var(--bg-app)",
              }}
            />
            <button
              onClick={handleRunExecution}
              disabled={runState.status === "running"}
              style={{ ...buttonStyle, background: "var(--accent)", color: "#0f1418", fontWeight: 600 }}
            >
              {runState.status === "running" ? "生成中..." : "実行"}
            </button>
            <button onClick={resetExecutionGraph} style={buttonStyle}>
              オリジナル構造に戻す
            </button>
            <button
              onClick={() => setShowSamplingOptions((v) => !v)}
              style={showSamplingOptions ? tabButtonActiveStyle : buttonStyle}
            >
              生成設定 {showSamplingOptions ? "▲" : "▼"}
            </button>
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              展開中 {expandedExecLayers.size} / {totalLayers} 層
            </span>
          </>
        )}

        {loadState.status === "loaded" && tab === "observe" && (
          <>
            <div style={{ width: 1, height: 20, background: "var(--border-default)" }} />
            <div style={{ display: "flex", gap: 4 }}>
              <button
                onClick={() => setObserveSubTab("internals")}
                style={observeSubTab === "internals" ? tabButtonActiveStyle : tabButtonStyle}
              >
                内部状態
              </button>
              <button
                onClick={() => setObserveSubTab("sensitivity")}
                style={observeSubTab === "sensitivity" ? tabButtonActiveStyle : tabButtonStyle}
              >
                Sensitivity Map
              </button>
              <button
                onClick={() => setObserveSubTab("journey")}
                style={observeSubTab === "journey" ? tabButtonActiveStyle : tabButtonStyle}
              >
                Token Journey
              </button>
              <button
                onClick={() => setObserveSubTab("battle")}
                style={observeSubTab === "battle" ? tabButtonActiveStyle : tabButtonStyle}
              >
                Battle View
              </button>
              <button
                onClick={() => setObserveSubTab("neuronSearch")}
                style={observeSubTab === "neuronSearch" ? tabButtonActiveStyle : tabButtonStyle}
              >
                Neuron Search
              </button>
              <button
                onClick={() => setObserveSubTab("ablation")}
                style={observeSubTab === "ablation" ? tabButtonActiveStyle : tabButtonStyle}
              >
                Layer Ablation
              </button>
              <button
                onClick={() => setObserveSubTab("steering")}
                style={observeSubTab === "steering" ? tabButtonActiveStyle : tabButtonStyle}
              >
                Activation Steering
              </button>
              <button
                onClick={() => setObserveSubTab("live")}
                style={observeSubTab === "live" ? tabButtonActiveStyle : tabButtonStyle}
                title="スライダーで層・ニューロン・ステアリングの強さを動かし、生成中に反映させる(Phase3、2026-09-24追加)"
              >
                Live Slider
              </button>
            </div>
            <div style={{ width: 1, height: 20, background: "var(--border-default)" }} />

            {observeSubTab === "internals" && (
              <>
                <input
                  type="text"
                  value={observeInputText}
                  onChange={(e) => setObserveInputText(e.target.value)}
                  placeholder="入力テキスト"
                  style={{
                    ...buttonStyle,
                    cursor: "text",
                    width: 180,
                    background: "var(--bg-app)",
                  }}
                />
                <button
                  onClick={handleRunInternals}
                  disabled={internalsRunState.status === "running"}
                  style={{ ...buttonStyle, background: "var(--accent)", color: "#0f1418", fontWeight: 600 }}
                >
                  {internalsRunState.status === "running" ? "解析中..." : "観察"}
                </button>
                <label
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    fontSize: 11,
                    color: "var(--text-secondary)",
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={aggregateHeads}
                    onChange={(e) => setAggregateHeads(e.target.checked)}
                  />
                  ヘッド平均で表示
                </label>
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  「実行」タブで組んだグラフ構造がそのまま使われます
                </span>
              </>
            )}

            {observeSubTab === "sensitivity" && (
              <>
                <input
                  type="text"
                  value={sensitivityInputText}
                  onChange={(e) => setSensitivityInputText(e.target.value)}
                  placeholder="入力テキスト（2トークン以上）"
                  style={{
                    ...buttonStyle,
                    cursor: "text",
                    width: 220,
                    background: "var(--bg-app)",
                  }}
                />
                <select
                  value={replacementMode}
                  onChange={(e) => setReplacementMode(e.target.value as ReplacementMode)}
                  style={{ ...buttonStyle, cursor: "pointer" }}
                  title="各トークンをどう変更して影響度を測るか"
                >
                  <option value="delete">削除</option>
                  <option value="pad">パディングに置換</option>
                  <option value="random">ランダムトークンに置換</option>
                  <option value="custom">指定トークンIDに置換</option>
                </select>
                {replacementMode === "custom" && (
                  <input
                    type="number"
                    value={customTokenId}
                    onChange={(e) =>
                      setCustomTokenId(e.target.value === "" ? "" : Number(e.target.value))
                    }
                    placeholder="トークンID"
                    style={{
                      ...buttonStyle,
                      cursor: "text",
                      width: 90,
                      background: "var(--bg-app)",
                    }}
                  />
                )}
                <button
                  onClick={handleRunSensitivity}
                  disabled={
                    sensitivityRunState.status === "running" ||
                    (replacementMode === "custom" && customTokenId === "")
                  }
                  style={{ ...buttonStyle, background: "var(--accent)", color: "#0f1418", fontWeight: 600 }}
                >
                  {sensitivityRunState.status === "running" ? "分析中..." : "分析"}
                </button>
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  各トークンを1つずつ変更し、出力への影響度(KLダイバージェンス)を測定します
                </span>
              </>
            )}

            {observeSubTab === "journey" && (
              <>
                <input
                  type="text"
                  value={journeyInputText}
                  onChange={(e) => setJourneyInputText(e.target.value)}
                  placeholder="入力テキスト"
                  style={{
                    ...buttonStyle,
                    cursor: "text",
                    width: 180,
                    background: "var(--bg-app)",
                  }}
                />
                <button
                  onClick={handleRunTokenJourney}
                  disabled={journeyRunState.status === "running"}
                  style={{ ...buttonStyle, background: "var(--accent)", color: "#0f1418", fontWeight: 600 }}
                >
                  {journeyRunState.status === "running" ? "追跡中..." : "追跡"}
                </button>
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  各トークンが、層を通過するごとに埋め込み空間上でどの単語に近づくかを追跡します
                </span>
              </>
            )}

            {observeSubTab === "battle" && (
              <>
                <button
                  onClick={handleAddBattleModel}
                  disabled={battleAddingModel}
                  style={buttonStyle}
                >
                  {battleAddingModel ? "読み込み中..." : "+ モデルを追加"}
                </button>
                <input
                  type="text"
                  value={battleInputText}
                  onChange={(e) => setBattleInputText(e.target.value)}
                  placeholder="入力テキスト"
                  style={{
                    ...buttonStyle,
                    cursor: "text",
                    width: 160,
                    background: "var(--bg-app)",
                  }}
                />
                <label
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    fontSize: 11,
                    color: "var(--text-secondary)",
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={battleCompareGeneration}
                    onChange={(e) => setBattleCompareGeneration(e.target.checked)}
                  />
                  生成結果
                </label>
                <label
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    fontSize: 11,
                    color: "var(--text-secondary)",
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={battleCompareInternals}
                    onChange={(e) => setBattleCompareInternals(e.target.checked)}
                  />
                  内部状態
                </label>
                {battleCompareGeneration && (
                  <input
                    type="number"
                    value={battleMaxNewTokens}
                    onChange={(e) => setBattleMaxNewTokens(Number(e.target.value))}
                    min={1}
                    title="生成トークン数"
                    style={{ ...buttonStyle, cursor: "text", width: 60, background: "var(--bg-app)" }}
                  />
                )}
                <button
                  onClick={handleRunBattle}
                  disabled={
                    battleRunState.status === "running" ||
                    battleModels.length === 0 ||
                    (!battleCompareGeneration && !battleCompareInternals)
                  }
                  style={{ ...buttonStyle, background: "var(--accent)", color: "#0f1418", fontWeight: 600 }}
                >
                  {battleRunState.status === "running" ? "比較中..." : "比較"}
                </button>
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  追加したモデル({battleModels.length}個)それぞれに、同じ入力での生成結果・内部状態を横並びで比較します
                </span>
              </>
            )}

            {observeSubTab === "neuronSearch" && (
              <>
                <div style={{ display: "flex", gap: 4 }}>
                  <button
                    onClick={() => setNeuronSearchMode("rank")}
                    style={{
                      ...(neuronSearchMode === "rank" ? tabButtonActiveStyle : tabButtonStyle),
                      padding: "4px 10px",
                    }}
                  >
                    ランキング
                  </button>
                  <button
                    onClick={() => setNeuronSearchMode("probe")}
                    style={{
                      ...(neuronSearchMode === "probe" ? tabButtonActiveStyle : tabButtonStyle),
                      padding: "4px 10px",
                    }}
                  >
                    Probe
                  </button>
                </div>
                {neuronSearchMode === "rank" && (
                  <>
                    <input
                      type="text"
                      value={neuronSearchInputText}
                      onChange={(e) => setNeuronSearchInputText(e.target.value)}
                      placeholder="入力テキスト"
                      style={{ ...buttonStyle, cursor: "text", width: 140, background: "var(--bg-app)" }}
                    />
                    <select
                      value={neuronKind}
                      onChange={(e) => setNeuronKind(e.target.value as NeuronKind)}
                      style={{ ...buttonStyle, cursor: "pointer" }}
                      title="探索対象のニューロン種類"
                    >
                      <option value="both">MLP+Attention</option>
                      <option value="mlp">MLPのみ</option>
                      <option value="attention_head">Attentionヘッドのみ</option>
                    </select>
                    <input
                      type="text"
                      value={neuronTargetLayersText}
                      onChange={(e) => setNeuronTargetLayersText(e.target.value)}
                      placeholder="対象層(空欄=全層。例: 0,3,7)"
                      style={{ ...buttonStyle, cursor: "text", width: 160, background: "var(--bg-app)" }}
                    />
                    <input
                      type="number"
                      value={neuronTopK}
                      onChange={(e) => setNeuronTopK(Number(e.target.value))}
                      min={1}
                      title="上位何件を表示するか"
                      style={{ ...buttonStyle, cursor: "text", width: 60, background: "var(--bg-app)" }}
                    />
                  </>
                )}
                {neuronSearchMode === "probe" && (
                  <>
                    <input
                      type="number"
                      value={probeLayerIndex}
                      onChange={(e) => setProbeLayerIndex(Number(e.target.value))}
                      min={0}
                      title="層番号"
                      style={{ ...buttonStyle, cursor: "text", width: 60, background: "var(--bg-app)" }}
                    />
                    <select
                      value={probeKind}
                      onChange={(e) => setProbeKind(e.target.value as "mlp" | "attention_head")}
                      style={{ ...buttonStyle, cursor: "pointer" }}
                    >
                      <option value="mlp">MLP</option>
                      <option value="attention_head">Attentionヘッド</option>
                    </select>
                    <input
                      type="number"
                      value={probeNeuronIndex}
                      onChange={(e) => setProbeNeuronIndex(Number(e.target.value))}
                      min={0}
                      title="ニューロン番号"
                      style={{ ...buttonStyle, cursor: "text", width: 80, background: "var(--bg-app)" }}
                    />
                    <textarea
                      value={probeInputTextsRaw}
                      onChange={(e) => setProbeInputTextsRaw(e.target.value)}
                      placeholder="入力文(1行に1つ)"
                      rows={1}
                      style={{
                        ...buttonStyle,
                        cursor: "text",
                        width: 200,
                        background: "var(--bg-app)",
                        resize: "vertical",
                        fontFamily: "inherit",
                      }}
                    />
                  </>
                )}
                <select
                  value={neuronAggregation}
                  onChange={(e) => setNeuronAggregation(e.target.value as NeuronAggregation)}
                  style={{ ...buttonStyle, cursor: "pointer" }}
                  title="トークン位置方向の集約方法"
                >
                  <option value="max_abs">最大絶対値</option>
                  <option value="mean_abs">平均絶対値</option>
                </select>
                <button
                  onClick={handleRunNeuronSearch}
                  disabled={neuronSearchRunState.status === "running"}
                  style={{ ...buttonStyle, background: "var(--accent)", color: "#0f1418", fontWeight: 600 }}
                >
                  {neuronSearchRunState.status === "running" ? "検索中..." : "検索"}
                </button>
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  「実行」タブで組んだグラフ構造がそのまま使われます
                </span>
              </>
            )}

            {observeSubTab === "ablation" && (
              <>
                <input
                  type="text"
                  value={ablationInputText}
                  onChange={(e) => setAblationInputText(e.target.value)}
                  placeholder="入力テキスト"
                  style={{ ...buttonStyle, cursor: "text", width: 160, background: "var(--bg-app)" }}
                />
                <label
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    fontSize: 11,
                    color: "var(--text-secondary)",
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={ablationCompareGeneration}
                    onChange={(e) => setAblationCompareGeneration(e.target.checked)}
                  />
                  生成テキストも比較
                </label>
                {ablationCompareGeneration && (
                  <input
                    type="number"
                    value={ablationMaxNewTokens}
                    onChange={(e) => setAblationMaxNewTokens(Number(e.target.value))}
                    min={1}
                    title="生成トークン数"
                    style={{ ...buttonStyle, cursor: "text", width: 60, background: "var(--bg-app)" }}
                  />
                )}
                <button
                  onClick={handleRunAblation}
                  disabled={ablationRunState.status === "running" || ablationSelectedNodeIds.size === 0}
                  style={{ ...buttonStyle, background: "var(--accent)", color: "#0f1418", fontWeight: 600 }}
                >
                  {ablationRunState.status === "running" ? "比較中..." : "無効化して比較"}
                </button>
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                  下のリストから無効化したいノードを選んでください({ablationSelectedNodeIds.size}個選択中)
                </span>
              </>
            )}

            {observeSubTab === "steering" && (
              <SteeringToolbar
                applyRefusalTunerPreset={applyRefusalTunerPreset}
                applySycophancyReducerPreset={applySycophancyReducerPreset}
                handleRunSteering={handleRunSteering}
                parsedSteeringNegativePrompts={parsedSteeringNegativePrompts}
                parsedSteeringPositivePrompts={parsedSteeringPositivePrompts}
                steeringCompareGeneration={steeringCompareGeneration}
                steeringInputText={steeringInputText}
                steeringInterventionMode={steeringInterventionMode}
                steeringMaxNewTokens={steeringMaxNewTokens}
                steeringMeasureRefusal={steeringMeasureRefusal}
                steeringMeasureSycophancy={steeringMeasureSycophancy}
                steeringNegativePromptsRaw={steeringNegativePromptsRaw}
                steeringPositivePromptsRaw={steeringPositivePromptsRaw}
                steeringRunState={steeringRunState}
                steeringScale={steeringScale}
                steeringTargetNodeIds={steeringTargetNodeIds}
                setSteeringCompareGeneration={setSteeringCompareGeneration}
                setSteeringInputText={setSteeringInputText}
                setSteeringInterventionMode={setSteeringInterventionMode}
                setSteeringMaxNewTokens={setSteeringMaxNewTokens}
                setSteeringMeasureRefusal={setSteeringMeasureRefusal}
                setSteeringMeasureSycophancy={setSteeringMeasureSycophancy}
                setSteeringNegativePromptsRaw={setSteeringNegativePromptsRaw}
                setSteeringPositivePromptsRaw={setSteeringPositivePromptsRaw}
                setSteeringScale={setSteeringScale}
              />
            )}
          </>
        )}

        {loadState.status === "loading" && (
          <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>
            解析中: {loadState.modelDir}
          </span>
        )}
        {loadState.status === "error" && (
          <span style={{ fontSize: 12, color: "#e08a8a" }}>エラー: {loadState.message}</span>
        )}
        {loadState.status === "loaded" && (
          <span
            style={{
              fontSize: 12,
              color: "var(--text-secondary)",
              fontFamily: "var(--font-mono)",
              marginLeft: tab === "dissect" ? 0 : "auto",
            }}
          >
            {loadState.summary.model_type} · {loadState.summary.num_layers ?? "?"} layers ·{" "}
            hidden={loadState.summary.hidden_size ?? "?"}
            {totalParams !== null && <> · {formatParamCount(totalParams)} params</>}
          </span>
        )}

        {tab === "dissect" && selectedCount > 1 && (
          <span style={{ fontSize: 12, color: "var(--accent-text)", marginLeft: "auto" }}>
            {selectedCount} 個選択中
          </span>
        )}
      </header>

      {loadState.status === "loaded" && tab === "execute" && showSamplingOptions && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            padding: "8px 16px",
            background: "var(--bg-panel)",
            borderBottom: "1px solid var(--border-default)",
            fontSize: 11,
          }}
        >
          <label style={{ display: "flex", alignItems: "center", gap: 4, color: "var(--text-secondary)" }}>
            Temperature
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(Math.max(0, Number(e.target.value) || 0))}
              title="0 = 貪欲デコード(常に最大確率のトークン)。値が大きいほどランダム性が増す"
              style={{ ...buttonStyle, cursor: "text", width: 56, background: "var(--bg-app)" }}
            />
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 4, color: "var(--text-secondary)" }}>
            Top-k
            <input
              type="number"
              min={0}
              value={topK}
              onChange={(e) => setTopK(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))}
              placeholder="無効"
              title="確率上位k個のトークンだけに絞ってサンプリング。空欄で無効"
              style={{ ...buttonStyle, cursor: "text", width: 56, background: "var(--bg-app)" }}
            />
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 4, color: "var(--text-secondary)" }}>
            Top-p
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={topP}
              onChange={(e) => setTopP(e.target.value === "" ? "" : Math.min(1, Math.max(0, Number(e.target.value))))}
              placeholder="無効"
              title="累積確率がpを超えるまでの上位トークンからサンプリング。空欄で無効"
              style={{ ...buttonStyle, cursor: "text", width: 56, background: "var(--bg-app)" }}
            />
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 4, color: "var(--text-secondary)" }}>
            繰り返しペナルティ
            <input
              type="number"
              min={1}
              max={2}
              step={0.05}
              value={repetitionPenalty}
              onChange={(e) => setRepetitionPenalty(Math.max(1, Number(e.target.value) || 1))}
              title="1.0で無効。大きいほど既出トークンの繰り返しを抑制する"
              style={{ ...buttonStyle, cursor: "text", width: 56, background: "var(--bg-app)" }}
            />
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 4, color: "var(--text-secondary)" }}>
            繰り返し禁止n-gram
            <input
              type="number"
              min={0}
              value={noRepeatNgramSize}
              onChange={(e) =>
                setNoRepeatNgramSize(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))
              }
              placeholder="無効"
              title="指定したnトークンの並び(n-gram)が2回出現するのを禁止する。同じ文パターンの繰り返しを強力に抑制。貪欲デコード・サンプリングどちらでも有効。空欄で無効"
              style={{ ...buttonStyle, cursor: "text", width: 56, background: "var(--bg-app)" }}
            />
          </label>
          <span style={{ color: "var(--text-muted)" }}>
            {temperature === 0
              ? "貪欲デコード中(常に最大確率のトークンを選択)"
              : "サンプリング中(ランダム性あり)"}
          </span>
        </div>
      )}

      <div style={{ flex: 1, position: "relative", background: "var(--bg-app)" }}>
        {tab === "dissect" && (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={handleNodeClick}
            onSelectionChange={handleSelectionChange}
            fitView
            minZoom={0.05}
            selectionOnDrag
            selectionMode={SelectionMode.Partial}
            panOnDrag={[1, 2]}
            multiSelectionKeyCode="Shift"
          >
            <Background color="var(--border-default)" gap={24} size={1} />
            <Controls />
            <MiniMap
              nodeColor={() => "var(--border-strong)"}
              style={{ background: "var(--bg-panel)" }}
            />
          </ReactFlow>
        )}

        {tab === "execute" && (
          <ReactFlow
            nodes={displayExecNodes}
            edges={execEdges}
            onNodesChange={handleExecNodesChange}
            onEdgesChange={handleExecEdgesChange}
            onConnect={handleExecConnect}
            onNodeClick={handleExecNodeClick}
            fitView
            minZoom={0.1}
            panOnDrag={[1, 2]}
          >
            <Background color="var(--border-default)" gap={24} size={1} />
            <Controls />
          </ReactFlow>
        )}

        {tab === "observe" && observeSubTab === "internals" && (
          <div
            style={{
              width: "100%",
              height: "100%",
              overflow: "auto",
              padding: 20,
              display: "flex",
              gap: 20,
            }}
          >
            {internalsRunState.status === "idle" && (
              <div style={emptyStateStyle}>
                入力テキストを指定して「観察」を押すと、Attention重みと各層の活性化値が表示されます
              </div>
            )}
            {internalsRunState.status === "running" && (
              <div style={{ color: "var(--text-secondary)" }}>解析中...</div>
            )}
            {internalsRunState.status === "done" && !internalsRunState.result.success && (
              <div style={{ color: "#e08a8a" }}>
                エラー ({internalsRunState.result.error_type}): {internalsRunState.result.message}
              </div>
            )}
            {internalsRunState.status === "done" && internalsRunState.result.success && (
              <>
                <div style={{ flex: "0 0 auto" }}>
                  <div
                    style={{
                      fontSize: 12,
                      color: "var(--text-secondary)",
                      marginBottom: 8,
                      display: "flex",
                      gap: 8,
                      alignItems: "center",
                    }}
                  >
                    <span>Attentionヒートマップ:</span>
                    <select
                      value={selectedAttentionNodeId ?? ""}
                      onChange={(e) => setSelectedAttentionNodeId(e.target.value)}
                      style={{ ...buttonStyle, cursor: "pointer" }}
                    >
                      {Object.keys(internalsRunState.result.attention_weights).map((nodeId) => (
                        <option key={nodeId} value={nodeId}>
                          {nodeId}
                        </option>
                      ))}
                    </select>
                  </div>
                  {selectedAttentionNodeId &&
                    internalsRunState.result.attention_weights[selectedAttentionNodeId] && (
                      <div
                        style={{
                          background: "var(--bg-panel)",
                          border: "1px solid var(--border-default)",
                          borderRadius: "var(--radius-md)",
                          padding: 12,
                          overflow: "auto",
                        }}
                      >
                        <AttentionHeatmap
                          entry={internalsRunState.result.attention_weights[selectedAttentionNodeId]}
                          tokenLabels={
                            internalsRunState.result.input_tokens_text ??
                            internalsRunState.result.input_token_ids.map((id) => String(id))
                          }
                        />
                      </div>
                    )}
                </div>

                <div style={{ flex: "1 1 auto", minWidth: 280 }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: 8,
                    }}
                  >
                    <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                      層ごとの活性化値統計(std=出力の広がり)
                    </div>
                    <div style={{ display: "flex", gap: 4 }}>
                      <button
                        onClick={() => setActivationScale("linear")}
                        style={{
                          ...buttonStyle,
                          padding: "3px 8px",
                          fontSize: 10,
                          ...(activationScale === "linear" ? tabButtonActiveStyle : {}),
                        }}
                        title="stdの値をそのまま棒の長さに使う"
                      >
                        線形
                      </button>
                      <button
                        onClick={() => setActivationScale("log")}
                        style={{
                          ...buttonStyle,
                          padding: "3px 8px",
                          fontSize: 10,
                          ...(activationScale === "log" ? tabButtonActiveStyle : {}),
                        }}
                        title="桁の異なる値も見比べやすい対数スケールで表示する"
                      >
                        対数
                      </button>
                    </div>
                  </div>
                  <div
                    style={{
                      background: "var(--bg-panel)",
                      border: "1px solid var(--border-default)",
                      borderRadius: "var(--radius-md)",
                      padding: 12,
                    }}
                  >
                    <ActivationStatsList
                      stats={internalsRunState.result.activation_stats}
                      order={internalsRunState.result.executed_order}
                      scale={activationScale}
                    />
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {tab === "observe" && observeSubTab === "sensitivity" && (
          <div
            style={{
              width: "100%",
              height: "100%",
              overflow: "auto",
              padding: 20,
            }}
          >
            {sensitivityRunState.status === "idle" && (
              <div style={emptyStateStyle}>
                入力テキストと変更方法を指定して「分析」を押すと、各トークンの重要度が表示されます
              </div>
            )}
            {sensitivityRunState.status === "running" && (
              <div style={{ color: "var(--text-secondary)" }}>
                分析中...（トークン数ぶんの追加推論が必要なため、時間がかかることがあります）
              </div>
            )}
            {sensitivityRunState.status === "done" && !sensitivityRunState.result.success && (
              <div style={{ color: "#e08a8a" }}>
                エラー ({sensitivityRunState.result.error_type}): {sensitivityRunState.result.message}
              </div>
            )}
            {sensitivityRunState.status === "done" && sensitivityRunState.result.success && (
              <div
                style={{
                  background: "var(--bg-panel)",
                  border: "1px solid var(--border-default)",
                  borderRadius: "var(--radius-md)",
                  padding: 16,
                  maxWidth: 900,
                }}
              >
                <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 12 }}>
                  変更方法: {sensitivityRunState.result.replacement_mode}
                </div>
                <SensitivityHeatmap
                  results={sensitivityRunState.result.results}
                  tokenLabels={
                    sensitivityRunState.result.input_tokens_text ??
                    sensitivityRunState.result.input_token_ids.map((id) => String(id))
                  }
                />
              </div>
            )}
          </div>
        )}

        {tab === "observe" && observeSubTab === "journey" && (
          <div
            style={{
              width: "100%",
              height: "100%",
              overflow: "auto",
              padding: 20,
            }}
          >
            {journeyRunState.status === "idle" && (
              <div style={emptyStateStyle}>
                入力テキストを指定して「追跡」を押すと、各トークンが層ごとにどの単語へ近づくかが表示されます
              </div>
            )}
            {journeyRunState.status === "running" && (
              <div style={{ color: "var(--text-secondary)" }}>追跡中...</div>
            )}
            {journeyRunState.status === "done" && !journeyRunState.result.success && (
              <div style={{ color: "#e08a8a" }}>
                エラー ({journeyRunState.result.error_type}): {journeyRunState.result.message}
              </div>
            )}
            {journeyRunState.status === "done" && journeyRunState.result.success && (
              <div
                style={{
                  background: "var(--bg-panel)",
                  border: "1px solid var(--border-default)",
                  borderRadius: "var(--radius-md)",
                  padding: 16,
                }}
              >
                <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginBottom: 12 }}>
                  セルの色が濃いほど、そのトークン位置とブロック出力のコサイン類似度が高いことを示す。
                  各セルは、そのブロックを通過した時点で埋め込み空間上最も近い単語(top-1)。
                </div>
                <TokenJourneyView
                  blockOrder={journeyRunState.result.block_order}
                  journey={journeyRunState.result.journey}
                  tokenLabels={
                    journeyRunState.result.input_tokens_text ??
                    journeyRunState.result.input_token_ids.map((id) => String(id))
                  }
                  shortenBlockId={shortenBlockId}
                />
              </div>
            )}
          </div>
        )}

        {tab === "observe" && observeSubTab === "battle" && (
          <div
            style={{
              width: "100%",
              height: "100%",
              overflow: "auto",
              padding: 20,
            }}
          >
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>
                比較対象モデル({battleModels.length}個) — 各モデルは自分自身の構造(層数・hidden_size)
                通りのデフォルトグラフで実行されます(「実行」タブで組んだグラフとは独立)
              </div>
              {battleModels.length === 0 ? (
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  「+ モデルを追加」から比較したいモデルを2個以上選んでください(1個でも動作します)
                </div>
              ) : (
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {battleModels.map((m) => (
                    <div
                      key={m.modelDir}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        background: "var(--bg-panel)",
                        border: "1px solid var(--border-default)",
                        borderRadius: "var(--radius-sm)",
                        padding: "4px 8px",
                        fontSize: 11.5,
                      }}
                    >
                      <span>{m.label}</span>
                      <button
                        onClick={() => handleRemoveBattleModel(m.modelDir)}
                        title="この モデルを比較対象から外す"
                        style={{
                          ...buttonStyle,
                          padding: "1px 6px",
                          fontSize: 11,
                          color: "#e08a8a",
                        }}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {battleRunState.status === "idle" && (
              <div style={emptyStateStyle}>
                モデルを追加し、入力テキストを指定して「比較」を押すと、生成結果・内部状態が横並びで表示されます
              </div>
            )}
            {battleRunState.status === "running" && (
              <div style={{ color: "var(--text-secondary)" }}>
                比較中...（複数モデルを同時にロードするため、時間がかかることがあります）
              </div>
            )}
            {battleRunState.status === "done" && !battleRunState.result.success && (
              <div style={{ color: "#e08a8a" }}>
                エラー ({battleRunState.result.error_type}): {battleRunState.result.message}
              </div>
            )}
            {battleRunState.status === "done" && battleRunState.result.success && (
              <BattleView results={battleRunState.result.results} activationScale={activationScale} />
            )}
          </div>
        )}

        {tab === "observe" && observeSubTab === "neuronSearch" && (
          <div
            style={{
              width: "100%",
              height: "100%",
              overflow: "auto",
              padding: 20,
            }}
          >
            {neuronSearchRunState.status === "idle" && (
              <div style={emptyStateStyle}>
                {neuronSearchMode === "rank"
                  ? "入力テキストを指定して「検索」を押すと、強く反応したニューロンがランキング表示されます"
                  : "層・種類・ニューロン番号・入力文を指定して「検索」を押すと、各入力への反応が表示されます"}
              </div>
            )}
            {neuronSearchRunState.status === "running" && (
              <div style={{ color: "var(--text-secondary)" }}>検索中...</div>
            )}
            {neuronSearchRunState.status === "done" && (
              <NeuronSearchView
                result={neuronSearchRunState.result}
                onProbeThisNeuron={handleProbeThisNeuron}
              />
            )}
          </div>
        )}

        {tab === "observe" && observeSubTab === "ablation" && (
          <div
            style={{
              width: "100%",
              height: "100%",
              overflow: "auto",
              padding: 20,
              display: "flex",
              gap: 20,
            }}
          >
            <div style={{ flex: "0 0 260px" }}>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>
                無効化するノードを選択(「実行」タブで組んだ現在のグラフ構造から選べます)
              </div>
              {ablatableNodes.length === 0 ? (
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  無効化可能なノードがありません(モデルを読み込んでください)
                </div>
              ) : (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 4,
                    maxHeight: "70vh",
                    overflow: "auto",
                    background: "var(--bg-panel)",
                    border: "1px solid var(--border-default)",
                    borderRadius: "var(--radius-md)",
                    padding: 10,
                  }}
                >
                  {ablatableNodes.map((n) => (
                    <label
                      key={n.id}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        fontSize: 12,
                        color: "var(--text-primary)",
                        cursor: "pointer",
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={ablationSelectedNodeIds.has(n.id)}
                        onChange={() => toggleAblationNode(n.id)}
                      />
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{n.id}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>

            <div style={{ flex: "1 1 auto", minWidth: 280 }}>
              {ablationRunState.status === "idle" && (
                <div style={{ color: "var(--text-muted)", fontSize: 13 }}>
                  ノードを選び、入力テキストを指定して「無効化して比較」を押すと、通常時との違いが表示されます
                </div>
              )}
              {ablationRunState.status === "running" && (
                <div style={{ color: "var(--text-secondary)" }}>比較中...</div>
              )}
              {ablationRunState.status === "done" && (
                <LayerAblationView result={ablationRunState.result} />
              )}
            </div>
          </div>
        )}

        {tab === "observe" && observeSubTab === "live" && loadState.status === "loaded" && rawResult && (
          // Live Slider(Phase3、2026-09-24追加)。状態・操作UIはLiveSliderPanel内で完結させている
          <LiveSliderPanel
            modelDir={loadState.modelDir}
            config={rawResult.raw_config}
            nodes={executionNodeDefs}
            edges={flowEdgesToExecutionEdges(execEdges)}
            useChatTemplate={useChatTemplate}
            steeringPositivePrompts={parsedSteeringPositivePrompts}
            steeringNegativePrompts={parsedSteeringNegativePrompts}
            steeringTargetNodeIds={Array.from(steeringTargetNodeIds)}
          />
        )}

        {tab === "observe" && observeSubTab === "steering" && (
          <div
            style={{
              width: "100%",
              height: "100%",
              overflow: "auto",
              padding: 20,
              display: "flex",
              gap: 20,
            }}
          >
            <div style={{ flex: "0 0 260px" }}>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>
                ステアリングを適用するノードを選択(「実行」タブで組んだ現在のグラフ構造から選べます)
              </div>
              {ablatableNodes.length === 0 ? (
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  適用可能なノードがありません(モデルを読み込んでください)
                </div>
              ) : (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 4,
                    maxHeight: "50vh",
                    overflow: "auto",
                    background: "var(--bg-panel)",
                    border: "1px solid var(--border-default)",
                    borderRadius: "var(--radius-md)",
                    padding: 10,
                  }}
                >
                  {ablatableNodes.map((n) => (
                    <label
                      key={n.id}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        fontSize: 12,
                        color: "var(--text-primary)",
                        cursor: "pointer",
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={steeringTargetNodeIds.has(n.id)}
                        onChange={() => toggleSteeringNode(n.id)}
                      />
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{n.id}</span>
                    </label>
                  ))}
                </div>
              )}
              <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 10 }}>
                positive: {parsedSteeringPositivePrompts.length}件 / negative:{" "}
                {parsedSteeringNegativePrompts.length}件
              </div>
            </div>

            <div style={{ flex: "1 1 auto", minWidth: 280 }}>
              {steeringRunState.status === "idle" && (
                <div style={{ color: "var(--text-muted)", fontSize: 13 }}>
                  positive/negativeプロンプトと適用先ノードを指定して「ステアリングを計算して比較」を押すと、
                  通常時との違いが表示されます
                </div>
              )}
              {steeringRunState.status === "running" && (
                <div style={{ color: "var(--text-secondary)" }}>計算中...</div>
              )}
              {steeringRunState.status === "done" && (
                <ActivationSteeringView result={steeringRunState.result} />
              )}
            </div>
          </div>
        )}

        {tab === "dissect" && <Legend />}

        {tab === "dissect" && selectedData && (
          <div style={infoPanelStyle}>
            {selectedData.kind === "layerSummary" ? (
              <>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>{selectedData.label}</div>
                <div style={{ color: "var(--text-secondary)" }}>
                  {selectedData.nodeCount} ノード ·{" "}
                  {selectedData.isExpanded ? "展開中" : "折りたたみ中"}
                </div>
                <div style={{ color: "var(--text-muted)", marginTop: 6 }}>
                  クリックで{selectedData.isExpanded ? "折りたたみ" : "展開"}
                </div>
              </>
            ) : (
              <>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>{selectedData.displayName}</div>
                <div style={{ color: "var(--text-secondary)", marginBottom: 2 }}>
                  役割: {selectedData.roleLabelJa}
                </div>
                {selectedData.shape && (
                  <div style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                    shape: [{selectedData.shape.join(", ")}]
                  </div>
                )}
                {selectedData.paramCount !== null && (
                  <div style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                    {formatParamCount(selectedData.paramCount)} params
                  </div>
                )}
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    wordBreak: "break-all",
                    marginTop: 8,
                    color: "var(--text-muted)",
                    fontSize: 10.5,
                  }}
                >
                  {selectedData.paramKey}
                </div>
              </>
            )}
          </div>
        )}

        {tab === "execute" && runState.status !== "idle" && (
          <div style={{ ...infoPanelStyle, width: 340 }}>
            {runState.status === "running" && (
              <div style={{ color: "var(--text-secondary)" }}>生成中...</div>
            )}
            {runState.status === "done" && runState.result.success && (
              <>
                <div style={{ fontWeight: 600, marginBottom: 6, color: "#8fd4a8" }}>
                  実行成功
                </div>
                {runState.result.generated_text !== null ? (
                  <div
                    style={{
                      color: "var(--text-primary)",
                      background: "var(--bg-app)",
                      border: "1px solid var(--border-default)",
                      borderRadius: "var(--radius-sm)",
                      padding: 8,
                      marginBottom: 8,
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                    }}
                  >
                    {runState.result.generated_text}
                  </div>
                ) : (
                  <div style={{ color: "var(--text-muted)", fontSize: 10.5, marginBottom: 8 }}>
                    (tokenizer.jsonが無いためテキスト表示不可。トークンIDのみ表示)
                  </div>
                )}
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    color: "var(--text-secondary)",
                    fontSize: 10.5,
                    wordBreak: "break-all",
                  }}
                >
                  input: [{runState.result.input_token_ids.join(", ")}]
                </div>
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    color: "var(--text-secondary)",
                    fontSize: 10.5,
                    wordBreak: "break-all",
                    marginTop: 2,
                  }}
                >
                  generated: [{runState.result.generated_token_ids.join(", ")}]
                </div>
                <div style={{ color: "var(--text-muted)", marginTop: 8, fontSize: 10.5 }}>
                  実行順序: {runState.result.executed_order.join(" → ")}
                </div>
                <div style={{ color: "var(--text-muted)", marginTop: 8, fontSize: 10.5 }}>
                  出力の意味的な妥当性は保証されません(Phase1の設計上の割り切り)。
                </div>
              </>
            )}
            {runState.status === "done" && !runState.result.success && (
              <>
                <div style={{ fontWeight: 600, marginBottom: 6, color: "#e08a8a" }}>
                  実行エラー ({runState.result.error_type})
                </div>
                <div style={{ color: "var(--text-secondary)", fontSize: 11 }}>
                  {runState.result.message}
                </div>
              </>
            )}
          </div>
        )}

        {loadState.status === "idle" && (
          <div style={emptyStateStyle}>
            「モデルフォルダを開く」からHuggingFace形式のモデルを選択してください
          </div>
        )}
      </div>
    </div>
  );
}
