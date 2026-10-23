import { contextBridge, ipcRenderer } from "electron";

/**
 * レンダラー(React側)に公開するAPI。
 * nodeIntegration: false のため、この橋渡し経由でのみ
 * メインプロセスの機能(ファイルダイアログ、Python起動)にアクセスできる。
 */
contextBridge.exposeInMainWorld("electronAPI", {
  // 2026-10-13: 表示言語をメインプロセスに伝える(Pythonのエラー文の言語になる)
  setLang: (lang: "ja" | "en"): void => ipcRenderer.send("lang:set", lang),

  selectModelDir: (): Promise<string | null> =>
    ipcRenderer.invoke("dialog:selectModelDir"),

  inspectModel: (modelDir: string): Promise<unknown> =>
    ipcRenderer.invoke("model:inspect", modelDir),

  executeModel: (payload: {
    modelDir: string;
    config: Record<string, unknown>;
    nodes: Array<{ id: string; role: string; layer_index: number | null }>;
    edges: Array<{ source: string; target: string }>;
    inputText?: string;
    useChatTemplate?: boolean;
    inputIds?: number[];
    maxNewTokens?: number;
    temperature?: number;
    topK?: number;
    topP?: number;
    repetitionPenalty?: number;
    noRepeatNgramSize?: number;
  }): Promise<unknown> => ipcRenderer.invoke("model:execute", payload),

  inspectInternals: (payload: {
    modelDir: string;
    config: Record<string, unknown>;
    nodes: Array<{ id: string; role: string; layer_index: number | null }>;
    edges: Array<{ source: string; target: string }>;
    inputText?: string;
    useChatTemplate?: boolean;
    inputIds?: number[];
    aggregateHeads?: boolean;
  }): Promise<unknown> => ipcRenderer.invoke("model:inspectInternals", payload),

  analyzeSensitivity: (payload: {
    modelDir: string;
    config: Record<string, unknown>;
    nodes: Array<{ id: string; role: string; layer_index: number | null }>;
    edges: Array<{ source: string; target: string }>;
    inputText?: string;
    useChatTemplate?: boolean;
    inputIds?: number[];
    replacementMode?: "delete" | "pad" | "random" | "custom";
    customTokenId?: number;
    padTokenId?: number;
  }): Promise<unknown> => ipcRenderer.invoke("model:analyzeSensitivity", payload),

  computeTokenJourney: (payload: {
    modelDir: string;
    config: Record<string, unknown>;
    nodes: Array<{ id: string; role: string; layer_index: number | null }>;
    edges: Array<{ source: string; target: string }>;
    inputText?: string;
    useChatTemplate?: boolean;
    inputIds?: number[];
    topK?: number;
  }): Promise<unknown> => ipcRenderer.invoke("model:computeTokenJourney", payload),

  battleCompare: (payload: {
    models: Array<{ modelDir: string; config: Record<string, unknown>; label?: string }>;
    inputText?: string;
    useChatTemplate?: boolean;
    inputIds?: number[];
    compareGeneration?: boolean;
    compareInternals?: boolean;
    maxNewTokens?: number;
    temperature?: number;
    topK?: number;
    topP?: number;
    repetitionPenalty?: number;
    noRepeatNgramSize?: number;
    aggregateHeads?: boolean;
  }): Promise<unknown> => ipcRenderer.invoke("model:battleCompare", payload),

  neuronSearchRank: (payload: {
    modelDir: string;
    config: Record<string, unknown>;
    nodes: Array<{ id: string; role: string; layer_index: number | null }>;
    edges: Array<{ source: string; target: string }>;
    inputText?: string;
    useChatTemplate?: boolean;
    inputIds?: number[];
    neuronKind?: "mlp" | "attention_head" | "both";
    targetLayers?: number[];
    topK?: number;
    aggregation?: "max_abs" | "mean_abs";
  }): Promise<unknown> => ipcRenderer.invoke("model:neuronSearchRank", payload),

  neuronSearchProbe: (payload: {
    modelDir: string;
    config: Record<string, unknown>;
    nodes: Array<{ id: string; role: string; layer_index: number | null }>;
    edges: Array<{ source: string; target: string }>;
    layerIndex: number;
    kind: "mlp" | "attention_head";
    neuronIndex: number;
    inputTexts: string[];
    useChatTemplate?: boolean;
    aggregation?: "max_abs" | "mean_abs";
  }): Promise<unknown> => ipcRenderer.invoke("model:neuronSearchProbe", payload),

  runLayerAblation: (payload: {
    modelDir: string;
    config: Record<string, unknown>;
    nodes: Array<{ id: string; role: string; layer_index: number | null }>;
    edges: Array<{ source: string; target: string }>;
    inputText?: string;
    useChatTemplate?: boolean;
    inputIds?: number[];
    ablateNodeIds: string[];
    compareGeneration?: boolean;
    maxNewTokens?: number;
    temperature?: number;
    topK?: number;
    topP?: number;
    repetitionPenalty?: number;
    noRepeatNgramSize?: number;
  }): Promise<unknown> => ipcRenderer.invoke("model:runLayerAblation", payload),

  // Noise Injection(2026-10-24追加)
  runNoiseInjection: (payload: {
    modelDir: string;
    config: Record<string, unknown>;
    nodes: Array<{ id: string; role: string; layer_index: number | null }>;
    edges: Array<{ source: string; target: string }>;
    inputText?: string;
    useChatTemplate?: boolean;
    targetNodeIds: string[];
    mode: "single" | "strength_sweep" | "layer_sweep";
    strength?: number;
    strengths?: number[];
    seed?: number;
    compareGeneration?: boolean;
    maxNewTokens?: number;
  }): Promise<unknown> => ipcRenderer.invoke("model:runNoiseInjection", payload),

  activationSteering: (payload: {
    modelDir: string;
    config: Record<string, unknown>;
    nodes: Array<{ id: string; role: string; layer_index: number | null }>;
    edges: Array<{ source: string; target: string }>;
    positivePrompts: string[];
    negativePrompts: string[];
    targetNodeIds: string[];
    scale?: number;
    inputText?: string;
    useChatTemplate?: boolean;
    inputIds?: number[];
    compareGeneration?: boolean;
    maxNewTokens?: number;
    temperature?: number;
    topK?: number;
    topP?: number;
    repetitionPenalty?: number;
    noRepeatNgramSize?: number;
    measureSycophancy?: boolean;
    measureRefusal?: boolean;
    interventionMode?: "add" | "ablate_direction";
  }): Promise<unknown> => ipcRenderer.invoke("model:activationSteering", payload),

  // Live Slider(Phase3、2026-09-24追加): 常駐するPythonセッションとのやり取り
  liveStart: (payload: {
    modelDir: string;
    config: Record<string, unknown>;
    nodes: Array<{ id: string; role: string; layer_index: number | null }>;
    edges: Array<{ source: string; target: string }>;
    useChatTemplate?: boolean;
  }): Promise<unknown> => ipcRenderer.invoke("live:start", payload),
  liveSend: (command: Record<string, unknown>): Promise<unknown> => ipcRenderer.invoke("live:send", command),
  liveStop: (): Promise<unknown> => ipcRenderer.invoke("live:stop"),
  // イベントの購読。戻り値の関数を呼ぶと購読を解除する
  onLiveEvent: (callback: (event: unknown) => void): (() => void) => {
    const listener = (_e: Electron.IpcRendererEvent, event: unknown) => callback(event);
    ipcRenderer.on("live:event", listener);
    return () => ipcRenderer.removeListener("live:event", listener);
  },
});
