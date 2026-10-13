import { app, BrowserWindow, ipcMain, dialog } from "electron";
import path from "node:path";
import { spawn, ChildProcessWithoutNullStreams } from "node:child_process";

// __dirname はビルド後 dist-electron/ 配下に配置される想定
const isDev = !app.isPackaged;

let mainWindow: BrowserWindow | null = null;

/**
 * Pythonの実行ファイル名。PYTHON_BIN環境変数があればそれを使う。
 * Windowsでは "python3" がMicrosoft Storeの案内用スタブ(実行すると code 9009 で失敗)を指すことが多いため、
 * 既定を "python" にする(2026-10-13、実機で model:inspect が code 9009 で失敗したため)。
 */
function defaultPythonBin(): string {
  return process.env.PYTHON_BIN || (process.platform === "win32" ? "python" : "python3");
}

/**
 * 画面の表示言語(2026-10-13、英語UI)。レンダラーから lang:set で受け取り、Python子プロセスに
 * 環境変数 AIDE_LANG で渡す(Python側のエラー文の言語。python/inference/i18n.py参照)。
 */
let uiLang: "ja" | "en" = "ja";
ipcMain.on("lang:set", (_event, lang: unknown) => {
  if (lang === "ja" || lang === "en") uiLang = lang;
});

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (isDev) {
    // Vite dev server
    mainWindow.loadURL("http://localhost:5173");
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
  }
}

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

/**
 * モデルフォルダ選択ダイアログを開く。
 * HuggingFace形式(config.json + safetensors)のディレクトリをユーザーに選ばせる。
 */
ipcMain.handle("dialog:selectModelDir", async () => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory"],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

/**
 * 指定ディレクトリのモデルをPythonサブプロセスで解析し、
 * ノードグラフ用のJSON構造を返す。
 *
 * 通信方式: Pythonスクリプトを都度起動し、標準出力にJSONを1行で出させて
 * それをパースする方式（MVPとしてはこれで十分。将来的に常駐プロセス化を検討）。
 */
ipcMain.handle("model:inspect", async (_event, modelDir: string) => {
  return new Promise((resolve, reject) => {
    // 開発時はリポジトリ直下のpython/を、パッケージ後はresourcesPath配下を参照
    const scriptPath = isDev
      ? path.join(__dirname, "../python/inspect_model.py")
      : path.join(process.resourcesPath, "python/inspect_model.py");

    const pythonBin = defaultPythonBin();
    const child: ChildProcessWithoutNullStreams = spawn(pythonBin, [
      scriptPath,
      modelDir,
    ], {
      env: { ...process.env, AIDE_LANG: uiLang, PYTHONIOENCODING: "utf-8" },
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (data) => {
      stdout += data.toString();
    });
    child.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`inspect_model.py failed (code ${code}): ${stderr}`));
        return;
      }
      try {
        const parsed = JSON.parse(stdout);
        resolve(parsed);
      } catch (e) {
        reject(new Error(`Failed to parse Python output as JSON: ${e}\nRaw output: ${stdout}`));
      }
    });

    child.on("error", (err) => {
      reject(new Error(`Failed to spawn Python process: ${err.message}`));
    });
  });
});

/**
 * ユーザーが組んだノードグラフ(ブロック単位)を実際に推論実行する。
 * execute_graph.py は標準入力からJSONを受け取る設計なので、spawn後にstdinへ書き込む。
 *
 * Phase1合意事項: グラフのバリデーションエラー(循環・shape不一致等)は
 * success:false の正常なJSONレスポンスとして返る。ここでのreject()は
 * プロセス起動失敗やJSON解析失敗など、真に予期しない失敗の場合のみ。
 */
ipcMain.handle(
  "model:execute",
  async (
    _event,
    payload: {
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
    }
  ) => {
    return new Promise((resolve, reject) => {
      const scriptPath = isDev
        ? path.join(__dirname, "../python/execute_graph.py")
        : path.join(process.resourcesPath, "python/execute_graph.py");

      const pythonBin = defaultPythonBin();
      const child: ChildProcessWithoutNullStreams = spawn(pythonBin, [scriptPath], {
        env: { ...process.env, AIDE_LANG: uiLang, PYTHONIOENCODING: "utf-8" },
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });
      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      child.on("close", (code) => {
        // execute_graph.py はバリデーションエラー時にcode=0を返す設計なので、
        // ここでのcode!==0チェックは「JSONすら出力できない致命的な失敗」の検出用
        try {
          const parsed = JSON.parse(stdout);
          resolve(parsed);
        } catch (e) {
          reject(
            new Error(
              `execute_graph.py failed (code ${code}): ${stderr || stdout || e}`
            )
          );
        }
      });

      child.on("error", (err) => {
        reject(new Error(`Failed to spawn Python process: ${err.message}`));
      });

      const requestBody = JSON.stringify({
        model_dir: payload.modelDir,
        config: payload.config,
        nodes: payload.nodes,
        edges: payload.edges,
        input_text: payload.inputText,
        use_chat_template: payload.useChatTemplate ?? false,
        input_ids: payload.inputText === undefined ? payload.inputIds ?? [1, 2, 3, 4, 5] : undefined,
        max_new_tokens: payload.maxNewTokens ?? 20,
        temperature: payload.temperature ?? 0.0,
        top_k: payload.topK,
        top_p: payload.topP,
        repetition_penalty: payload.repetitionPenalty ?? 1.0,
        no_repeat_ngram_size: payload.noRepeatNgramSize,
      });
      child.stdin.write(requestBody);
      child.stdin.end();
    });
  }
);

/**
 * Phase2: 内部状態可視化(Attention重み・活性化値)を取得する。
 * inspect_internals.py を呼び出し、1回のforward passの内部状態をJSONで受け取る。
 * model:execute とは別のPythonスクリプトを呼ぶ(目的が異なるため分離している)。
 */
ipcMain.handle(
  "model:inspectInternals",
  async (
    _event,
    payload: {
      modelDir: string;
      config: Record<string, unknown>;
      nodes: Array<{ id: string; role: string; layer_index: number | null }>;
      edges: Array<{ source: string; target: string }>;
      inputText?: string;
      useChatTemplate?: boolean;
      inputIds?: number[];
      aggregateHeads?: boolean;
    }
  ) => {
    return new Promise((resolve, reject) => {
      const scriptPath = isDev
        ? path.join(__dirname, "../python/inspect_internals.py")
        : path.join(process.resourcesPath, "python/inspect_internals.py");

      const pythonBin = defaultPythonBin();
      const child: ChildProcessWithoutNullStreams = spawn(pythonBin, [scriptPath], {
        env: { ...process.env, AIDE_LANG: uiLang, PYTHONIOENCODING: "utf-8" },
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });
      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      child.on("close", (code) => {
        try {
          const parsed = JSON.parse(stdout);
          resolve(parsed);
        } catch (e) {
          reject(
            new Error(
              `inspect_internals.py failed (code ${code}): ${stderr || stdout || e}`
            )
          );
        }
      });

      child.on("error", (err) => {
        reject(new Error(`Failed to spawn Python process: ${err.message}`));
      });

      const requestBody = JSON.stringify({
        model_dir: payload.modelDir,
        config: payload.config,
        nodes: payload.nodes,
        edges: payload.edges,
        input_text: payload.inputText,
        use_chat_template: payload.useChatTemplate ?? false,
        input_ids: payload.inputText === undefined ? payload.inputIds ?? [1, 2, 3, 4, 5] : undefined,
        aggregate_heads: payload.aggregateHeads ?? true,
      });
      child.stdin.write(requestBody);
      child.stdin.end();
    });
  }
);

/**
 * Phase2: Prompt Sensitivity Map — 入力文の各トークンを削除/置換した際の
 * 出力への影響度(KLダイバージェンス)を測定する。
 * analyze_sensitivity.py を呼び出す。入力トークン数ぶんの追加forward passが
 * 発生するため、他のIPCハンドラより実行に時間がかかることがある。
 */
ipcMain.handle(
  "model:analyzeSensitivity",
  async (
    _event,
    payload: {
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
    }
  ) => {
    return new Promise((resolve, reject) => {
      const scriptPath = isDev
        ? path.join(__dirname, "../python/analyze_sensitivity.py")
        : path.join(process.resourcesPath, "python/analyze_sensitivity.py");

      const pythonBin = defaultPythonBin();
      const child: ChildProcessWithoutNullStreams = spawn(pythonBin, [scriptPath], {
        env: { ...process.env, AIDE_LANG: uiLang, PYTHONIOENCODING: "utf-8" },
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });
      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      child.on("close", (code) => {
        try {
          const parsed = JSON.parse(stdout);
          resolve(parsed);
        } catch (e) {
          reject(
            new Error(
              `analyze_sensitivity.py failed (code ${code}): ${stderr || stdout || e}`
            )
          );
        }
      });

      child.on("error", (err) => {
        reject(new Error(`Failed to spawn Python process: ${err.message}`));
      });

      const requestBody = JSON.stringify({
        model_dir: payload.modelDir,
        config: payload.config,
        nodes: payload.nodes,
        edges: payload.edges,
        input_text: payload.inputText,
        use_chat_template: payload.useChatTemplate ?? false,
        input_ids: payload.inputText === undefined ? payload.inputIds ?? [1, 2, 3, 4, 5] : undefined,
        replacement_mode: payload.replacementMode ?? "pad",
        custom_token_id: payload.customTokenId,
        pad_token_id: payload.padTokenId,
      });
      child.stdin.write(requestBody);
      child.stdin.end();
    });
  }
);

/**
 * Phase2: Token Journey — 各ブロック(Attention直後・MLP直後)を通過した後の
 * トークンの隠れ状態を埋め込み空間に逆投影し、最も近い単語を追跡する。
 * compute_token_journey.py を呼び出す。
 */
ipcMain.handle(
  "model:computeTokenJourney",
  async (
    _event,
    payload: {
      modelDir: string;
      config: Record<string, unknown>;
      nodes: Array<{ id: string; role: string; layer_index: number | null }>;
      edges: Array<{ source: string; target: string }>;
      inputText?: string;
      useChatTemplate?: boolean;
      inputIds?: number[];
      topK?: number;
    }
  ) => {
    return new Promise((resolve, reject) => {
      const scriptPath = isDev
        ? path.join(__dirname, "../python/compute_token_journey.py")
        : path.join(process.resourcesPath, "python/compute_token_journey.py");

      const pythonBin = defaultPythonBin();
      const child: ChildProcessWithoutNullStreams = spawn(pythonBin, [scriptPath], {
        env: { ...process.env, AIDE_LANG: uiLang, PYTHONIOENCODING: "utf-8" },
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });
      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      child.on("close", (code) => {
        try {
          const parsed = JSON.parse(stdout);
          resolve(parsed);
        } catch (e) {
          reject(
            new Error(
              `compute_token_journey.py failed (code ${code}): ${stderr || stdout || e}`
            )
          );
        }
      });

      child.on("error", (err) => {
        reject(new Error(`Failed to spawn Python process: ${err.message}`));
      });

      const requestBody = JSON.stringify({
        model_dir: payload.modelDir,
        config: payload.config,
        nodes: payload.nodes,
        edges: payload.edges,
        input_text: payload.inputText,
        use_chat_template: payload.useChatTemplate ?? false,
        input_ids: payload.inputText === undefined ? payload.inputIds ?? [1, 2, 3, 4, 5] : undefined,
        top_k: payload.topK ?? 1,
      });
      child.stdin.write(requestBody);
      child.stdin.end();
    });
  }
);

/**
 * Phase2: Multi-Model Battle View — 同じ入力に対する複数モデルの生成結果・
 * 内部状態を横並びで比較する。run_battle_view.py を呼び出す。
 * 他のIPCハンドラと違い、単一のmodelDirではなくmodels配列(2個以上を想定、
 * 可変)を渡す。異なるアーキテクチャのモデル同士も想定するため、モデルごとに
 * 個別のconfigを持たせる(2026-09-10確認事項)。
 * 複数モデルを同時にロードするため、他のIPCハンドラより時間・メモリを要する。
 */
ipcMain.handle(
  "model:battleCompare",
  async (
    _event,
    payload: {
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
    }
  ) => {
    return new Promise((resolve, reject) => {
      const scriptPath = isDev
        ? path.join(__dirname, "../python/run_battle_view.py")
        : path.join(process.resourcesPath, "python/run_battle_view.py");

      const pythonBin = defaultPythonBin();
      const child: ChildProcessWithoutNullStreams = spawn(pythonBin, [scriptPath], {
        env: { ...process.env, AIDE_LANG: uiLang, PYTHONIOENCODING: "utf-8" },
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });
      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      child.on("close", (code) => {
        try {
          const parsed = JSON.parse(stdout);
          resolve(parsed);
        } catch (e) {
          reject(
            new Error(
              `run_battle_view.py failed (code ${code}): ${stderr || stdout || e}`
            )
          );
        }
      });

      child.on("error", (err) => {
        reject(new Error(`Failed to spawn Python process: ${err.message}`));
      });

      const requestBody = JSON.stringify({
        models: payload.models.map((m) => ({
          model_dir: m.modelDir,
          config: m.config,
          label: m.label,
        })),
        input_text: payload.inputText,
        use_chat_template: payload.useChatTemplate ?? false,
        input_ids: payload.inputText === undefined ? payload.inputIds ?? [1, 2, 3, 4, 5] : undefined,
        compare_generation: payload.compareGeneration ?? true,
        compare_internals: payload.compareInternals ?? true,
        max_new_tokens: payload.maxNewTokens ?? 20,
        temperature: payload.temperature ?? 0.0,
        top_k: payload.topK,
        top_p: payload.topP,
        repetition_penalty: payload.repetitionPenalty ?? 1.0,
        no_repeat_ngram_size: payload.noRepeatNgramSize,
        aggregate_heads: payload.aggregateHeads ?? true,
      });
      child.stdin.write(requestBody);
      child.stdin.end();
    });
  }
);

/**
 * Phase2: Neuron Search — MLP中間次元・Attentionヘッド単位のニューロンを探索する。
 * run_neuron_search.py を呼び出す(rank/probeの2モードとも同じスクリプトで、
 * requestBody内の mode フィールドで切り替える)。
 */
ipcMain.handle(
  "model:neuronSearchRank",
  async (
    _event,
    payload: {
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
    }
  ) => {
    return new Promise((resolve, reject) => {
      const scriptPath = isDev
        ? path.join(__dirname, "../python/run_neuron_search.py")
        : path.join(process.resourcesPath, "python/run_neuron_search.py");

      const pythonBin = defaultPythonBin();
      const child: ChildProcessWithoutNullStreams = spawn(pythonBin, [scriptPath], {
        env: { ...process.env, AIDE_LANG: uiLang, PYTHONIOENCODING: "utf-8" },
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });
      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      child.on("close", (code) => {
        try {
          const parsed = JSON.parse(stdout);
          resolve(parsed);
        } catch (e) {
          reject(
            new Error(
              `run_neuron_search.py failed (code ${code}): ${stderr || stdout || e}`
            )
          );
        }
      });

      child.on("error", (err) => {
        reject(new Error(`Failed to spawn Python process: ${err.message}`));
      });

      const requestBody = JSON.stringify({
        mode: "rank",
        model_dir: payload.modelDir,
        config: payload.config,
        nodes: payload.nodes,
        edges: payload.edges,
        input_text: payload.inputText,
        use_chat_template: payload.useChatTemplate ?? false,
        input_ids: payload.inputText === undefined ? payload.inputIds ?? [1, 2, 3, 4, 5] : undefined,
        neuron_kind: payload.neuronKind ?? "both",
        target_layers: payload.targetLayers ?? null,
        top_k: payload.topK ?? 20,
        aggregation: payload.aggregation ?? "max_abs",
      });
      child.stdin.write(requestBody);
      child.stdin.end();
    });
  }
);

ipcMain.handle(
  "model:neuronSearchProbe",
  async (
    _event,
    payload: {
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
    }
  ) => {
    return new Promise((resolve, reject) => {
      const scriptPath = isDev
        ? path.join(__dirname, "../python/run_neuron_search.py")
        : path.join(process.resourcesPath, "python/run_neuron_search.py");

      const pythonBin = defaultPythonBin();
      const child: ChildProcessWithoutNullStreams = spawn(pythonBin, [scriptPath], {
        env: { ...process.env, AIDE_LANG: uiLang, PYTHONIOENCODING: "utf-8" },
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });
      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      child.on("close", (code) => {
        try {
          const parsed = JSON.parse(stdout);
          resolve(parsed);
        } catch (e) {
          reject(
            new Error(
              `run_neuron_search.py failed (code ${code}): ${stderr || stdout || e}`
            )
          );
        }
      });

      child.on("error", (err) => {
        reject(new Error(`Failed to spawn Python process: ${err.message}`));
      });

      const requestBody = JSON.stringify({
        mode: "probe",
        model_dir: payload.modelDir,
        config: payload.config,
        nodes: payload.nodes,
        edges: payload.edges,
        layer_index: payload.layerIndex,
        kind: payload.kind,
        neuron_index: payload.neuronIndex,
        input_texts: payload.inputTexts,
        use_chat_template: payload.useChatTemplate ?? false,
        aggregation: payload.aggregation ?? "max_abs",
      });
      child.stdin.write(requestBody);
      child.stdin.end();
    });
  }
);

/**
 * Phase2: Layer Ablation Wizard — 選択したノードを無効化(出力ゼロ化)した状態と
 * 通常時とを比較する。run_layer_ablation.py を呼び出す。
 * 「全層を自動で1層ずつ無効化してランキングする」バッチ処理ではなく、
 * ユーザーがGUI上で選んだノード(1つ以上)を一度に無効化して1回だけ比較する
 * (2026-09-12確認事項)。
 */
ipcMain.handle(
  "model:runLayerAblation",
  async (
    _event,
    payload: {
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
    }
  ) => {
    return new Promise((resolve, reject) => {
      const scriptPath = isDev
        ? path.join(__dirname, "../python/run_layer_ablation.py")
        : path.join(process.resourcesPath, "python/run_layer_ablation.py");

      const pythonBin = defaultPythonBin();
      const child: ChildProcessWithoutNullStreams = spawn(pythonBin, [scriptPath], {
        env: { ...process.env, AIDE_LANG: uiLang, PYTHONIOENCODING: "utf-8" },
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });
      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      child.on("close", (code) => {
        try {
          const parsed = JSON.parse(stdout);
          resolve(parsed);
        } catch (e) {
          reject(
            new Error(
              `run_layer_ablation.py failed (code ${code}): ${stderr || stdout || e}`
            )
          );
        }
      });

      child.on("error", (err) => {
        reject(new Error(`Failed to spawn Python process: ${err.message}`));
      });

      const requestBody = JSON.stringify({
        model_dir: payload.modelDir,
        config: payload.config,
        nodes: payload.nodes,
        edges: payload.edges,
        input_text: payload.inputText,
        use_chat_template: payload.useChatTemplate ?? false,
        input_ids: payload.inputText === undefined ? payload.inputIds ?? [1, 2, 3, 4, 5] : undefined,
        ablate_node_ids: payload.ablateNodeIds,
        compare_generation: payload.compareGeneration ?? true,
        max_new_tokens: payload.maxNewTokens ?? 20,
        temperature: payload.temperature ?? 0.0,
        top_k: payload.topK,
        top_p: payload.topP,
        repetition_penalty: payload.repetitionPenalty ?? 1.0,
        no_repeat_ngram_size: payload.noRepeatNgramSize,
      });
      child.stdin.write(requestBody);
      child.stdin.end();
    });
  }
);

/**
 * Phase3: Activation Steering Studio — 対照的なプロンプト群(positive/negative)から
 * ステアリングベクトルを計算し、選択したノードに加算して出力を誘導する。
 * run_activation_steering.py を呼び出す。Layer Ablation Wizardと同じ設計パターン
 * (既存の実行エンジンをそのまま使い回し、通常時との比較を1回の実行で返す)。
 * 2026-09-15確認事項: ベクトルは対照プロンプトの差分方式、適用対象はユーザー選択、
 * 効果はKLダイバージェンス(定量)+生成テキスト比較(目視)の両方を返す。
 */
ipcMain.handle(
  "model:activationSteering",
  async (
    _event,
    payload: {
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
    }
  ) => {
    return new Promise((resolve, reject) => {
      const scriptPath = isDev
        ? path.join(__dirname, "../python/run_activation_steering.py")
        : path.join(process.resourcesPath, "python/run_activation_steering.py");

      const pythonBin = defaultPythonBin();
      const child: ChildProcessWithoutNullStreams = spawn(pythonBin, [scriptPath], {
        env: { ...process.env, AIDE_LANG: uiLang, PYTHONIOENCODING: "utf-8" },
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });
      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      child.on("close", (code) => {
        try {
          const parsed = JSON.parse(stdout);
          resolve(parsed);
        } catch (e) {
          reject(
            new Error(
              `run_activation_steering.py failed (code ${code}): ${stderr || stdout || e}`
            )
          );
        }
      });

      child.on("error", (err) => {
        reject(new Error(`Failed to spawn Python process: ${err.message}`));
      });

      const requestBody = JSON.stringify({
        model_dir: payload.modelDir,
        config: payload.config,
        nodes: payload.nodes,
        edges: payload.edges,
        positive_prompts: payload.positivePrompts,
        negative_prompts: payload.negativePrompts,
        target_node_ids: payload.targetNodeIds,
        scale: payload.scale ?? 1.0,
        input_text: payload.inputText,
        use_chat_template: payload.useChatTemplate ?? false,
        input_ids: payload.inputText === undefined ? payload.inputIds ?? [1, 2, 3, 4, 5] : undefined,
        compare_generation: payload.compareGeneration ?? true,
        max_new_tokens: payload.maxNewTokens ?? 20,
        temperature: payload.temperature ?? 0.0,
        top_k: payload.topK,
        top_p: payload.topP,
        repetition_penalty: payload.repetitionPenalty ?? 1.0,
        no_repeat_ngram_size: payload.noRepeatNgramSize,
        measure_sycophancy: payload.measureSycophancy ?? false,
        measure_refusal: payload.measureRefusal ?? false,
        intervention_mode: payload.interventionMode ?? "add",
      });
      child.stdin.write(requestBody);
      child.stdin.end();
    });
  }
);

/**
 * Live Slider(Phase3、2026-09-24追加): スライダーによるリアルタイム出力操作。
 *
 * 他の機能(1回のリクエスト→1回の応答)と違い、python/run_live_session.py を常駐させ、
 * 1行1JSONのコマンドをstdinに送り、1行1JSONのイベントをstdoutから受け取って、
 * そのままレンダラーへ "live:event" として流す。生成中のスライダー操作(set_controls)も
 * 同じstdinに送るので、Python側で次のトークンから反映される。
 * 同時に動かすセッションは1つだけ(新しく開始すると前のセッションは終了させる)。
 */
let liveChild: ChildProcessWithoutNullStreams | null = null;

function sendLiveEvent(event: unknown) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("live:event", event);
  }
}

function stopLiveChild() {
  if (!liveChild) return;
  const child = liveChild;
  liveChild = null;
  try {
    child.stdin.write(JSON.stringify({ cmd: "shutdown" }) + "\n");
    child.stdin.end();
  } catch {
    // 既に終了している場合は無視
  }
  setTimeout(() => {
    if (child.exitCode === null) child.kill();
  }, 3000);
}

ipcMain.handle(
  "live:start",
  async (
    _event,
    payload: {
      modelDir: string;
      config: Record<string, unknown>;
      nodes: Array<{ id: string; role: string; layer_index: number | null }>;
      edges: Array<{ source: string; target: string }>;
      useChatTemplate?: boolean;
    }
  ) => {
    stopLiveChild();
    const scriptPath = isDev
      ? path.join(__dirname, "../python/run_live_session.py")
      : path.join(process.resourcesPath, "python/run_live_session.py");
    const pythonBin = defaultPythonBin();
    const child: ChildProcessWithoutNullStreams = spawn(pythonBin, [scriptPath], {
      env: { ...process.env, AIDE_LANG: uiLang, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
    });
    liveChild = child;

    let buffer = "";
    let stderrTail = "";
    child.stdout.on("data", (data) => {
      buffer += data.toString();
      let newline = buffer.indexOf("\n");
      while (newline >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line) {
          try {
            sendLiveEvent(JSON.parse(line));
          } catch {
            sendLiveEvent({ event: "error", error_type: "InvalidOutput", message: line });
          }
        }
        newline = buffer.indexOf("\n");
      }
    });
    child.stderr.on("data", (data) => {
      stderrTail = (stderrTail + data.toString()).slice(-4000);
    });
    child.on("close", (code) => {
      if (liveChild === child) liveChild = null;
      sendLiveEvent({ event: "exited", code, stderr: stderrTail });
    });
    child.on("error", (err) => {
      if (liveChild === child) liveChild = null;
      sendLiveEvent({ event: "error", error_type: "SpawnError", message: `Failed to spawn Python process: ${err.message}` });
    });

    child.stdin.write(
      JSON.stringify({
        cmd: "init",
        model_dir: payload.modelDir,
        config: payload.config,
        nodes: payload.nodes,
        edges: payload.edges,
        use_chat_template: payload.useChatTemplate ?? false,
      }) + "\n"
    );
    return { started: true };
  }
);

ipcMain.handle("live:send", async (_event, command: Record<string, unknown>) => {
  if (!liveChild) return { sent: false, message: uiLang === "en" ? "No session is running" : "セッションが開始されていません" };
  liveChild.stdin.write(JSON.stringify(command) + "\n");
  return { sent: true };
});

ipcMain.handle("live:stop", async () => {
  stopLiveChild();
  return { stopped: true };
});

app.on("before-quit", () => stopLiveChild());
