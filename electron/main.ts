import { app, BrowserWindow, ipcMain, dialog } from "electron";
import path from "node:path";
import { spawn, ChildProcessWithoutNullStreams } from "node:child_process";

// __dirname はビルド後 dist-electron/ 配下に配置される想定
const isDev = !app.isPackaged;

let mainWindow: BrowserWindow | null = null;

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

    const pythonBin = process.env.PYTHON_BIN || "python3";
    const child: ChildProcessWithoutNullStreams = spawn(pythonBin, [
      scriptPath,
      modelDir,
    ], {
      env: { ...process.env, PYTHONIOENCODING: "utf-8" },
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
