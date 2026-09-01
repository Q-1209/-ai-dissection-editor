import { contextBridge, ipcRenderer } from "electron";

/**
 * レンダラー(React側)に公開するAPI。
 * nodeIntegration: false のため、この橋渡し経由でのみ
 * メインプロセスの機能(ファイルダイアログ、Python起動)にアクセスできる。
 */
contextBridge.exposeInMainWorld("electronAPI", {
  selectModelDir: (): Promise<string | null> =>
    ipcRenderer.invoke("dialog:selectModelDir"),

  inspectModel: (modelDir: string): Promise<unknown> =>
    ipcRenderer.invoke("model:inspect", modelDir),
});
