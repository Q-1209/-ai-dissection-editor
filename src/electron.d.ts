export interface ModelInspectResult {
  model_type: string;
  num_layers: number;
  hidden_size: number | null;
  nodes: ModelNode[];
  edges: ModelEdge[];
  errors: string[];
}

export interface ModelNode {
  id: string;
  label: string;
  role: string; // 例: "attention", "mlp", "norm", "embedding", "lm_head"
  layer_index: number | null;
  shape: number[] | null;
  param_key: string; // 元のstate_dictキー
}

export interface ModelEdge {
  id: string;
  source: string;
  target: string;
}

export interface ElectronAPI {
  selectModelDir: () => Promise<string | null>;
  inspectModel: (modelDir: string) => Promise<ModelInspectResult>;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}
