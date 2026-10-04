// App.tsxの状態型(2026-09-22、App.tsx分割の第1段階で切り出し)。
import type {
  ModelInspectResult,
  ExecutionResult,
  InternalsResult,
  SensitivityResult,
  TokenJourneyResult,
  BattleResult,
  NeuronSearchResult,
  LayerAblationResult,
  ActivationSteeringResult,
} from "../electron.d";

export type LoadState =
  | { status: "idle" }
  | { status: "loading"; modelDir: string }
  | { status: "error"; message: string }
  | { status: "loaded"; modelDir: string; summary: ModelInspectResult };

export type Tab = "dissect" | "execute" | "observe";

export type RunState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "done"; result: ExecutionResult };

export type InternalsRunState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "done"; result: InternalsResult };

export type SensitivityRunState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "done"; result: SensitivityResult };

export type TokenJourneyRunState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "done"; result: TokenJourneyResult };

export type BattleRunState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "done"; result: BattleResult };

export type NeuronSearchRunState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "done"; result: NeuronSearchResult };

export type LayerAblationRunState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "done"; result: LayerAblationResult };

export type ActivationSteeringRunState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "done"; result: ActivationSteeringResult };

// Battle Viewに追加した比較対象モデル1件ぶんの情報。
// inspectModel()で取得したconfigをそのまま保持しておき、実行時にPython側へ渡す
// (2026-09-10確認事項: 異なるアーキテクチャのモデル同士も比較対象になりうるため、
// モデルごとに個別のconfigを持たせる)。
export interface BattleModelEntry {
  modelDir: string;
  label: string;
  config: Record<string, unknown>;
}
