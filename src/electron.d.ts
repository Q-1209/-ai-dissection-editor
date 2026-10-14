export interface ModelInspectResult {
  model_type: string;
  num_layers: number;
  hidden_size: number | null;
  nodes: ModelNode[];
  edges: ModelEdge[];
  errors: string[];
  raw_config: Record<string, unknown>; // config.jsonの中身をそのまま保持(実行時に使用)
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

// Phase1: 実行用グラフのノード役割。可視化(Phase0)の細かい役割(attention/mlp/norm等)とは
// 異なり、ブロック単位(Attention一式・MLP一式)で扱う。
export type ExecutionRole =
  | "attention_block"
  | "mlp_block"
  | "layer_block"
  | "norm"
  | "embedding"
  | "lm_head";

export interface ExecutionNode {
  id: string;
  role: ExecutionRole;
  layer_index: number | null;
  // このノードが複数の入力(合流)を受け取る場合の組み合わせ方。
  // "add": 要素ごとの加算。"average": 要素ごとの平均。入力が1つの場合は無視される。
  merge_mode?: "add" | "average";
}

export interface ExecutionEdge {
  source: string;
  target: string;
}

export interface ExecutionRequest {
  modelDir: string;
  config: Record<string, unknown>;
  nodes: ExecutionNode[];
  edges: ExecutionEdge[];
  inputText?: string;
  // 2026-09-22追加(全タブ共通): trueなら入力文をモデルのチャットテンプレートで包んでからエンコードする
  useChatTemplate?: boolean;
  inputIds?: number[];
  maxNewTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  noRepeatNgramSize?: number;
}

export interface ExecutionSuccessResult {
  success: true;
  input_token_ids: number[];
  generated_token_ids: number[];
  generated_text: string | null; // input_textで実行した場合のみ非null
  executed_order: string[];
}

export interface ExecutionErrorResult {
  success: false;
  error_type: "GraphValidationError" | "UnexpectedError" | "InvalidInput" | "TokenizerLoadError";
  message: string;
}

export type ExecutionResult = ExecutionSuccessResult | ExecutionErrorResult;

// Phase2: 内部状態可視化(Attention重み・活性化値)関連の型
export interface InternalsRequest {
  modelDir: string;
  config: Record<string, unknown>;
  nodes: ExecutionNode[];
  edges: ExecutionEdge[];
  inputText?: string;
  // 2026-09-22追加(全タブ共通): trueなら入力文をモデルのチャットテンプレートで包んでからエンコードする
  useChatTemplate?: boolean;
  inputIds?: number[];
  aggregateHeads?: boolean; // true: ヘッド平均、false: 全ヘッド分。省略時true
}

export interface AttentionWeightsEntry {
  shape: number[]; // aggregateHeads=trueなら[seq_len, seq_len]、falseなら[num_heads, seq_len, seq_len]
  data: number[][] | number[][][]; // shapeに対応したネスト配列
}

export interface ActivationStatsEntry {
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface InternalsSuccessResult {
  success: true;
  input_token_ids: number[];
  input_tokens_text: string[] | null; // input_textで実行した場合のみ非null
  executed_order: string[];
  attention_weights: Record<string, AttentionWeightsEntry>; // キーはノードID
  activation_stats: Record<string, ActivationStatsEntry>; // キーはノードID
}

export interface InternalsErrorResult {
  success: false;
  error_type: "GraphValidationError" | "UnexpectedError" | "InvalidInput" | "TokenizerLoadError";
  message: string;
}

export type InternalsResult = InternalsSuccessResult | InternalsErrorResult;

// Phase2: Prompt Sensitivity Map関連の型
export type ReplacementMode = "delete" | "pad" | "random" | "custom";

export interface SensitivityRequest {
  modelDir: string;
  config: Record<string, unknown>;
  nodes: ExecutionNode[];
  edges: ExecutionEdge[];
  inputText?: string;
  // 2026-09-22追加(全タブ共通): trueなら入力文をモデルのチャットテンプレートで包んでからエンコードする
  useChatTemplate?: boolean;
  inputIds?: number[];
  replacementMode?: ReplacementMode; // 省略時"pad"
  customTokenId?: number; // replacementMode="custom"のとき必須
  padTokenId?: number; // 省略時0
}

export interface SensitivityTokenResult {
  position: number;
  kl_divergence: number;
  top_token_changed: boolean;
  token_text?: string; // input_textで実行した場合のみ
  original_top_token_text?: string;
  modified_top_token_text?: string;
}

export interface SensitivitySuccessResult {
  success: true;
  input_token_ids: number[];
  input_tokens_text: string[] | null;
  replacement_mode: ReplacementMode;
  results: SensitivityTokenResult[];
}

export interface SensitivityErrorResult {
  success: false;
  error_type: "GraphValidationError" | "UnexpectedError" | "InvalidInput" | "TokenizerLoadError";
  message: string;
}

export type SensitivityResult = SensitivitySuccessResult | SensitivityErrorResult;

// Phase2: Token Journey関連の型
export interface TokenJourneyRequest {
  modelDir: string;
  config: Record<string, unknown>;
  nodes: ExecutionNode[];
  edges: ExecutionEdge[];
  inputText?: string;
  // 2026-09-22追加(全タブ共通): trueなら入力文をモデルのチャットテンプレートで包んでからエンコードする
  useChatTemplate?: boolean;
  inputIds?: number[];
  topK?: number; // 各(トークン位置, ブロック)ごとに何件の近傍単語を返すか。省略時1
}

export interface NearestWordMatch {
  token_id: number;
  similarity: number;
  token_text?: string; // input_textで実行した場合のみ
}

export interface TokenJourneySuccessResult {
  success: true;
  input_token_ids: number[];
  input_tokens_text: string[] | null;
  block_order: string[]; // ブロックの実行順序(意味変遷を追う順番)
  // journey[blockId][tokenPosition] = そのブロック・そのトークン位置での上位top_k件
  journey: Record<string, NearestWordMatch[][]>;
}

export interface TokenJourneyErrorResult {
  success: false;
  error_type: "GraphValidationError" | "UnexpectedError" | "InvalidInput" | "TokenizerLoadError";
  message: string;
}

export type TokenJourneyResult = TokenJourneySuccessResult | TokenJourneyErrorResult;

// Phase2: Multi-Model Battle View関連の型
// 2026-09-10確認事項: 比較対象は生成結果・内部状態の両方、モデル数は可変(N個)、
// 複数モデルの同時ロードは許容、異なるアーキテクチャ間の比較も想定する。
// そのため各モデルは自分自身のconfigを個別に持ち(hidden_size等が異なってもよい)、
// モデル間でテンソルを混ぜ合わせる処理は行わない(結果はモデルごとに独立)。
export interface BattleModelSpec {
  modelDir: string;
  config: Record<string, unknown>;
  label?: string; // 省略時はmodelDirのフォルダ名がラベルになる(Python側で決定)
}

export interface BattleRequest {
  models: BattleModelSpec[];
  inputText?: string;
  // 2026-09-22追加(全タブ共通): trueなら入力文をモデルのチャットテンプレートで包んでからエンコードする
  useChatTemplate?: boolean;
  inputIds?: number[];
  compareGeneration?: boolean; // 省略時true
  compareInternals?: boolean; // 省略時true
  maxNewTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  noRepeatNgramSize?: number;
  aggregateHeads?: boolean;
}

export interface BattleModelSuccessResult {
  label: string;
  model_dir: string;
  success: true;
  input_token_ids: number[];
  input_tokens_text: string[] | null;
  generated_token_ids: number[] | null; // compareGeneration=falseならnull
  generated_text: string | null;
  executed_order: string[];
  // モデルごとに独立(異なるhidden_size同士でも比較軸を揃える処理はしない、GUI側で
  // 個別カラム表示して目視比較する)。compareInternals=falseならnull
  activation_stats: Record<string, ActivationStatsEntry> | null;
  attention_weights: Record<string, AttentionWeightsEntry> | null;
}

export interface BattleModelErrorResult {
  label: string;
  model_dir: string;
  success: false;
  error_type: "GraphValidationError" | "UnexpectedError" | "InvalidInput" | "TokenizerLoadError";
  message: string;
}

// 1つのモデルの失敗は他のモデルの結果に影響しない(モデルごとに隔離、run_battle_view.py参照)
export type BattleModelResult = BattleModelSuccessResult | BattleModelErrorResult;

export interface BattleSuccessResult {
  success: true;
  results: BattleModelResult[];
}

export interface BattleErrorResult {
  success: false;
  error_type: "UnexpectedError" | "InvalidInput";
  message: string;
}

export type BattleResult = BattleSuccessResult | BattleErrorResult;

// Phase2: Neuron Search関連の型
// 2026-09-12確認事項: 「ニューロン」はMLP中間次元・Attentionヘッドの両方に対応、
// 探索方向は(a)入力に対するランキング(rank)、(b)特定ニューロンの複数入力への
// 反応の比較(probe)の両方に対応する。既存の観察タブ「内部状態」(層単位の統計)を
// ニューロン単位まで細分化したものという位置づけ。
export type NeuronKind = "mlp" | "attention_head" | "both";
export type NeuronAggregation = "max_abs" | "mean_abs";

export interface NeuronSearchRankRequest {
  modelDir: string;
  config: Record<string, unknown>;
  nodes: ExecutionNode[];
  edges: ExecutionEdge[];
  inputText?: string;
  // 2026-09-22追加(全タブ共通): trueなら入力文をモデルのチャットテンプレートで包んでからエンコードする
  useChatTemplate?: boolean;
  inputIds?: number[];
  neuronKind?: NeuronKind; // 省略時"both"
  targetLayers?: number[]; // 省略時(undefined)は全層が対象
  topK?: number; // 省略時20
  aggregation?: NeuronAggregation; // 省略時"max_abs"
}

export interface RankedNeuronEntry {
  layer_index: number;
  kind: "mlp" | "attention_head";
  neuron_index: number;
  score: number;
}

export interface NeuronSearchRankSuccessResult {
  success: true;
  mode: "rank";
  input_token_ids: number[];
  input_tokens_text: string[] | null;
  ranked_neurons: RankedNeuronEntry[];
}

export interface NeuronSearchProbeRequest {
  modelDir: string;
  config: Record<string, unknown>;
  nodes: ExecutionNode[];
  edges: ExecutionEdge[];
  layerIndex: number;
  kind: "mlp" | "attention_head";
  neuronIndex: number;
  inputTexts: string[]; // 1つ以上
  useChatTemplate?: boolean; // 2026-09-22追加(全タブ共通)
  aggregation?: NeuronAggregation; // 省略時"max_abs"
}

export interface NeuronProbeResultEntry {
  input_text: string;
  score: number;
}

export interface NeuronSearchProbeSuccessResult {
  success: true;
  mode: "probe";
  layer_index: number;
  kind: "mlp" | "attention_head";
  neuron_index: number;
  results: NeuronProbeResultEntry[];
}

export interface NeuronSearchErrorResult {
  success: false;
  error_type: "GraphValidationError" | "UnexpectedError" | "InvalidInput" | "TokenizerLoadError";
  message: string;
}

export type NeuronSearchResult =
  | NeuronSearchRankSuccessResult
  | NeuronSearchProbeSuccessResult
  | NeuronSearchErrorResult;

// Phase2: Layer Ablation Wizard関連の型
// 2026-09-12確認事項: 「無効化」=ノードの出力をゼロにする(残差接続も含めて寄与を
// 完全に断つ)。全層自動バッチ処理ではなく、ユーザーがGUI上で選んだ1つ以上の
// ノードを無効化した状態と通常時とを1回の実行で比較する、対話的なツール。
// 測定指標はKLダイバージェンスに加えて、生成テキストの変化も含める。
export interface LayerAblationRequest {
  modelDir: string;
  config: Record<string, unknown>;
  nodes: ExecutionNode[];
  edges: ExecutionEdge[];
  inputText?: string;
  // 2026-09-22追加(全タブ共通): trueなら入力文をモデルのチャットテンプレートで包んでからエンコードする
  useChatTemplate?: boolean;
  inputIds?: number[];
  ablateNodeIds: string[]; // 無効化するノードID(1つ以上)
  compareGeneration?: boolean; // 省略時true
  maxNewTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  noRepeatNgramSize?: number;
}

export interface LayerAblationSuccessResult {
  success: true;
  input_token_ids: number[];
  input_tokens_text: string[] | null;
  ablated_node_ids: string[];
  kl_divergence: number; // 最後の位置の次トークン分布、通常時→無効化時のKL
  top_token_changed: boolean;
  baseline_top_token_id: number;
  ablated_top_token_id: number;
  baseline_top_token_text: string | null;
  ablated_top_token_text: string | null;
  generated_text_baseline: string | null; // compareGeneration=falseならnull
  generated_text_ablated: string | null;
  executed_order: string[];
}

export interface LayerAblationErrorResult {
  success: false;
  error_type: "GraphValidationError" | "UnexpectedError" | "InvalidInput" | "TokenizerLoadError";
  message: string;
}

export type LayerAblationResult = LayerAblationSuccessResult | LayerAblationErrorResult;

// Phase3: Activation Steering Studio関連の型
// 2026-09-15確認事項: ステアリングベクトルは対照プロンプト群(positive/negative)の
// 平均活性化値の差分方式。適用対象はユーザーがGUI上で選んだ1つ以上のノード
// (Layer Ablation Wizardと同じチェックボックス選択UX)。効果はKLダイバージェンス
// (定量指標)と生成テキストの変化(目視)の両方で確認する。
export interface ActivationSteeringRequest {
  modelDir: string;
  config: Record<string, unknown>;
  nodes: ExecutionNode[];
  edges: ExecutionEdge[];
  positivePrompts: string[]; // 対照プロンプト群(ステアリングしたい方向)
  negativePrompts: string[]; // 対照プロンプト群(逆方向)
  targetNodeIds: string[]; // ベクトルを計算・適用する対象ノード(1つ以上)
  scale?: number; // ステアリングの強さ(符号反転で逆方向、0で無効化と同等)。省略時1.0
  inputText?: string;
  // 2026-09-22追加(全タブ共通): trueなら入力文をモデルのチャットテンプレートで包んでからエンコードする
  useChatTemplate?: boolean;
  inputIds?: number[];
  compareGeneration?: boolean; // 省略時true
  maxNewTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  noRepeatNgramSize?: number;
  // Sycophancy Reducer(2026-09-15追加、Activation Steering Studioの拡張)用。
  // trueかつcompareGeneration=trueの場合、生成テキスト中の迎合的定型語彙の出現回数を
  // 機械的指標として追加で返す(省略時false、Activation Steering Studio本体には影響しない)
  measureSycophancy?: boolean;
  // Refusal Behavior Tuner(2026-09-22追加)用。trueなら拒否定型句の出現回数も返す(省略時false)
  measureRefusal?: boolean;
  // 2026-09-22追加。"add"=scale倍して加算(従来どおり)、"ablate_direction"=差分ベクトルの方向の
  // 成分を射影で除去(scaleは使わない)。省略時"add"
  interventionMode?: InterventionMode;
}

export type InterventionMode = "add" | "ablate_direction";

// 迎合的な定型語彙の出現回数(Sycophancy Reducer、2026-09-15追加)。
// total: 合計出現回数、by_phrase: 実際に出現した語彙ごとの回数(0回のものは含まない)
export interface SycophancyPhraseCount {
  total: number;
  by_phrase: Record<string, number>;
}

export interface ActivationSteeringSuccessResult {
  success: true;
  target_node_ids: string[];
  scale: number;
  vector_norms: Record<string, number>; // 各対象ノードの差分ベクトルのL2ノルム(GUI表示用)
  warnings?: string[]; // 2026-10-14: 1トークンのプロンプトなど、結果が信用しにくい条件への警告
  input_token_ids: number[];
  input_tokens_text: string[] | null;
  kl_divergence: number; // 最後の位置の次トークン分布、通常時→ステアリング適用時のKL
  top_token_changed: boolean;
  baseline_top_token_id: number;
  steered_top_token_id: number;
  baseline_top_token_text: string | null;
  steered_top_token_text: string | null;
  generated_text_baseline: string | null; // compareGeneration=falseならnull
  generated_text_steered: string | null;
  // measureSycophancy=trueかつ生成テキストが得られた場合のみ値が入る(それ以外はnull)
  sycophancy_phrase_count_baseline: SycophancyPhraseCount | null;
  sycophancy_phrase_count_steered: SycophancyPhraseCount | null;
  // measureRefusal=trueかつ生成テキストが得られた場合のみ値が入る(形式はSycophancyPhraseCountと同じ)
  refusal_phrase_count_baseline: SycophancyPhraseCount | null;
  refusal_phrase_count_steered: SycophancyPhraseCount | null;
  intervention_mode: InterventionMode;
  used_chat_template: boolean;
  executed_order: string[];
}

export interface ActivationSteeringErrorResult {
  success: false;
  error_type: "GraphValidationError" | "UnexpectedError" | "InvalidInput" | "TokenizerLoadError";
  message: string;
}

export type ActivationSteeringResult = ActivationSteeringSuccessResult | ActivationSteeringErrorResult;

// ---------------------------------------------------------------------------
// Live Slider(Phase3、2026-09-24追加): スライダーによるリアルタイム出力操作
// python/run_live_session.py(常駐プロセス)と1行1JSONでやり取りする。形式は同ファイル冒頭参照
// ---------------------------------------------------------------------------

export type LiveNeuronKind = "mlp" | "attention_head";

export interface LiveNeuronControl {
  layer: number;
  kind: LiveNeuronKind;
  index: number; // MLPの中間次元番号、またはヘッド番号
  scale: number;
}

export interface LiveControls {
  node_scales: Record<string, number>; // ブロックの寄与(残差への足し分)の倍率。1=通常、0=素通り
  neurons: LiveNeuronControl[];
  steering_scale: number; // ステアリングベクトルを何倍で足すか(0=なし)
}

export type LiveSweepTarget =
  | { type: "node"; id: string }
  | { type: "neuron"; layer: number; kind: LiveNeuronKind; index: number }
  | { type: "steering" };

export interface LiveStartRequest {
  modelDir: string;
  config: Record<string, unknown>;
  nodes: ExecutionNode[];
  edges: ExecutionEdge[];
  useChatTemplate?: boolean;
}

export type LiveCommand =
  | { cmd: "set_controls"; controls: LiveControls }
  | { cmd: "set_steering"; positive_prompts: string[]; negative_prompts: string[]; target_node_ids: string[] }
  | {
      cmd: "generate";
      input_text: string;
      max_new_tokens: number;
      compare_baseline: boolean;
      temperature?: number;
      top_k?: number | null;
      top_p?: number | null;
      repetition_penalty?: number;
      no_repeat_ngram_size?: number | null;
    }
  | { cmd: "stop" }
  | { cmd: "sweep"; input_text: string; target: LiveSweepTarget; start: number; stop: number; steps: number }
  | { cmd: "shutdown" };

export interface LiveSweepPoint {
  scale: number;
  kl: number;
  top_token_text: string;
  top_prob: number;
  baseline_top_prob: number; // 通常時の最有力トークンが、この倍率でどれだけの確率を保っているか
}

export type LiveEvent =
  | {
      event: "ready";
      limits: { node_ids: string[]; num_layers: number; intermediate_size: number; num_heads: number };
      use_chat_template: boolean;
    }
  | { event: "steering_ready"; vector_norms: Record<string, number>; warnings?: string[] }
  | { event: "controls_applied"; version: number; controls: LiveControls }
  | { event: "started"; input_token_count: number }
  | { event: "baseline"; text: string; token_count: number }
  | {
      event: "token";
      step: number;
      token_id: number;
      text: string; // ここまでの全文(操作後)
      kl: number; // このトークンの予測分布の、通常時→操作後のKL
      top_token_changed: boolean;
      baseline_top_token_text: string;
      controls_version: number;
    }
  | { event: "done"; text: string; token_count: number; reason: "max_new_tokens" | "end_of_turn" | "stopped"; mean_kl: number; max_kl: number }
  | {
      event: "sweep_result";
      target: LiveSweepTarget;
      points: LiveSweepPoint[];
      baseline_top_token_text: string;
      baseline_top_prob: number;
    }
  | { event: "error"; error_type: string; message: string; cmd?: string | null }
  | { event: "exited"; code: number | null; stderr: string };

export interface ElectronAPI {
  setLang?: (lang: "ja" | "en") => void;
  selectModelDir: () => Promise<string | null>;
  inspectModel: (modelDir: string) => Promise<ModelInspectResult>;
  executeModel: (payload: ExecutionRequest) => Promise<ExecutionResult>;
  inspectInternals: (payload: InternalsRequest) => Promise<InternalsResult>;
  analyzeSensitivity: (payload: SensitivityRequest) => Promise<SensitivityResult>;
  computeTokenJourney: (payload: TokenJourneyRequest) => Promise<TokenJourneyResult>;
  battleCompare: (payload: BattleRequest) => Promise<BattleResult>;
  neuronSearchRank: (payload: NeuronSearchRankRequest) => Promise<NeuronSearchResult>;
  neuronSearchProbe: (payload: NeuronSearchProbeRequest) => Promise<NeuronSearchResult>;
  runLayerAblation: (payload: LayerAblationRequest) => Promise<LayerAblationResult>;
  activationSteering: (payload: ActivationSteeringRequest) => Promise<ActivationSteeringResult>;
  liveStart: (payload: LiveStartRequest) => Promise<{ started: boolean }>;
  liveSend: (command: LiveCommand) => Promise<{ sent: boolean; message?: string }>;
  liveStop: () => Promise<{ stopped: boolean }>;
  onLiveEvent: (callback: (event: LiveEvent) => void) => () => void;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}
