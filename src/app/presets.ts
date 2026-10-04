// Activation Steering系プリセット(Sycophancy Reducer・Refusal Behavior Tuner)。
// 2026-09-22: 文例の実体は python/presets/steering_presets.json に一本化した(以前はApp.tsxと
// Python側に同じ文例を二重に書いていた)。文例を変えるときはJSONだけを編集する。
import presets from "../../python/presets/steering_presets.json";

// Sycophancy Reducer: positive=迎合的な応答例、negative=中立的・事実重視の応答例。
// 迎合を減らすには負のscaleで適用する(2026-09-15確認事項)。
export const SYCOPHANCY_POSITIVE_PROMPTS: string[] = presets.sycophancy.positive_prompts;
export const SYCOPHANCY_NEGATIVE_PROMPTS: string[] = presets.sycophancy.negative_prompts;

// Refusal Behavior Tuner: positive=拒否を引き出しやすい依頼文、negative=1対1で対応する無害な依頼文。
// 差分(positive - negative)が「拒否方向」になる(2026-09-22確認事項)。
export const REFUSAL_POSITIVE_PROMPTS: string[] = presets.refusal.positive_prompts;
export const REFUSAL_NEGATIVE_PROMPTS: string[] = presets.refusal.negative_prompts;
