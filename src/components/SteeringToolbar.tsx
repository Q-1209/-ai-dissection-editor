import type { Dispatch, SetStateAction } from "react";
import type { InterventionMode } from "../electron.d";
import type { ActivationSteeringRunState } from "../app/types";
import { buttonStyle } from "../app/styles";
import { useLang } from "../app/i18n";

// 観察タブ→Activation Steeringサブタブのツールバー(2026-09-22、App.tsx分割の第1段階で切り出し)。
// 状態はApp.tsx側に置いたまま(実行ハンドラ・他タブとの共有があるため)、表示だけをここに分けている。
// Sycophancy Reducer・Refusal Behavior Tunerのプリセットボタン、介入方式(加算/方向除去)、
// 計測チェックボックスもこのツールバーに含まれる。
export interface SteeringToolbarProps {
  applyRefusalTunerPreset: () => void;
  applySycophancyReducerPreset: () => void;
  handleRunSteering: () => void;
  parsedSteeringNegativePrompts: string[];
  parsedSteeringPositivePrompts: string[];
  steeringCompareGeneration: boolean;
  steeringInputText: string;
  steeringInterventionMode: InterventionMode;
  steeringMaxNewTokens: number;
  steeringMeasureRefusal: boolean;
  steeringMeasureSycophancy: boolean;
  steeringNegativePromptsRaw: string;
  steeringPositivePromptsRaw: string;
  steeringRunState: ActivationSteeringRunState;
  steeringScale: number;
  steeringTargetNodeIds: Set<string>;
  setSteeringCompareGeneration: Dispatch<SetStateAction<boolean>>;
  setSteeringInputText: Dispatch<SetStateAction<string>>;
  setSteeringInterventionMode: Dispatch<SetStateAction<InterventionMode>>;
  setSteeringMaxNewTokens: Dispatch<SetStateAction<number>>;
  setSteeringMeasureRefusal: Dispatch<SetStateAction<boolean>>;
  setSteeringMeasureSycophancy: Dispatch<SetStateAction<boolean>>;
  setSteeringNegativePromptsRaw: Dispatch<SetStateAction<string>>;
  setSteeringPositivePromptsRaw: Dispatch<SetStateAction<string>>;
  setSteeringScale: Dispatch<SetStateAction<number>>;
}

export function SteeringToolbar(props: SteeringToolbarProps) {
  const { t } = useLang();
  const {
    applyRefusalTunerPreset,
    applySycophancyReducerPreset,
    handleRunSteering,
    parsedSteeringNegativePrompts,
    parsedSteeringPositivePrompts,
    steeringCompareGeneration,
    steeringInputText,
    steeringInterventionMode,
    steeringMaxNewTokens,
    steeringMeasureRefusal,
    steeringMeasureSycophancy,
    steeringNegativePromptsRaw,
    steeringPositivePromptsRaw,
    steeringRunState,
    steeringScale,
    steeringTargetNodeIds,
    setSteeringCompareGeneration,
    setSteeringInputText,
    setSteeringInterventionMode,
    setSteeringMaxNewTokens,
    setSteeringMeasureRefusal,
    setSteeringMeasureSycophancy,
    setSteeringNegativePromptsRaw,
    setSteeringPositivePromptsRaw,
    setSteeringScale,
  } = props;
  return (
    <>
        <button
          onClick={applySycophancyReducerPreset}
          style={buttonStyle}
          title={t("対照プロンプト・scale・迎合語彙カウントをSycophancy Reducer用に一括設定", "Fill in contrastive prompts, scale and the agreeing-phrase count for the Sycophancy Reducer")}
        >
          {t("Sycophancy Reducerプリセット", "Sycophancy preset")}
        </button>
        <button
          onClick={applyRefusalTunerPreset}
          style={buttonStyle}
          title={t("対照プロンプト・介入方式(方向除去)・拒否定型句カウント・チャットテンプレート適用をRefusal Behavior Tuner用に一括設定", "Fill in contrastive prompts, directional ablation, the refusal-phrase count and the chat template for the Refusal Tuner")}
        >
          {t("Refusal Tunerプリセット", "Refusal preset")}
        </button>
        <select
          value={steeringInterventionMode}
          onChange={(e) => setSteeringInterventionMode(e.target.value as InterventionMode)}
          title={t("介入方式: 加算=scale倍したベクトルを足す(符号で方向を反転)、方向除去=その方向の成分を射影で取り除く(scaleは使わない)", "Intervention: Add = add the vector times scale (negative flips it). Ablate = project out that direction (scale unused).")}
          style={{ ...buttonStyle, background: "var(--bg-app)" }}
        >
          <option value="add">{t("加算", "Add")}</option>
          <option value="ablate_direction">{t("方向除去", "Ablate direction")}</option>
        </select>
        <textarea
          value={steeringPositivePromptsRaw}
          onChange={(e) => setSteeringPositivePromptsRaw(e.target.value)}
          placeholder={t("positiveプロンプト(1行に1つ)", "positive prompts (one per line)")}
          rows={1}
          style={{
            ...buttonStyle,
            cursor: "text",
            width: 180,
            background: "var(--bg-app)",
            resize: "vertical",
            fontFamily: "inherit",
          }}
        />
        <textarea
          value={steeringNegativePromptsRaw}
          onChange={(e) => setSteeringNegativePromptsRaw(e.target.value)}
          placeholder={t("negativeプロンプト(1行に1つ)", "negative prompts (one per line)")}
          rows={1}
          style={{
            ...buttonStyle,
            cursor: "text",
            width: 180,
            background: "var(--bg-app)",
            resize: "vertical",
            fontFamily: "inherit",
          }}
        />
        <input
          type="number"
          value={steeringScale}
          onChange={(e) => setSteeringScale(Number(e.target.value))}
          step={0.5}
          disabled={steeringInterventionMode === "ablate_direction"}
          title={
            steeringInterventionMode === "ablate_direction"
              ? t("方向除去モードではscaleは使いません", "Scale is not used when ablating a direction")
              : t("ステアリングの強さ(符号反転で逆方向)", "Steering strength (negative = opposite direction)")
          }
          style={{
            ...buttonStyle,
            cursor: "text",
            width: 70,
            background: "var(--bg-app)",
            opacity: steeringInterventionMode === "ablate_direction" ? 0.4 : 1,
          }}
        />
        <input
          type="text"
          value={steeringInputText}
          onChange={(e) => setSteeringInputText(e.target.value)}
          placeholder={t("入力テキスト", "Input text")}
          style={{ ...buttonStyle, cursor: "text", width: 140, background: "var(--bg-app)" }}
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
            checked={steeringCompareGeneration}
            onChange={(e) => setSteeringCompareGeneration(e.target.checked)}
          />
          {t("生成テキストも比較", "Also compare generated text")}
        </label>
        {steeringCompareGeneration && (
          <input
            type="number"
            value={steeringMaxNewTokens}
            onChange={(e) => setSteeringMaxNewTokens(Number(e.target.value))}
            min={1}
            title={t("生成トークン数", "Tokens to generate")}
            style={{ ...buttonStyle, cursor: "text", width: 60, background: "var(--bg-app)" }}
          />
        )}
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 4,
            fontSize: 11,
            color: "var(--text-secondary)",
            cursor: "pointer",
          }}
          title={t("生成テキスト中の迎合的定型語彙の出現回数を機械的指標として表示する(Sycophancy Reducer)", "Count agreeing stock phrases in the generated text as a simple metric (Sycophancy Reducer)")}
        >
          <input
            type="checkbox"
            checked={steeringMeasureSycophancy}
            onChange={(e) => setSteeringMeasureSycophancy(e.target.checked)}
          />
          {t("迎合語彙カウントも計測", "Count agreeing phrases")}
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
          title={t("生成テキスト中の拒否定型句の出現回数を機械的指標として表示する(Refusal Behavior Tuner)", "Count refusal stock phrases in the generated text as a simple metric (Refusal Tuner)")}
        >
          <input
            type="checkbox"
            checked={steeringMeasureRefusal}
            onChange={(e) => setSteeringMeasureRefusal(e.target.checked)}
          />
          {t("拒否語彙カウントも計測", "Count refusal phrases")}
        </label>
        <button
          onClick={handleRunSteering}
          disabled={
            steeringRunState.status === "running" ||
            steeringTargetNodeIds.size === 0 ||
            parsedSteeringPositivePrompts.length === 0 ||
            parsedSteeringNegativePrompts.length === 0
          }
          style={{ ...buttonStyle, background: "var(--accent)", color: "#0f1418", fontWeight: 600 }}
        >
          {steeringRunState.status === "running" ? t("計算中...", "Computing...") : t("ステアリングを計算して比較", "Compute steering and compare")}
        </button>
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          {t(`下のリストから適用先ノードを選んでください(${steeringTargetNodeIds.size}個選択中)`, `Pick target nodes from the list below (${steeringTargetNodeIds.size} selected)`)}
        </span>
    </>
  );
}
