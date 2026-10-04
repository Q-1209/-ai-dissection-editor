import type { AttentionWeightsEntry } from "../electron.d";

interface AttentionHeatmapProps {
  entry: AttentionWeightsEntry;
  tokenLabels: string[];
  headIndex?: number; // entry.dataが3次元(ヘッド別)の場合、どのヘッドを表示するか
}

const CELL_SIZE = 28;
const LABEL_WIDTH = 80;
const LABEL_HEIGHT = 60;

// 値(0〜1程度のattention weight)を色に変換する。0=背景色、1=アクセント色に近づく。
function valueToColor(value: number): string {
  const clamped = Math.max(0, Math.min(1, value));
  // var(--accent) は #5b9bd5 相当。背景(var(--bg-app) #0f1418)から線形補間する。
  const r = Math.round(0x0f + (0x5b - 0x0f) * clamped);
  const g = Math.round(0x14 + (0x9b - 0x14) * clamped);
  const b = Math.round(0x18 + (0xd5 - 0x18) * clamped);
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Attention重みのヒートマップ。2次元(ヘッド平均済み、[query_len, key_len])を想定。
 * 3次元(ヘッド別)の場合は headIndex で指定したヘッドのスライスを描画する。
 */
export function AttentionHeatmap({ entry, tokenLabels, headIndex }: AttentionHeatmapProps) {
  const is3D = entry.shape.length === 3;
  const matrix: number[][] = is3D
    ? ((entry.data as number[][][])[headIndex ?? 0] ?? [])
    : (entry.data as number[][]);

  const seqLen = matrix.length;
  const width = LABEL_WIDTH + seqLen * CELL_SIZE;
  const height = LABEL_HEIGHT + seqLen * CELL_SIZE;

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {/* 列ラベル(key側、上部に縦書き風に配置) */}
      {tokenLabels.map((label, i) => (
        <text
          key={`col-${i}`}
          x={LABEL_WIDTH + i * CELL_SIZE + CELL_SIZE / 2}
          y={LABEL_HEIGHT - 8}
          fontSize={10}
          fill="var(--text-secondary)"
          textAnchor="start"
          transform={`rotate(-45, ${LABEL_WIDTH + i * CELL_SIZE + CELL_SIZE / 2}, ${LABEL_HEIGHT - 8})`}
        >
          {label}
        </text>
      ))}

      {/* 行ラベル(query側、左に配置) */}
      {tokenLabels.map((label, i) => (
        <text
          key={`row-${i}`}
          x={LABEL_WIDTH - 8}
          y={LABEL_HEIGHT + i * CELL_SIZE + CELL_SIZE / 2 + 4}
          fontSize={10}
          fill="var(--text-secondary)"
          textAnchor="end"
        >
          {label}
        </text>
      ))}

      {/* ヒートマップ本体 */}
      {matrix.map((row, i) =>
        row.map((value, j) => (
          <rect
            key={`cell-${i}-${j}`}
            x={LABEL_WIDTH + j * CELL_SIZE}
            y={LABEL_HEIGHT + i * CELL_SIZE}
            width={CELL_SIZE - 1}
            height={CELL_SIZE - 1}
            fill={valueToColor(value)}
          >
            <title>
              {`query: ${tokenLabels[i] ?? i} -> key: ${tokenLabels[j] ?? j}\nweight: ${value.toFixed(4)}`}
            </title>
          </rect>
        ))
      )}
    </svg>
  );
}
