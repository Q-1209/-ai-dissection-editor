import type { ActivationStatsEntry } from "../electron.d";

interface ActivationStatsListProps {
  stats: Record<string, ActivationStatsEntry>;
  // 表示順序を制御するための、実行順序リスト(execution.executed_order)
  order: string[];
  // "linear": stdの値をそのまま棒の長さに使う。層間でstdの桁が大きく異なる場合
  // (例: Massive Activationsが起きている層とそうでない層が混在する場合)、
  // 小さい値がほぼ見えなくなる。
  // "log": log(1+std)を棒の長さに使う。桁が大きく異なる値同士を同時に見比べやすくする。
  scale: "linear" | "log";
}

function barWidthPercent(std: number, maxStd: number, scale: "linear" | "log"): number {
  if (scale === "linear") {
    return Math.min(100, (std / maxStd) * 100);
  }
  // log(1+x)を使うことで、std=0のときも0のまま、std>=0の範囲で単調増加を保つ
  const logStd = Math.log1p(std);
  const logMax = Math.log1p(maxStd);
  if (logMax <= 0) return 0;
  return Math.min(100, (logStd / logMax) * 100);
}

/**
 * 各ブロックの活性化値統計(mean/std/min/max)を一覧表示する。
 * stdを横棒グラフの長さとして視覚化し、層を追うごとに値の広がりが
 * どう変化するかを一目で把握できるようにする。
 */
export function ActivationStatsList({ stats, order, scale }: ActivationStatsListProps) {
  const entries = order.filter((id) => id in stats).map((id) => ({ id, ...stats[id] }));
  if (entries.length === 0) return null;

  const maxStd = Math.max(...entries.map((e) => e.std), 1e-6);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {entries.map((e) => (
        <div key={e.id} style={{ fontSize: 11 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              color: "var(--text-secondary)",
              marginBottom: 2,
            }}
          >
            <span style={{ fontFamily: "var(--font-mono)" }}>{e.id}</span>
            <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
              mean={e.mean.toFixed(3)} std={e.std.toFixed(3)} [{e.min.toFixed(2)}, {e.max.toFixed(2)}]
            </span>
          </div>
          <div
            style={{
              height: 6,
              background: "var(--bg-panel-raised)",
              borderRadius: 3,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${barWidthPercent(e.std, maxStd, scale)}%`,
                background: "var(--accent)",
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
