import { useMemo, useState } from "react";
import type { LiveSweepPoint } from "../electron.d";

interface SweepChartProps {
  points: LiveSweepPoint[];
  currentScale: number | null; // 今のスライダーの値(縦の破線で示す)
  xLabel: string;
}

/**
 * Live Slider(2026-09-24追加)の倍率スイープ結果: 倍率(横軸)ごとのKLダイバージェンス(縦軸)の折れ線。
 * 1系列なので凡例は付けず、見出しで系列を示す。線2px・点8px、ホバーで各点の値を表示、
 * 補助線は控えめ、今の倍率を縦の破線で示す。数値は下の表でも確認できる。
 */
export function SweepChart({ points, currentScale, xLabel }: SweepChartProps) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 560;
  const H = 220;
  const M = { top: 12, right: 16, bottom: 34, left: 48 };
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;

  const geo = useMemo(() => {
    const xs = points.map((p) => p.scale);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMaxRaw = Math.max(...points.map((p) => p.kl), 1e-6);
    const yMax = niceMax(yMaxRaw);
    const x = (v: number) => M.left + ((v - xMin) / (xMax - xMin || 1)) * iw;
    const y = (v: number) => M.top + ih - (v / yMax) * ih;
    const yTicks = [0, yMax / 2, yMax];
    const xTicks = uniqueTicks([xMin, (xMin + xMax) / 2, xMax]);
    return { x, y, yMax, yTicks, xTicks, xMin, xMax };
  }, [points, iw, ih, M.left, M.top]);

  if (points.length === 0) return null;
  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${geo.x(p.scale)},${geo.y(p.kl)}`).join(" ");
  const hovered = hover !== null ? points[hover] : null;

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * iw + M.left;
    let best = 0;
    points.forEach((p, i) => {
      if (Math.abs(geo.x(p.scale) - px) < Math.abs(geo.x(points[best].scale) - px)) best = i;
    });
    setHover(best);
  };

  return (
    <div style={{ position: "relative", maxWidth: W }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`${xLabel}ごとのKLダイバージェンス`}>
        {geo.yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={M.left} x2={W - M.right} y1={geo.y(t)} y2={geo.y(t)} stroke="var(--border-default)" strokeWidth={1} />
            <text x={M.left - 6} y={geo.y(t) + 4} textAnchor="end" fontSize={10.5} fill="var(--text-muted)">
              {formatNum(t)}
            </text>
          </g>
        ))}
        {geo.xTicks.map((t) => (
          <text key={`x${t}`} x={geo.x(t)} y={H - M.bottom + 16} textAnchor="middle" fontSize={10.5} fill="var(--text-muted)">
            {formatNum(t)}
          </text>
        ))}
        <text x={M.left + iw / 2} y={H - 4} textAnchor="middle" fontSize={10.5} fill="var(--text-secondary)">
          {xLabel}
        </text>
        <text x={12} y={M.top + ih / 2} textAnchor="middle" fontSize={10.5} fill="var(--text-secondary)"
          transform={`rotate(-90 12 ${M.top + ih / 2})`}>
          KL
        </text>

        {currentScale !== null && currentScale >= geo.xMin && currentScale <= geo.xMax && (
          <line x1={geo.x(currentScale)} x2={geo.x(currentScale)} y1={M.top} y2={M.top + ih}
            stroke="var(--text-secondary)" strokeWidth={1} strokeDasharray="4 4" />
        )}

        <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {points.map((p, i) => (
          <circle key={i} cx={geo.x(p.scale)} cy={geo.y(p.kl)} r={hover === i ? 5.5 : 4}
            fill="var(--accent)" stroke="var(--bg-panel)" strokeWidth={2} />
        ))}
        {hovered && (
          <line x1={geo.x(hovered.scale)} x2={geo.x(hovered.scale)} y1={M.top} y2={M.top + ih}
            stroke="var(--border-strong)" strokeWidth={1} />
        )}
        <rect x={M.left} y={M.top} width={iw} height={ih} fill="transparent"
          onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
      </svg>

      {hovered && (
        <div
          style={{
            position: "absolute",
            left: `${(geo.x(hovered.scale) / W) * 100}%`,
            top: 0,
            transform: geo.x(hovered.scale) > W * 0.65 ? "translateX(-105%)" : "translateX(8px)",
            background: "var(--bg-panel-raised)",
            border: "1px solid var(--border-strong)",
            borderRadius: "var(--radius-sm)",
            padding: "6px 8px",
            fontSize: 11,
            color: "var(--text-primary)",
            pointerEvents: "none",
            whiteSpace: "nowrap",
          }}
        >
          <div>倍率 {formatNum(hovered.scale)}</div>
          <div>KL {hovered.kl.toFixed(4)}</div>
          <div style={{ color: "var(--text-secondary)" }}>
            最有力: {JSON.stringify(hovered.top_token_text)}（{(hovered.top_prob * 100).toFixed(1)}%）
          </div>
        </div>
      )}
    </div>
  );
}

function niceMax(v: number): number {
  const exp = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) {
    if (m * exp >= v) return m * exp;
  }
  return 10 * exp;
}

function uniqueTicks(ts: number[]): number[] {
  return Array.from(new Set(ts.map((t) => Math.round(t * 1000) / 1000)));
}

function formatNum(v: number): string {
  if (v === 0) return "0";
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 1) return String(Math.round(v * 100) / 100);
  return String(Math.round(v * 1000) / 1000);
}
