// 表示用の小さな整形関数(2026-09-22、App.tsx分割の第1段階で切り出し)。

export function formatParamCount(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${n}`;
}

// Token Journeyの表ヘッダー用に、ノードIDを短く読みやすい形に変換する。
// 例: "model.layers.3.self_attn" -> "L3-Attn", "model.layers.12.mlp" -> "L12-MLP"
export function shortenBlockId(blockId: string): string {
  const match = blockId.match(/model\.layers\.(\d+)\.(self_attn|mlp)/);
  if (!match) return blockId;
  const [, layerIndex, kind] = match;
  return `L${layerIndex}-${kind === "self_attn" ? "Attn" : "MLP"}`;
}
