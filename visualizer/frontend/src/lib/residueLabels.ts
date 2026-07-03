import type { ResidueEmbedding } from "./types";

/** Position label from governed residue fields — no fold-specific heuristics. */
export function formatResiduePosition(residue: ResidueEmbedding): string {
  const name =
    residue.residue_name && residue.residue_name.trim().length > 0
      ? ` · ${residue.residue_name}`
      : "";
  return `Chain ${residue.chain_label} · ${residue.residue_index}${name}`;
}

export function formatResidueLabel(residueId: string): string {
  const parts = residueId.split(":");
  if (parts.length >= 3) {
    return `${parts[1]} · ${parts[2]}`;
  }
  return residueId.replace(":", " · ");
}
