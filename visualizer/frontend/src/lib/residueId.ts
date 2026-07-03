import type { ResidueEmbedding, SelectedResidueInfo } from "./types";

export interface ResidueLocator {
  structureId: string | null;
  chain: string;
  seq: number;
}

/** Parse structure:chain:seq, chain:seq, or legacy forms into chain + sequence. */
export function parseResidueLocator(id: string): ResidueLocator | null {
  const parts = id.split(":");
  if (parts.length < 2) return null;
  const seq = parseInt(parts[parts.length - 1], 10);
  if (Number.isNaN(seq)) return null;
  const chain = parts[parts.length - 2];
  if (parts.length >= 3) {
    return {
      structureId: parts.slice(0, -2).join(":"),
      chain,
      seq,
    };
  }
  return { structureId: null, chain, seq };
}

export function parseResidueChainSeq(
  id: string,
): { chain: string; seq: number } | null {
  const locator = parseResidueLocator(id);
  if (!locator) return null;
  return { chain: locator.chain, seq: locator.seq };
}

export function canonicalResidueId(
  structureId: string,
  chain: string,
  seq: number,
): string {
  return `${structureId}:${chain}:${seq}`;
}

export function residueIdsMatch(a: string, b: string): boolean {
  if (a === b) return true;
  const la = parseResidueLocator(a);
  const lb = parseResidueLocator(b);
  if (!la || !lb) return false;
  return la.chain === lb.chain && la.seq === lb.seq;
}

/** Resolve a click/highlight id to the canonical id used by hydrate embeddings. */
export function resolveCanonicalResidueId(
  rawId: string,
  structureId: string | null,
  knownIds?: Iterable<string>,
): string {
  const locator = parseResidueLocator(rawId);
  if (!locator) return rawId;

  const candidates: string[] = [];
  if (locator.structureId) {
    candidates.push(
      canonicalResidueId(locator.structureId, locator.chain, locator.seq),
    );
  }
  if (structureId) {
    candidates.push(
      canonicalResidueId(structureId, locator.chain, locator.seq),
    );
  }
  candidates.push(`${locator.chain}:${locator.seq}`);

  if (knownIds) {
    const known = new Set(knownIds);
    for (const candidate of candidates) {
      if (known.has(candidate)) return candidate;
    }
    const suffix = `:${locator.chain}:${locator.seq}`;
    for (const id of known) {
      if (id.endsWith(suffix)) return id;
    }
  }

  return candidates[0] ?? rawId;
}

export function findResidueEmbedding(
  rawId: string,
  structureId: string | null,
  residues: ResidueEmbedding[],
): ResidueEmbedding | undefined {
  const knownIds = residues.map((r) => r.residue_id);
  const canonical = resolveCanonicalResidueId(rawId, structureId, knownIds);
  return residues.find(
    (r) =>
      r.residue_id === canonical ||
      r.residue_id === rawId ||
      residueIdsMatch(r.residue_id, rawId),
  );
}

export function embeddingToSelectedResidue(
  embedding: ResidueEmbedding,
): SelectedResidueInfo {
  return {
    residue_id: embedding.residue_id,
    residue_name: embedding.residue_name ?? null,
    chain_label: embedding.chain_label ?? null,
    epistemic_uncertainty: embedding.epistemic_uncertainty ?? null,
    cone_depth: embedding.cone_depth ?? null,
  };
}

export function selectedResidueFromId(
  residueId: string,
  structureId: string | null,
  residues: ResidueEmbedding[],
): SelectedResidueInfo {
  const embedding = findResidueEmbedding(residueId, structureId, residues);
  if (embedding) return embeddingToSelectedResidue(embedding);

  const canonical = resolveCanonicalResidueId(
    residueId,
    structureId,
    residues.map((r) => r.residue_id),
  );
  const locator = parseResidueLocator(canonical);
  return {
    residue_id: canonical,
    residue_name: null,
    chain_label: locator?.chain ?? null,
    epistemic_uncertainty: null,
    cone_depth: null,
  };
}
