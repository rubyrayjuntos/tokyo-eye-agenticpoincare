import { describe, expect, it } from "vitest";

import {
  canonicalResidueId,
  findResidueEmbedding,
  parseResidueLocator,
  resolveCanonicalResidueId,
  residueIdsMatch,
} from "../residueId";
import type { ResidueEmbedding } from "../types";

const SAMPLE_RESIDUES: ResidueEmbedding[] = [
  {
    residue_id: "4uj1:A:105",
    residue_index: 104,
    chain_label: "A",
    residue_name: "GLY",
    x: 0.1,
    y: 0.2,
    cone_depth: 0.5,
    epistemic_uncertainty: 0.3,
    aleatoric_uncertainty: 0.1,
  },
];

describe("residueId", () => {
  it("parses canonical and short ids", () => {
    expect(parseResidueLocator("4uj1:A:105")).toEqual({
      structureId: "4uj1",
      chain: "A",
      seq: 105,
    });
    expect(parseResidueLocator("A:105")).toEqual({
      structureId: null,
      chain: "A",
      seq: 105,
    });
  });

  it("resolves molstar-style ids against known embeddings", () => {
    const canonical = resolveCanonicalResidueId(
      "A:105",
      "4uj1",
      SAMPLE_RESIDUES.map((r) => r.residue_id),
    );
    expect(canonical).toBe("4uj1:A:105");
  });

  it("matches residues across id formats", () => {
    expect(residueIdsMatch("4uj1:A:105", "A:105")).toBe(true);
    expect(residueIdsMatch("4uj1:A:105", "4uj1:B:105")).toBe(false);
  });

  it("finds embeddings from molstar click ids", () => {
    const hit = findResidueEmbedding("A:105", "4uj1", SAMPLE_RESIDUES);
    expect(hit?.residue_id).toBe("4uj1:A:105");
  });

  it("builds canonical ids", () => {
    expect(canonicalResidueId("4uj1", "A", 12)).toBe("4uj1:A:12");
  });
});
