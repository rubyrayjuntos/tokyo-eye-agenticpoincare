import { useEffect, useState } from "react";

import { useDashboard } from "../lib/context";
import { api } from "../lib/api";
import type { ResidueEmbedding } from "../lib/types";

export function useEmbeddingResidues(structureId: string | null) {
  const { refreshKey } = useDashboard();
  const [residues, setResidues] = useState<ResidueEmbedding[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!structureId) {
      setResidues([]);
      setError(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .getEmbeddings(structureId)
      .then((result) => {
        if (cancelled) return;
        setResidues(result.residues ?? []);
      })
      .catch((err) => {
        if (cancelled) return;
        setResidues([]);
        setError(err instanceof Error ? err.message : "Failed to load embeddings");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [structureId, refreshKey]);

  return { residues, loading, error };
}
