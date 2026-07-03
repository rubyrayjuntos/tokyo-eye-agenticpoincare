import { useEffect, useRef, useState } from "react";

import {
  canonicalResidueId,
  parseResidueChainSeq,
} from "../../lib/residueId";
import { applyMolstarResidueColors } from "../../lib/molstarResidueColors";
import type { ResidueEmbedding } from "../../lib/types";
import type { ResidueMetricSeries, ViewportColorMode } from "../../lib/viewportColorMetrics";

declare global {
  interface Window {
    molstar?: {
      Viewer: {
        create: (
          parent: HTMLElement,
          options: Record<string, unknown>,
        ) => Promise<MolstarViewerInstance>;
      };
      lib?: {
        structure: {
          Queries: { generators: { atoms: (q: unknown) => (ctx: unknown) => unknown } };
          QueryContext: new (structure: unknown) => unknown;
          StructureSelection: { toLociWithSourceUnits: (sel: unknown) => unknown };
          StructureProperties: {
            chain: { auth_asym_id: (el: unknown) => string };
            residue: { auth_seq_id: (el: unknown) => number };
          };
        };
      };
    };
  }
}

interface MolstarViewerInstance {
  plugin: {
    managers: {
      structure: {
        hierarchy: { current: { structures: Array<{ cell?: { obj?: { data?: unknown } } }> } };
        focus: { clear: () => void; setFromLoci: (loci: unknown) => void };
      };
      interactivity: { lociSelects: { deselectAll: () => void; selectOnly: (p: { loci: unknown }) => void } };
      camera: { focusLoci: (loci: unknown) => void };
    };
    canvas3d?: { setProps: (p: Record<string, unknown>) => void };
    handleResize: () => void;
    behaviors?: {
      interaction?: {
        click?: { subscribe: (fn: (event: MolstarClickEvent) => void) => void };
      };
    };
  };
  loadPdb: (pdbId: string) => Promise<unknown>;
  loadStructureFromData?: (
    data: string,
    format: string,
    options?: { dataLabel?: string },
  ) => Promise<void>;
  dispose?: () => void;
}

interface MolstarClickEvent {
  current?: {
    loci?: {
      kind?: string;
      elements?: Array<{ unit: MolstarUnit; indices: number[] }>;
    };
  };
}

interface MolstarUnit {
  model?: {
    atomicHierarchy?: {
      residueAtomSegments: { index: number[] };
      chainAtomSegments: { index: number[] };
      residues: {
        auth_seq_id?: { value: (i: number) => number };
        label_seq_id?: { value: (i: number) => number };
      };
      chains: {
        auth_asym_id?: { value: (i: number) => string };
        label_asym_id?: { value: (i: number) => string };
      };
    };
  };
  elements: number[];
}

/** Pinned version — unversioned /npm/molstar/ can shift API and hang loads. */
const MOLSTAR_VERSION = "4.6.0";
const MOLSTAR_CSS = `https://cdn.jsdelivr.net/npm/molstar@${MOLSTAR_VERSION}/build/viewer/molstar.css`;
const MOLSTAR_JS = `https://cdn.jsdelivr.net/npm/molstar@${MOLSTAR_VERSION}/build/viewer/molstar.js`;
const RCSB_PDB_URL = "https://files.rcsb.org/download";
const LOAD_PDB_TIMEOUT_MS = 45_000;

let molstarLoadPromise: Promise<void> | null = null;

function loadMolstarAssets(): Promise<void> {
  if (window.molstar?.Viewer) return Promise.resolve();

  if (molstarLoadPromise) return molstarLoadPromise;

  molstarLoadPromise = new Promise((resolve, reject) => {
    const finish = () => {
      if (window.molstar?.Viewer) {
        resolve();
      } else {
        molstarLoadPromise = null;
        reject(new Error("Mol* loaded but Viewer API is unavailable"));
      }
    };

    if (!document.querySelector(`link[href="${MOLSTAR_CSS}"]`)) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = MOLSTAR_CSS;
      document.head.appendChild(link);
    }

    const existing = document.querySelector(
      `script[src="${MOLSTAR_JS}"]`,
    ) as HTMLScriptElement | null;

    if (existing) {
      if (window.molstar?.Viewer) {
        resolve();
        return;
      }
      existing.addEventListener("load", finish, { once: true });
      existing.addEventListener(
        "error",
        () => {
          molstarLoadPromise = null;
          reject(new Error("Mol* script failed to load"));
        },
        { once: true },
      );
      return;
    }

    const script = document.createElement("script");
    script.src = MOLSTAR_JS;
    script.async = true;
    script.onload = finish;
    script.onerror = () => {
      molstarLoadPromise = null;
      reject(new Error("Mol* script failed to load from CDN"));
    };
    document.head.appendChild(script);
  });

  return molstarLoadPromise;
}

function normalizePdbId(pdbId: string): string {
  const trimmed = pdbId.trim().toLowerCase();
  const colon = trimmed.lastIndexOf(":");
  if (colon >= 0) {
    return trimmed.slice(colon + 1);
  }
  return trimmed.replace(/^rcsb_/, "").replace(/^rcsb-/, "");
}

async function loadStructureWithFallback(
  viewer: MolstarViewerInstance,
  pdbId: string,
): Promise<void> {
  const normalized = normalizePdbId(pdbId);

  const loadViaRcsb = async () => {
    const task = viewer.loadPdb(normalized);
    await Promise.race([
      task,
      new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error("RCSB loadPdb timed out")), LOAD_PDB_TIMEOUT_MS);
      }),
    ]);
  };

  try {
    await loadViaRcsb();
    return;
  } catch (primaryError) {
    if (!viewer.loadStructureFromData) {
      throw primaryError;
    }
    const response = await fetch(`${RCSB_PDB_URL}/${normalized}.pdb`);
    if (!response.ok) {
      throw new Error(
        `Could not load ${normalized.toUpperCase()} from RCSB (${response.status})`,
      );
    }
    const pdbText = await response.text();
    await viewer.loadStructureFromData(pdbText, "pdb", {
      dataLabel: normalized.toUpperCase(),
    });
  }
}

export interface MolstarViewerProps {
  pdbId: string | null;
  structureId?: string | null;
  colorMode?: ViewportColorMode;
  residues?: ResidueEmbedding[];
  metricSeries?: ResidueMetricSeries;
  highlightedResidues?: string[];
  selectedResidueId?: string | null;
  onResidueClick?: (residueId: string) => void;
  onClearSelection?: () => void;
}

export function MolstarViewer({
  pdbId,
  structureId = null,
  colorMode = "cone_depth",
  residues = [],
  metricSeries = { values: new Map(), min: 0, max: 1, dataAvailable: true },
  highlightedResidues = [],
  selectedResidueId = null,
  onResidueClick,
  onClearSelection,
}: MolstarViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<MolstarViewerInstance | null>(null);
  const onResidueClickRef = useRef(onResidueClick);
  const onClearSelectionRef = useRef(onClearSelection);
  const structureIdRef = useRef(structureId);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    onResidueClickRef.current = onResidueClick;
  }, [onResidueClick]);

  useEffect(() => {
    onClearSelectionRef.current = onClearSelection;
  }, [onClearSelection]);

  useEffect(() => {
    structureIdRef.current = structureId;
  }, [structureId]);

  useEffect(() => {
    if (!pdbId) {
      setStatus("idle");
      setError(null);
      return;
    }

    const container = containerRef.current;
    if (!container) return;

    let disposed = false;
    setStatus("loading");
    setError(null);

    const mount = async () => {
      try {
        await loadMolstarAssets();
        if (disposed || !containerRef.current) return;

        if (viewerRef.current) {
          try {
            viewerRef.current.dispose?.();
          } catch {
            /* ignore */
          }
          viewerRef.current = null;
        }

        const viewer = await window.molstar!.Viewer.create(containerRef.current, {
          layoutIsExpanded: false,
          layoutShowControls: false,
          layoutShowSequence: false,
          layoutShowLog: false,
          layoutShowLeftPanel: false,
          viewportShowExpand: true,
          viewportShowControls: true,
          viewportShowSelectionMode: false,
          pdbProvider: "rcsb",
          emdbProvider: "rcsb",
        });

        if (disposed) {
          viewer.dispose?.();
          return;
        }

        viewerRef.current = viewer;
        try {
          viewer.plugin.canvas3d?.setProps({
            renderer: { backgroundColor: 0x070810 },
            marking: {
              selectEdgeColor: 0xc026d3,
              highlightEdgeColor: 0xeca53a,
              edgeScale: 1.4,
            },
          });
        } catch {
          /* optional mol* props */
        }

        viewer.plugin.behaviors?.interaction?.click?.subscribe((event) => {
          const loci = event.current?.loci;
          if (!loci || loci.kind !== "element-loci" || !loci.elements?.length) {
            onClearSelectionRef.current?.();
            return;
          }
          const element = loci.elements[0];
          const unit = element.unit;
          const hierarchy = unit.model?.atomicHierarchy;
          if (!hierarchy) return;
          const atomIndex = unit.elements[element.indices[0]];
          const rI = hierarchy.residueAtomSegments.index[atomIndex];
          const cI = hierarchy.chainAtomSegments.index[atomIndex];
          const seqIdCol =
            hierarchy.residues.auth_seq_id ?? hierarchy.residues.label_seq_id;
          const chainIdCol =
            hierarchy.chains.auth_asym_id ?? hierarchy.chains.label_asym_id;
          const seqId = seqIdCol?.value(rI);
          const chainId = chainIdCol?.value(cI);
          if (seqId != null && chainId) {
            const sid = structureIdRef.current;
            const clickId = sid
              ? canonicalResidueId(sid, chainId, seqId)
              : `${chainId}:${seqId}`;
            onResidueClickRef.current?.(clickId);
          }
        });

        await loadStructureWithFallback(viewer, pdbId);
        if (!disposed) {
          viewer.plugin.handleResize();
          setStatus("ready");
        }
      } catch (err) {
        if (!disposed) {
          setStatus("error");
          setError(err instanceof Error ? err.message : "Mol* load failed");
        }
      }
    };

    void mount();

    const ro = new ResizeObserver(() => {
      try {
        viewerRef.current?.plugin.handleResize();
      } catch {
        /* ignore */
      }
    });
    ro.observe(container);

    return () => {
      disposed = true;
      ro.disconnect();
      try {
        viewerRef.current?.dispose?.();
      } catch {
        /* ignore */
      }
      viewerRef.current = null;
    };
  }, [pdbId]);

  useEffect(() => {
    const viewer = viewerRef.current;
    const S = window.molstar?.lib?.structure;
    if (!viewer || !S || status !== "ready") return;

    try {
      const data = viewer.plugin.managers.structure.hierarchy.current.structures[0];
      const structure = data?.cell?.obj?.data;
      if (!structure) return;

      const im = viewer.plugin.managers.interactivity;
      const selectionIds = [
        ...highlightedResidues,
        ...(selectedResidueId ? [selectedResidueId] : []),
      ];
      const chainSeqKeys = new Set(
        selectionIds
          .map(parseResidueChainSeq)
          .filter((v): v is { chain: string; seq: number } => v != null)
          .map((v) => `${v.chain}:${v.seq}`),
      );

      im.lociSelects.deselectAll();
      try {
        viewer.plugin.managers.structure.focus.clear();
      } catch {
        /* ignore */
      }
      if (!chainSeqKeys.size) return;

      const query = S.Queries.generators.atoms({
        residueTest: (ctx: { element: unknown }) => {
          const chain = S.StructureProperties.chain.auth_asym_id(ctx.element);
          const seq = S.StructureProperties.residue.auth_seq_id(ctx.element);
          return chainSeqKeys.has(`${chain}:${seq}`);
        },
      });
      const sel = query(new S.QueryContext(structure));
      const loci = S.StructureSelection.toLociWithSourceUnits(sel);
      im.lociSelects.selectOnly({ loci });
      try {
        viewer.plugin.managers.structure.focus.setFromLoci(loci);
      } catch {
        /* ignore */
      }
      if (selectionIds.length === 1) {
        try {
          viewer.plugin.managers.camera.focusLoci(loci);
        } catch {
          /* ignore */
        }
      }
    } catch {
      /* highlight best-effort */
    }
  }, [highlightedResidues, selectedResidueId, status]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || status !== "ready") return;

    void applyMolstarResidueColors(
      viewer.plugin as Parameters<typeof applyMolstarResidueColors>[0],
      colorMode,
      residues,
      metricSeries,
    ).catch(() => {
      /* coloring is best-effort */
    });
  }, [colorMode, metricSeries, residues, status]);

  return (
    <div className="relative h-full min-h-0 w-full bg-[#070810]">
      <div ref={containerRef} className="absolute inset-0 [&_.msp-plugin]:h-full" />
      {status === "loading" && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#070810] text-[11px] text-text-muted">
          Loading {pdbId ? normalizePdbId(pdbId).toUpperCase() : "structure"} from RCSB…
        </div>
      )}
      {status === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-[#070810] px-4 text-center text-[11px] text-error">
          <span>{error ?? "Mol* could not load structure"}</span>
          <span className="text-[10px] text-text-muted">
            Check network access to cdn.jsdelivr.net and files.rcsb.org
          </span>
        </div>
      )}
    </div>
  );
}
