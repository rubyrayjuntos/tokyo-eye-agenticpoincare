import { useEffect, useRef, useState, useMemo } from "react";
import { useDashboard } from "../lib/context";
import { useHydration } from "../context/HydrationProvider";

declare global {
  interface Window {
    $3Dmol: any;
  }
}

const RCSB_PDB_URL = "https://files.rcsb.org/download";

type StructureColorMode = "spectrum" | "cone_depth" | "epistemic" | "aleatoric" | "plasticity" | "allosteric" | "resistance" | "pockets" | "drug_candidates";

/** Viridis-inspired color scale matching PoincareScatter */
function valueToHex(v: number, min = 0, max = 1): string {
  const t = Math.max(0, Math.min(1, (v - min) / (max - min + 1e-8)));
  const r = Math.round(68 + t * (253 - 68));
  const g = Math.round(1 + t * (231 - 1));
  const b = Math.round(
    84 + (t < 0.5 ? t * 2 * (168 - 84) : (1 - t) * 2 * 168)
  );
  return `0x${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

/** Diverging blue → white → orange → bright red/yellow for plasticity risk.
 *  Tuned so high-risk pops against dark background. */
function plasticityToHex(t: number): string {
  let r: number, g: number, b: number;
  if (t < 0.4) {
    // Blue (safe) → light gray
    const s = t / 0.4;
    r = Math.round(40 + s * (180 - 40));
    g = Math.round(80 + s * (180 - 80));
    b = Math.round(220 + s * (200 - 220));
  } else if (t < 0.7) {
    // Light gray → orange
    const s = (t - 0.4) / 0.3;
    r = Math.round(180 + s * (255 - 180));
    g = Math.round(180 - s * (180 - 140));
    b = Math.round(200 - s * (200 - 30));
  } else {
    // Orange → bright red/yellow (vibrant danger zone)
    const s = (t - 0.7) / 0.3;
    r = 255;
    g = Math.round(140 - s * (140 - 50));
    b = Math.round(30 + s * (20 - 30));
  }
  return `0x${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

/** Compute plasticity risk for a residue */
function computePlasticityRisk(
  coneDepth: number, uncertainty: number,
  dMin: number, dMax: number, uMin: number, uMax: number
): number {
  const normDepth = (coneDepth - dMin) / (dMax - dMin + 1e-8);
  const normUncert = (uncertainty - uMin) / (uMax - uMin + 1e-8);
  return Math.pow((1.0 - normDepth) * 0.55 + normUncert * 0.45, 1.3);
}

/** Burial descriptor from cone depth */
function burialLabel(coneDepth: number): string {
  if (coneDepth < 2.5) return "Surface";
  if (coneDepth < 4.5) return "Semi-buried";
  return "Buried";
}

/** Uncertainty descriptor */
function uncertaintyLabel(uncertainty: number): string {
  if (uncertainty > 8.0) return "High";
  if (uncertainty > 3.0) return "Medium";
  return "Low";
}

/** Green → Cyan → Magenta for allosteric potential */
function allostericToHex(t: number): string {
  let r: number, g: number, b: number;
  if (t < 0.5) {
    const s = t * 2;
    r = Math.round(20 + s * (0 - 20));
    g = Math.round(80 + s * (220 - 80));
    b = Math.round(40 + s * (220 - 40));
  } else {
    const s = (t - 0.5) * 2;
    r = Math.round(0 + s * (220));
    g = Math.round(220 - s * (220 - 40));
    b = Math.round(220 + s * (255 - 220));
  }
  return `0x${Math.round(r).toString(16).padStart(2, "0")}${Math.round(g).toString(16).padStart(2, "0")}${Math.round(b).toString(16).padStart(2, "0")}`;
}

/** Sequential warm colormap: white → yellow → orange → deep red for resistance sensitivity */
function resistanceToHex(t: number): string {
  let r: number, g: number, b: number;
  if (t < 0.3) {
    // White/light gray → light yellow
    const s = t / 0.3;
    r = Math.round(240 + s * (255 - 240));
    g = Math.round(240 - s * (240 - 220));
    b = Math.round(240 - s * (240 - 100));
  } else if (t < 0.6) {
    // Light yellow → orange
    const s = (t - 0.3) / 0.3;
    r = 255;
    g = Math.round(220 - s * (220 - 140));
    b = Math.round(100 - s * (100 - 20));
  } else {
    // Orange → deep red
    const s = (t - 0.6) / 0.4;
    r = Math.round(255 - s * (255 - 180));
    g = Math.round(140 - s * (140 - 20));
    b = Math.round(20 - s * 10);
  }
  return `0x${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

/** Green → yellow → red gradient for pocket druggability score */
function pocketDruggabilityToHex(score: number): string {
  const t = Math.max(0, Math.min(1, score));
  let r: number, g: number, b: number;
  if (t < 0.5) {
    // Green → yellow
    const s = t * 2;
    r = Math.round(40 + s * (240 - 40));
    g = Math.round(200 + s * (220 - 200));
    b = Math.round(80 - s * (80 - 30));
  } else {
    // Yellow → bright red-orange
    const s = (t - 0.5) * 2;
    r = Math.round(240 + s * (255 - 240));
    g = Math.round(220 - s * (220 - 60));
    b = Math.round(30 + s * (40 - 30));
  }
  return `0x${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

/** Drug candidate coloring: higher druggability = warmer. ADMET pass = brighter saturation */
function candidateDruggabilityToHex(score: number, admetPass: boolean): string {
  const t = Math.max(0, Math.min(1, score));
  let r: number, g: number, b: number;
  if (t < 0.4) {
    // Muted teal
    r = 40; g = 160; b = 180;
  } else if (t < 0.7) {
    // Teal → gold
    const s = (t - 0.4) / 0.3;
    r = Math.round(40 + s * (220 - 40));
    g = Math.round(160 + s * (200 - 160));
    b = Math.round(180 - s * (180 - 40));
  } else {
    // Gold → bright amber
    const s = (t - 0.7) / 0.3;
    r = Math.round(220 + s * (255 - 220));
    g = Math.round(200 - s * (200 - 140));
    b = Math.round(40 - s * (40 - 20));
  }
  // ADMET pass: boost brightness; fail: dim
  if (!admetPass) {
    r = Math.round(r * 0.5);
    g = Math.round(g * 0.5);
    b = Math.round(b * 0.5);
  }
  return `0x${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

/** Risk interpretation for tooltip */
function riskInterpretation(risk: number, normDepth: number, normUncert: number): string {
  if (risk > 0.65) {
    if (normDepth < 0.4 && normUncert > 0.5)
      return "Buried hinge residue with high conformational uncertainty. Likely conformational switch / resistance hotspot.";
    if (normDepth < 0.5)
      return "Semi-buried position with dynamic instability. Potential resistance escape route.";
    if (normUncert > 0.7)
      return "High conformational uncertainty — potential allosteric switch.";
    return "Buried position with elevated plasticity. Monitor for resistance mutations.";
  }
  if (risk > 0.4) {
    if (normUncert > 0.5)
      return "Flexible region with moderate burial. May contribute to induced-fit dynamics.";
    return "Partially exposed with some conformational freedom.";
  }
  return "Stable position with low conformational uncertainty.";
}

/** Infer functional group from secondary structure + residue position.
 *  For kinases, uses well-known motif residue ranges.
 *  For general proteins, uses SS + burial heuristics. */
function inferFunctionalGroup(
  resIndex: number, ss: string, coneDepth: number, pdbId: string
): string | null {
  // Known kinase motifs (BCR-ABL / ABL1 numbering)
  const pdb = pdbId.toLowerCase();
  if (pdb === "2hyy" || pdb === "1iep" || pdb === "2gqg" || pdb === "1opj") {
    // ABL1 kinase domain numbering
    if (resIndex >= 248 && resIndex <= 256) return "P-loop (Gly-rich)";
    if (resIndex >= 286 && resIndex <= 290) return "αC-helix";
    if (resIndex >= 313 && resIndex <= 317) return "Hinge region";
    if (resIndex === 315) return "Gatekeeper (T315)";
    if (resIndex >= 381 && resIndex <= 402) return "Activation loop (A-loop)";
    if (resIndex >= 356 && resIndex <= 362) return "Catalytic loop (HRD)";
    if (resIndex >= 371 && resIndex <= 377) return "DFG motif";
  }
  // Adenylate kinase (4AKE)
  if (pdb === "4ake") {
    if (resIndex >= 1 && resIndex <= 29) return "P-loop / NMP-binding";
    if (resIndex >= 30 && resIndex <= 59) return "NMP-binding domain";
    if (resIndex >= 113 && resIndex <= 176) return "LID domain";
    if (resIndex >= 60 && resIndex <= 112) return "CORE domain";
  }
  // General heuristic based on SS + burial
  if (ss === "Helix" && coneDepth > 5.0) return "Buried helix (core)";
  if (ss === "Sheet" && coneDepth > 4.0) return "Buried β-strand (core)";
  if (ss === "Coil" && coneDepth < 2.5) return "Exposed loop";
  if (ss === "Coil" && coneDepth >= 2.5 && coneDepth < 4.5) return "Linker / hinge";
  return null;
}

/** Load 3Dmol.js from CDN if not already loaded */
function ensure3Dmol(): Promise<void> {
  if (window.$3Dmol) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const existing = document.querySelector('script[data-3dmol]');
    if (existing) {
      existing.addEventListener("load", () => resolve());
      return;
    }
    const script = document.createElement("script");
    script.src = "https://3Dmol.org/build/3Dmol-min.js";
    script.setAttribute("data-3dmol", "true");
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load 3Dmol.js"));
    document.head.appendChild(script);
  });
}

interface MolecularViewerProps {
  highlightResidues?: string[];
  onColorModeChange?: (mode: StructureColorMode) => void;
  onRiskThresholdChange?: (threshold: number) => void;
}

export default function MolecularViewer({ highlightResidues = [], onColorModeChange, onRiskThresholdChange }: MolecularViewerProps) {
  const { activeStructure, refreshKey, currentDirective, isRadarActive, setIsRadarActive } = useDashboard();
  const { embeddings, graphMetrics, resistanceData, pharmacophorePockets, drugCandidates, allostericSites } = useHydration();
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [colorMode, setColorMode] = useState<StructureColorMode>("spectrum");
  const [riskThreshold, setRiskThreshold] = useState(0); // 0 = show all, 0.8 = top 20%

  // Sync standalone "allosteric" color mode to the global radar effect (exact same pseudo-bloom ultramarine + locks)
  // This makes the dropdown trigger the full radar visual, and the global flag ensures persistence.
  useEffect(() => {
    if (colorMode === "allosteric" && !isRadarActive) {
      setIsRadarActive(true);
    }
  }, [colorMode, isRadarActive, setIsRadarActive]);
  const [hoveredResidue, setHoveredResidue] = useState<{
    residue_id: string;
    residue_index: number;
    chain_label: string;
    resn: string;
    ss: string;
    cone_depth: number;
    uncertainty: number;
    risk: number;
    x: number;
    y: number;
  } | null>(null);

  // Propagate color mode and risk threshold to parent
  useEffect(() => {
    onColorModeChange?.(colorMode);
  }, [colorMode, onColorModeChange]);

  useEffect(() => {
    onRiskThresholdChange?.(riskThreshold);
  }, [riskThreshold, onRiskThresholdChange]);

  // Precompute plasticity risk values for all residues
  const plasticityData = useMemo(() => {
    if (!embeddings || embeddings.residues.length === 0) return null;
    const depths = embeddings.residues.map((r) => r.cone_depth);
    const uncerts = embeddings.residues.map((r) => r.epistemic_uncertainty);
    const dMin = Math.min(...depths);
    const dMax = Math.max(...depths);
    const uMin = Math.min(...uncerts);
    const uMax = Math.max(...uncerts);
    const risks = embeddings.residues.map((r) =>
      computePlasticityRisk(r.cone_depth, r.epistemic_uncertainty, dMin, dMax, uMin, uMax)
    );
    return { dMin, dMax, uMin, uMax, risks };
  }, [embeddings]);

  useEffect(() => {
    if (!activeStructure || !containerRef.current) return;
    let cancelled = false;

    async function loadStructure() {
      setLoading(true);
      setError(null);
      try {
        await ensure3Dmol();
        if (cancelled || !containerRef.current) return;

        if (viewerRef.current) {
          viewerRef.current.clear();
        } else {
          containerRef.current.innerHTML = "";
          viewerRef.current = window.$3Dmol.createViewer(containerRef.current, {
            backgroundColor: "0x09090b",
          });
        }

        const pdbId = activeStructure.pdb_id.toLowerCase();
        const url = `${RCSB_PDB_URL}/${pdbId}.pdb`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`PDB fetch failed: ${response.status}`);
        const pdbData = await response.text();
        if (cancelled) return;

        const viewer = viewerRef.current;
        viewer.addModel(pdbData, "pdb");
        viewer.setStyle({}, { cartoon: { color: "spectrum" } });
        viewer.zoomTo();
        viewer.render();
      } catch (e: any) {
        if (!cancelled) setError(e.message || "Failed to load structure");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadStructure();
    return () => { cancelled = true; };
  }, [activeStructure, refreshKey]);

  // Apply coloring + highlights
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !activeStructure) return;

    if (colorMode !== "spectrum" && embeddings && embeddings.residues.length > 0) {
      if (colorMode === "plasticity" && plasticityData) {
        const { dMin, dMax, uMin, uMax, risks } = plasticityData;
        viewer.setStyle({}, { cartoon: { color: "0x1a1a2e" } });
        for (let i = 0; i < embeddings.residues.length; i++) {
          const residue = embeddings.residues[i];
          const risk = risks[i];
          // Apply threshold: below threshold → dim gray
          if (risk < riskThreshold) {
            viewer.setStyle(
              { resi: residue.residue_index, chain: residue.chain_label },
              { cartoon: { color: "0x222233", opacity: 0.4 } }
            );
          } else {
            const hex = plasticityToHex(Math.max(0, Math.min(1, risk)));
            viewer.setStyle(
              { resi: residue.residue_index, chain: residue.chain_label },
              { cartoon: { color: hex } }
            );
          }
        }
        // Add glow spheres for top risk residues
        viewer.removeAllShapes();
        for (let i = 0; i < embeddings.residues.length; i++) {
          if (risks[i] > 0.75) {
            const r = embeddings.residues[i];
            viewer.addStyle(
              { resi: r.residue_index, chain: r.chain_label, atom: "CA" },
              { sphere: { radius: 1.6, color: "0xff3333", opacity: 0.6 } }
            );
          }
        }
      } else if (colorMode === "allosteric" && graphMetrics && graphMetrics.metrics.length > 0) {
        // Local Allosteric Potential: betweenness × uncertainty
        // High betweenness = controls domain communication
        // High uncertainty = conformationally dynamic
        // Combined = residues that are both structurally critical AND plastic
        const metricsMap = new Map(graphMetrics.metrics.map((m) => [m.residue_id, m]));
        const scores: number[] = [];
        for (const residue of embeddings.residues) {
          const gm = metricsMap.get(residue.residue_id);
          const betweenness = gm?.betweenness ?? 0;
          const normUncert = residue.epistemic_uncertainty;
          // Allosteric potential = sqrt(betweenness) × uncertainty^0.8
          // sqrt dampens extreme betweenness outliers (hubs); ^0.8 keeps uncertainty meaningful
          const score = Math.sqrt(betweenness) * Math.pow(normUncert + 0.01, 0.8);
          scores.push(score);
        }
        const sMin = Math.min(...scores);
        const sMax = Math.max(...scores);

        viewer.setStyle({}, { cartoon: { color: "0x1a1a2e" } });
        viewer.removeAllShapes();
        for (let i = 0; i < embeddings.residues.length; i++) {
          const residue = embeddings.residues[i];
          const t = (scores[i] - sMin) / (sMax - sMin + 1e-8);
          if (t < riskThreshold) {
            viewer.setStyle(
              { resi: residue.residue_index, chain: residue.chain_label },
              { cartoon: { color: "0x1a2a1a", opacity: 0.4 } }
            );
          } else {
            const hex = allostericToHex(Math.max(0, Math.min(1, t)));
            viewer.setStyle(
              { resi: residue.residue_index, chain: residue.chain_label },
              { cartoon: { color: hex } }
            );
          }
          // Glow spheres for top allosteric positions
          if (t > 0.8) {
            viewer.addStyle(
              { resi: residue.residue_index, chain: residue.chain_label, atom: "CA" },
              { sphere: { radius: 1.4, color: "0xcc44ff", opacity: 0.55 } }
            );
          }
        }
      } else if (colorMode === "resistance" && resistanceData && resistanceData.residues.length > 0) {
        // Resistance sensitivity mode: color by per-residue sensitivity score
        const resistanceMap = new Map(
          resistanceData.residues.map((r) => [r.residue_id, r])
        );
        viewer.setStyle({}, { cartoon: { color: "0x1a1a2e" } });
        viewer.removeAllShapes();

        for (const residue of embeddings.residues) {
          const rd = resistanceMap.get(residue.residue_id);
          const score = rd?.sensitivity_score ?? 0;
          if (score < riskThreshold) {
            viewer.setStyle(
              { resi: residue.residue_index, chain: residue.chain_label },
              { cartoon: { color: "0x222233", opacity: 0.4 } }
            );
          } else {
            const hex = resistanceToHex(Math.max(0, Math.min(1, score)));
            viewer.setStyle(
              { resi: residue.residue_index, chain: residue.chain_label },
              { cartoon: { color: hex } }
            );
          }
          // Glow spheres for high_sensitivity residues
          if (rd?.classification === "high_sensitivity") {
            viewer.addStyle(
              { resi: residue.residue_index, chain: residue.chain_label, atom: "CA" },
              { sphere: { radius: 1.5, color: "0xff2222", opacity: 0.6 } }
            );
          }
        }
      } else if (colorMode === "pockets" && pharmacophorePockets && pharmacophorePockets.pockets.length > 0) {
        // Pockets color mode: color residues by pocket membership
        // Non-pocket residues = gray. Pocket residues colored by druggability (green → yellow → red)
        // Centroid spheres rendered for each pocket.
        viewer.setStyle({}, { cartoon: { color: "0x222233", opacity: 0.5 } });
        viewer.removeAllShapes();

        // Build residue → pocket lookup
        const residueToPocket = new Map<string, { pocketIndex: number; druggability: number }>();
        for (const pocket of pharmacophorePockets.pockets) {
          for (const rid of pocket.residue_ids) {
            residueToPocket.set(rid, {
              pocketIndex: pocket.pocket_index,
              druggability: pocket.druggability_score,
            });
          }
        }

        // Color pocket residues
        for (const residue of embeddings.residues) {
          const pocketInfo = residueToPocket.get(residue.residue_id);
          if (pocketInfo) {
            const hex = pocketDruggabilityToHex(pocketInfo.druggability);
            viewer.setStyle(
              { resi: residue.residue_index, chain: residue.chain_label },
              { cartoon: { color: hex } }
            );
          }
        }

        // Centroid spheres for each pocket
        for (const pocket of pharmacophorePockets.pockets) {
          const color = pocketDruggabilityToHex(pocket.druggability_score);
          viewer.addSphere({
            center: { x: pocket.center_x, y: pocket.center_y, z: pocket.center_z },
            radius: Math.max(2.0, Math.cbrt(pocket.volume_estimate) * 0.3),
            color,
            opacity: 0.3,
          });
          // Pocket index label near centroid
          viewer.addLabel(String(pocket.pocket_index), {
            position: { x: pocket.center_x, y: pocket.center_y + 2, z: pocket.center_z },
            fontSize: 10,
            fontColor: "white",
            backgroundOpacity: 0.6,
            backgroundColor: "0x333333",
          });
        }
      } else if (colorMode === "drug_candidates" && drugCandidates && drugCandidates.candidates.length > 0) {
        // Drug candidates mode: color pocket residues by combined_druggability
        // ADMET-passed pockets get brighter coloring. Non-candidate residues muted.
        viewer.setStyle({}, { cartoon: { color: "0x1a1a2e", opacity: 0.35 } });
        viewer.removeAllShapes();

        // Build pocket_index → candidate lookup
        const candidateByPocket = new Map<number, { druggability: number; admetPass: boolean }>();
        for (const c of drugCandidates.candidates) {
          candidateByPocket.set(c.pocket_index, {
            druggability: c.combined_druggability,
            admetPass: c.admet_pass,
          });
        }

        // Color residues by candidate pocket membership
        if (pharmacophorePockets) {
          for (const pocket of pharmacophorePockets.pockets) {
            const candidate = candidateByPocket.get(pocket.pocket_index);
            if (!candidate) continue;
            const baseHex = candidateDruggabilityToHex(candidate.druggability, candidate.admetPass);
            for (const rid of pocket.residue_ids) {
              const residue = embeddings.residues.find((r) => r.residue_id === rid);
              if (residue) {
                viewer.setStyle(
                  { resi: residue.residue_index, chain: residue.chain_label },
                  { cartoon: { color: baseHex }, stick: { color: baseHex, radius: 0.15 } }
                );
              }
            }
          }
        }
      } else {
        const values = embeddings.residues.map((r) =>
          colorMode === "cone_depth"
            ? r.cone_depth
            : colorMode === "epistemic"
            ? r.epistemic_uncertainty
            : r.aleatoric_uncertainty
        );
        const min = Math.min(...values);
        const max = Math.max(...values);
        viewer.setStyle({}, { cartoon: { color: "0x333333" } });
        for (const residue of embeddings.residues) {
          const val =
            colorMode === "cone_depth"
              ? residue.cone_depth
              : colorMode === "epistemic"
              ? residue.epistemic_uncertainty
              : residue.aleatoric_uncertainty;
          const hex = valueToHex(val, min, max);
          viewer.setStyle(
            { resi: residue.residue_index, chain: residue.chain_label },
            { cartoon: { color: hex } }
          );
        }
      }
    } else {
      viewer.setStyle({}, { cartoon: { color: "spectrum" } });
    }

    // Biophysical Radar Mode — now primarily driven by global persisted `isRadarActive` (hoisted in DashboardContext)
    // for cross-interaction / structure-switch persistence + color-mode sync ("allosteric" dropdown).
    // Falls back to legacy directive label for explicit "highlight locks" actions from other panels.
    const isRadarMode = isRadarActive || !!(
      currentDirective &&
      (currentDirective.action === "highlight" || currentDirective.action === "focus") &&
      currentDirective.highlight_groups &&
      currentDirective.highlight_groups.some(g =>
        g.label && (g.label.includes("Allosteric Locks") || g.label.includes("remote control via network"))
      )
    );

    if (isRadarMode && currentDirective && currentDirective.highlight_groups) {
      // Radar Mode: deep ultramarine rest state + amber active network + green target
      // Layer 1: Resting state - everything dark ultramarine/indigo, low opacity ghost
      viewer.setStyle(
        {},
        {
          cartoon: { color: "0x1a2744", opacity: 0.22 },
          stick: { color: "0x1a2744", radius: 0.08, opacity: 0.15 },
        }
      );

      // Layer 2: Active groups (locks in warm amber "burning", pocket in bright green)
      for (const group of currentDirective.highlight_groups) {
        const resiNumbers = group.residue_ids
          .map((id) => parseInt(id.replace(/\D/g, ""), 10))
          .filter((n) => !isNaN(n));
        if (resiNumbers.length > 0) {
          const isLockGroup = group.label && group.label.includes("Allosteric Locks");
          const color = isLockGroup ? "0xf59e0b" : (group.color || "0x4ade80"); // amber for locks, green for target
          if (isLockGroup) {
            // Pseudo-bloom glow layer (larger radius, low opacity) underneath for soft halo
            viewer.setStyle(
              { resi: resiNumbers },
              { stick: { color, radius: 0.4, opacity: 0.3 } }
            );
          }
          viewer.setStyle(
            { resi: resiNumbers },
            {
              cartoon: { color, opacity: 0.95 },
              stick: { color, radius: isLockGroup ? 0.18 : 0.12, opacity: 0.85 },
            }
          );
        }
      }

      // Optional: slight zoom/focus on the active network if focus action
      if (currentDirective.action === "focus" && currentDirective.highlight_groups[0]) {
        const focusResi = currentDirective.highlight_groups[0].residue_ids
          .map((id) => parseInt(id.replace(/\D/g, ""), 10))
          .filter((n) => !isNaN(n));
        if (focusResi.length > 0) {
          viewer.zoomTo({ resi: focusResi });
        }
      }

      // Global radar fallback: if no specific residues in groups (e.g. from Radar toggle), still highlight all locks from allosteric data in amber
      if (currentDirective.highlight_groups.some(g => !g.residue_ids || g.residue_ids.length === 0) && allostericSites?.sites?.[0]?.residue_ids) {
        const lockIds = allostericSites.sites[0].residue_ids;
        const resiNumbers = lockIds.map((id) => parseInt(id.replace(/\D/g, ""), 10)).filter((n) => !isNaN(n));
        if (resiNumbers.length > 0) {
          viewer.setStyle({ resi: resiNumbers }, { stick: { color: "0xf59e0b", radius: 0.4, opacity: 0.3 } });
          viewer.setStyle(
            { resi: resiNumbers },
            { cartoon: { color: "0xf59e0b", opacity: 0.95 }, stick: { color: "0xf59e0b", radius: 0.18, opacity: 0.85 } }
          );
        }
      }
    } else if (
      currentDirective &&
      (currentDirective.action === "highlight" || currentDirective.action === "focus") &&
      currentDirective.highlight_groups &&
      currentDirective.highlight_groups.length > 0
    ) {
      // Standard directive highlights (non-radar)
      for (const group of currentDirective.highlight_groups) {
        const resiNumbers = group.residue_ids
          .map((id) => parseInt(id.replace(/\D/g, ""), 10))
          .filter((n) => !isNaN(n));
        if (resiNumbers.length > 0) {
          viewer.setStyle(
            { resi: resiNumbers },
            {
              cartoon: { color: group.color, opacity: 1.0 },
              stick: { color: group.color, radius: 0.24, opacity: 0.95 },
              sphere: { color: group.color, radius: 0.9, opacity: 0.9 },
            }
          );
        }
      }
      if (currentDirective.action === "focus" && currentDirective.highlight_groups[0]) {
        const focusResi = currentDirective.highlight_groups[0].residue_ids
          .map((id) => parseInt(id.replace(/\D/g, ""), 10))
          .filter((n) => !isNaN(n));
        if (focusResi.length > 0) {
          viewer.zoomTo({ resi: focusResi });
        }
      }
    } else if (highlightResidues.length > 0) {
      const resiNumbers = highlightResidues
        .map((id) => parseInt(id.replace(/\D/g, ""), 10))
        .filter((n) => !isNaN(n));
      if (resiNumbers.length > 0) {
        viewer.setStyle(
          { resi: resiNumbers },
          {
            cartoon: { color: "yellow", opacity: 1.0 },
            stick: { color: "yellow", radius: 0.24, opacity: 0.95 },
            sphere: { color: "yellow", radius: 0.9, opacity: 0.9 },
          }
        );
      }
    }

    viewer.render();
  }, [highlightResidues, activeStructure, currentDirective, colorMode, embeddings, plasticityData, riskThreshold, graphMetrics, resistanceData, pharmacophorePockets, drugCandidates, isRadarActive, allostericSites]);

  // 3Dmol hover callback for tooltip (works in ALL modes)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !activeStructure) {
      setHoveredResidue(null);
      return;
    }

    const hoverCb = (atom: any) => {
      if (!atom || !atom.resi) {
        setHoveredResidue(null);
        return;
      }
      const chain = atom.chain || "A";
      const resn = atom.resn || "???";
      const ss = atom.ss === "h" ? "Helix" : atom.ss === "s" ? "Sheet" : "Coil";

      // Look up embedding data if available
      let coneDepth = 0;
      let uncertainty = 0;
      let risk = 0;
      let residueId = `${chain}:${resn}${atom.resi}`;

      if (embeddings) {
        const idx = embeddings.residues.findIndex(
          (r) => r.residue_index === atom.resi && r.chain_label === chain
        );
        if (idx !== -1) {
          const r = embeddings.residues[idx];
          residueId = r.residue_id;
          coneDepth = r.cone_depth;
          uncertainty = r.epistemic_uncertainty;
          if (plasticityData) {
            risk = plasticityData.risks[idx];
          }
        }
      }

      setHoveredResidue({
        residue_id: residueId,
        residue_index: atom.resi,
        chain_label: chain,
        resn,
        ss,
        cone_depth: coneDepth,
        uncertainty,
        risk,
        x: 0,
        y: 0,
      });
    };

    const unhoverCb = () => setHoveredResidue(null);

    viewer.setHoverable({}, true, hoverCb, unhoverCb);
    return () => {
      try { viewer.setHoverable({}, false, null, null); } catch {}
    };
  }, [activeStructure, embeddings, plasticityData]);

  if (!activeStructure) {
    return (
      <div className="h-full flex items-center justify-center text-zinc-600 text-xs">
        Select a structure for molecular view
      </div>
    );
  }

  const thresholdPercent = Math.round((1 - riskThreshold) * 100);

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between px-2 py-1 shrink-0">
        <span className="text-[10px] uppercase tracking-wider text-zinc-500">
          Molecular Viewer
        </span>
        <div className="flex items-center gap-1">
          <select
            value={colorMode}
            onChange={(e) => setColorMode(e.target.value as StructureColorMode)}
            className="text-[10px] bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-zinc-300"
          >
            <option value="spectrum">Spectrum</option>
            <option value="cone_depth">Cone Depth</option>
            <option value="epistemic">Uncertainty</option>
            <option value="aleatoric">Aleatoric</option>
            <option value="plasticity">Plasticity Risk</option>
            <option value="allosteric">Allosteric Potential</option>
            <option value="resistance" disabled={!resistanceData}>Resistance</option>
            <option value="pockets" disabled={!pharmacophorePockets}>Pockets</option>
            <option value="drug_candidates" disabled={!drugCandidates}>Drug Candidates</option>
          </select>
          <span className="text-[10px] text-zinc-600">
            {activeStructure.pdb_id.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Risk threshold slider — shown in plasticity, allosteric, and resistance modes */}
      {(colorMode === "plasticity" || colorMode === "allosteric" || colorMode === "resistance" || colorMode === "pockets" || colorMode === "drug_candidates") && (
        <div className="flex items-center gap-2 px-2 py-0.5 shrink-0">
          <span className="text-[9px] text-zinc-500 whitespace-nowrap">Threshold</span>
          <input
            type="range"
            min="0"
            max="95"
            value={riskThreshold * 100}
            onChange={(e) => setRiskThreshold(Number(e.target.value) / 100)}
            className={`flex-1 h-1 ${colorMode === "allosteric" ? "accent-purple-500" : colorMode === "resistance" ? "accent-orange-500" : "accent-red-500"}`}
          />
          <span className="text-[9px] text-zinc-400 w-12 text-right">
            top {thresholdPercent}%
          </span>
        </div>
      )}

      <div className="flex-1 min-h-0 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-900/80 z-10">
            <span className="text-xs text-zinc-400">Loading structure…</span>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-900/80 z-10">
            <span className="text-xs text-red-400">{error}</span>
          </div>
        )}
        <div ref={containerRef} className="w-full h-full" />

        {/* Hover tooltip — shows in ALL color modes */}
        {hoveredResidue && (
          <div
            className="absolute z-30 pointer-events-none bg-zinc-900/95 border border-zinc-700 rounded-md p-2.5 shadow-xl text-xs max-w-[260px] font-mono"
            style={{ left: 12, bottom: 12 }}
          >
            <div className="font-bold text-zinc-100 text-[11px]">
              {activeStructure.pdb_id.toUpperCase()}_{hoveredResidue.chain_label}:{hoveredResidue.residue_index} ({hoveredResidue.resn}{hoveredResidue.residue_index})
            </div>
            <div className="text-[10px] text-zinc-500 mt-0.5">
              {hoveredResidue.ss}
              {(() => {
                const fg = inferFunctionalGroup(
                  hoveredResidue.residue_index,
                  hoveredResidue.ss,
                  hoveredResidue.cone_depth,
                  activeStructure.pdb_id
                );
                return fg ? <span className="text-cyan-400 ml-1">· {fg}</span> : null;
              })()}
            </div>

            {(hoveredResidue.cone_depth > 0 || hoveredResidue.uncertainty > 0) && (
              <div className="mt-1.5 space-y-0.5 text-[10px]">
                <div className="flex justify-between">
                  <span className="text-zinc-400">Cone Depth</span>
                  <span className="text-cyan-300">
                    {hoveredResidue.cone_depth.toFixed(2)}{" "}
                    <span className="text-zinc-500">({burialLabel(hoveredResidue.cone_depth)})</span>
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-400">Uncertainty</span>
                  <span className="text-amber-300">
                    {hoveredResidue.uncertainty.toFixed(2)}{" "}
                    <span className="text-zinc-500">({uncertaintyLabel(hoveredResidue.uncertainty)})</span>
                  </span>
                </div>
                {colorMode === "plasticity" && hoveredResidue.risk > 0 && (
                  <div className="flex justify-between items-center pt-0.5 border-t border-zinc-800">
                    <span className="text-zinc-300 font-semibold">Plasticity Risk</span>
                    <span className={
                      hoveredResidue.risk > 0.65
                        ? "text-red-400 font-bold"
                        : hoveredResidue.risk > 0.4
                        ? "text-orange-300 font-semibold"
                        : "text-blue-300"
                    }>
                      {hoveredResidue.risk.toFixed(2)}
                      {hoveredResidue.risk > 0.65 && " ★ HIGH"}
                    </span>
                  </div>
                )}
              </div>
            )}

            {colorMode === "plasticity" && hoveredResidue.risk > 0.4 && plasticityData && (
              <div className="mt-1.5 text-[9px] text-zinc-400 italic leading-tight border-t border-zinc-800 pt-1.5">
                {riskInterpretation(
                  hoveredResidue.risk,
                  (hoveredResidue.cone_depth - plasticityData.dMin) / (plasticityData.dMax - plasticityData.dMin + 1e-8),
                  (hoveredResidue.uncertainty - plasticityData.uMin) / (plasticityData.uMax - plasticityData.uMin + 1e-8)
                )}
              </div>
            )}

            {/* Resistance sensitivity tooltip enrichment */}
            {resistanceData && (() => {
              const rd = resistanceData.residues.find(
                (r) => r.residue_id === hoveredResidue.residue_id
              );
              if (!rd) return null;
              return (
                <div className="mt-1.5 space-y-0.5 text-[10px] border-t border-zinc-800 pt-1.5">
                  <div className="flex justify-between items-center">
                    <span className="text-zinc-400">Sensitivity</span>
                    <span className={
                      rd.classification === "high_sensitivity"
                        ? "text-red-400 font-bold"
                        : rd.classification === "moderate"
                        ? "text-orange-300"
                        : "text-green-300"
                    }>
                      {rd.sensitivity_score.toFixed(3)} ({rd.classification.replace("_", " ")})
                    </span>
                  </div>
                  {rd.classification === "high_sensitivity" && (
                    <div className="text-[9px] text-red-300 font-semibold">
                      ⚠ High Sensitivity — potential resistance mutation hotspot
                    </div>
                  )}
                  {rd.is_hinge && (
                    <div className="text-[9px] text-purple-300">
                      ⛓ Hinge residue (domain boundary)
                    </div>
                  )}
                </div>
              );
            })()}
          </div>
        )}
      </div>
    </div>
  );
}
