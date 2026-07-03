import { parseResidueLocator } from "./residueId";
import type { ResidueEmbedding } from "./types";
import {
  metricValueToColor,
  type ResidueMetricSeries,
  type ViewportColorMode,
} from "./viewportColorMetrics";

const COLOR_QUANT_STEPS = 24;

function cssColorToHex(color: string): number {
  if (typeof document === "undefined") return 0x888888;
  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 1;
  const ctx = canvas.getContext("2d");
  if (!ctx) return 0x888888;
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, 1, 1);
  const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
  return (r << 16) | (g << 8) | b;
}

function quantizeMetric(value: number, min: number, max: number): number {
  const t = (value - min) / (max - min || 1);
  return Math.max(
    0,
    Math.min(COLOR_QUANT_STEPS - 1, Math.round(t * (COLOR_QUANT_STEPS - 1))),
  );
}

interface MolstarColorTarget {
  chain: string;
  seq: number;
}

interface MolstarStructureLib {
  Queries: { generators: { atoms: (q: unknown) => (ctx: unknown) => unknown } };
  QueryContext: new (structure: unknown) => unknown;
  StructureProperties: {
    chain: { auth_asym_id: (el: unknown) => string };
    residue: { auth_seq_id: (el: unknown) => number };
  };
}

interface MolstarComponentManager {
  updateRepresentationsTheme: (
    components: unknown[],
    params: Record<string, unknown>,
  ) => Promise<void>;
  applyTheme: (params: {
    selection: {
      getSelection: (
        plugin: unknown,
        ctx: unknown,
        structure: unknown,
      ) => Promise<unknown>;
    };
    action:
      | { name: "color"; params: { color: number } }
      | { name: "resetColor"; params: Record<string, never> };
  }) => Promise<void>;
}

function getStructureLib(): MolstarStructureLib | null {
  return (
    (window.molstar as { lib?: { structure?: MolstarStructureLib } })?.lib
      ?.structure ?? null
  );
}

function getPolymerComponent(structures: Array<{ components?: unknown[] }>) {
  const components = structures[0]?.components;
  if (!components?.length) return null;
  return (
    components.find(
      (component) =>
        (component as { cell?: { obj?: { label?: string } } }).cell?.obj?.label ===
        "Polymer",
    ) ?? components[0]
  );
}

async function resetMolstarOverpaint(
  componentManager: MolstarComponentManager,
  S: MolstarStructureLib,
): Promise<void> {
  await componentManager.applyTheme({
    selection: {
      getSelection: async (_plugin, _ctx, structure) => {
        const query = S.Queries.generators.atoms({});
        return query(new S.QueryContext(structure));
      },
    },
    action: { name: "resetColor", params: {} },
  });
}

export async function applyMolstarResidueColors(
  plugin: {
    managers: {
      structure: {
        hierarchy: { current: { structures: Array<{ components?: unknown[] }> } };
        component: MolstarComponentManager;
      };
    };
  },
  colorMode: ViewportColorMode,
  residues: ResidueEmbedding[],
  metricSeries: ResidueMetricSeries,
): Promise<void> {
  const S = getStructureLib();
  if (!S) return;

  const structures = plugin.managers.structure.hierarchy.current.structures;
  const polymer = getPolymerComponent(structures);
  if (!polymer) return;

  const componentManager = plugin.managers.structure.component;
  const themeComponents = [polymer];

  await resetMolstarOverpaint(componentManager, S);

  if (colorMode === "spectrum") {
    await componentManager.updateRepresentationsTheme(themeComponents, {
      color: "sequence-id",
    });
    return;
  }

  await componentManager.updateRepresentationsTheme(themeComponents, {
    color: "uniform",
    colorParams: { value: 0x1a1a2e },
  });

  if (!residues.length) return;

  const groups = new Map<number, MolstarColorTarget[]>();

  for (const residue of residues) {
    const locator = parseResidueLocator(residue.residue_id);
    if (!locator) continue;
    const metric = metricSeries.values.get(residue.residue_id) ?? 0;
    const bucket = quantizeMetric(metric, metricSeries.min, metricSeries.max);
    const sampleValue =
      metricSeries.min +
      (bucket / Math.max(COLOR_QUANT_STEPS - 1, 1)) *
        (metricSeries.max - metricSeries.min || 1);
    const rgb = metricValueToColor(
      sampleValue,
      colorMode,
      metricSeries.min,
      metricSeries.max,
    );
    const hex = cssColorToHex(rgb);
    const targets = groups.get(hex) ?? [];
    targets.push({ chain: locator.chain, seq: locator.seq });
    groups.set(hex, targets);
  }

  for (const [hex, targets] of groups) {
    await componentManager.applyTheme({
      selection: {
        getSelection: async (_plugin, _ctx, structure) => {
          const query = S.Queries.generators.atoms({
            residueTest: (ctx: { element: unknown }) => {
              const chain = S.StructureProperties.chain.auth_asym_id(ctx.element);
              const seq = S.StructureProperties.residue.auth_seq_id(ctx.element);
              return targets.some(
                (target) => target.chain === chain && target.seq === seq,
              );
            },
          });
          return query(new S.QueryContext(structure));
        },
      },
      action: { name: "color", params: { color: hex } },
    });
  }
}
