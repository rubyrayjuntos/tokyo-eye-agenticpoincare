import type { SelectedResidueInfo } from "../../lib/types";

/**
 * ResidueContextCard — floating context card showing selected residue metrics
 *
 * Appears between/overlaid on the Poincaré panel when a residue is selected.
 * Gives the visual selection scientific meaning at a glance.
 */

export interface ResidueContextCardProps {
  residue: SelectedResidueInfo;
  /** Additional metrics from hydration if available */
  coneDepth?: number | null;
  epistemicUncertainty?: number | null;
  isDehydron?: boolean | null;
  allostericCoupling?: number | null;
}

function MetricRow({ label, value, unit, color, barFill }: { label: string; value: string | number; unit?: string; color?: string; barFill?: number }) {
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between gap-3">
        <span className="text-[10px] text-text-muted uppercase tracking-wide">{label}</span>
        <span className={`text-xs font-mono tabular-nums ${color ?? "text-text-primary"}`}>
          {typeof value === "number" ? value.toFixed(3) : value}
          {unit && <span className="text-text-muted ml-0.5">{unit}</span>}
        </span>
      </div>
      {barFill != null && (
        <div className="h-1 w-full rounded-full bg-slate overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${Math.min(100, Math.max(0, barFill * 100))}%`,
              background: barFill > 0.7
                ? "linear-gradient(90deg, var(--color-magenta-dim), var(--color-magenta-bright))"
                : barFill > 0.4
                ? "linear-gradient(90deg, var(--color-warning), var(--color-magenta-dim))"
                : "linear-gradient(90deg, var(--color-teal-dim), var(--color-teal))",
            }}
          />
        </div>
      )}
    </div>
  );
}

export function ResidueContextCard({
  residue,
  coneDepth,
  epistemicUncertainty,
  isDehydron,
  allostericCoupling,
}: ResidueContextCardProps) {
  return (
    <div className="residue-card-enter absolute bottom-3 left-3 z-20 w-56 bg-bg-elevated/95 backdrop-blur-sm border border-slate-light rounded-[var(--radius-card)] p-3 shadow-lg">
      {/* Residue identity */}
      <div className="flex items-center gap-2 mb-2 pb-2 border-b border-slate">
        <div className="w-2 h-2 rounded-full bg-teal selection-pulse" />
        <span className="text-xs font-mono text-teal-bright">
          {residue.chain_label ?? "?"}{residue.residue_id}
        </span>
        {residue.residue_name && (
          <span className="text-[10px] text-text-secondary">
            {residue.residue_name}
          </span>
        )}
        {isDehydron && (
          <span className="text-[8px] px-1 py-0.5 rounded bg-magenta-dim/30 text-magenta border border-magenta-dim/50 ml-auto">
            DHR
          </span>
        )}
      </div>

      {/* Metrics */}
      <div className="space-y-1.5">
        {coneDepth != null && (
          <MetricRow label="Cone Depth" value={coneDepth} color="text-teal" barFill={coneDepth / 3} />
        )}
        {epistemicUncertainty != null && (
          <MetricRow
            label="Epistemic σ"
            value={epistemicUncertainty}
            color={epistemicUncertainty > 0.5 ? "text-warning" : "text-text-primary"}
            barFill={epistemicUncertainty}
          />
        )}
        {residue.cone_depth != null && coneDepth == null && (
          <MetricRow label="Cone Depth" value={residue.cone_depth} color="text-teal" barFill={residue.cone_depth / 3} />
        )}
        {residue.epistemic_uncertainty != null && epistemicUncertainty == null && (
          <MetricRow
            label="Epistemic σ"
            value={residue.epistemic_uncertainty}
            color={residue.epistemic_uncertainty > 0.5 ? "text-warning" : "text-text-primary"}
            barFill={residue.epistemic_uncertainty}
          />
        )}
        {allostericCoupling != null && (
          <MetricRow
            label="Allosteric"
            value={allostericCoupling}
            color={allostericCoupling > 0.7 ? "text-magenta" : "text-text-primary"}
            barFill={allostericCoupling}
          />
        )}
      </div>
    </div>
  );
}
