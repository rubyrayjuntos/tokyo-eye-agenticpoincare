import type { ResidueEmbedding } from "../../lib/types";
import { formatResiduePosition } from "../../lib/residueLabels";

export function ResidueReadout({ residue }: { residue: ResidueEmbedding | null }) {
  if (!residue) {
    return (
      <div className="pointer-events-none rounded-md border border-slate/60 bg-bg/80 px-3 py-2 text-[10px] text-text-muted backdrop-blur-sm">
        Hover or select a residue on the disc
      </div>
    );
  }

  return (
    <div className="pointer-events-none rounded-md border border-slate/60 bg-bg/90 px-3 py-2 backdrop-blur-sm">
      <div className="mb-2 flex items-center gap-2">
        <span className="font-mono text-sm text-text-primary">{residue.residue_id}</span>
        <span className="rounded-full border border-teal-dim/40 px-2 py-0.5 text-[9px] uppercase tracking-[0.12em] text-teal">
          {formatResiduePosition(residue)}
        </span>
      </div>
      <div className="flex flex-wrap gap-4">
        <Metric label="depth" value={residue.cone_depth.toFixed(3)} color="text-teal" />
        <Metric label="e σ" value={residue.epistemic_uncertainty.toFixed(3)} color="text-warning" />
        <Metric label="a σ" value={residue.aleatoric_uncertainty.toFixed(3)} color="text-magenta" />
        <Metric
          label="r"
          value={Math.hypot(residue.x, residue.y).toFixed(3)}
          color="text-text-secondary"
        />
      </div>
    </div>
  );
}

function Metric({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div>
      <div className="text-[8px] uppercase tracking-[0.1em] text-text-muted">{label}</div>
      <div className={`mt-0.5 font-mono text-xs ${color}`}>{value}</div>
    </div>
  );
}
