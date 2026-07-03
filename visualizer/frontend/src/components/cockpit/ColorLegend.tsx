import type { ViewportColorMode } from "../../lib/viewportColorMetrics";

const LEGEND_GRADIENTS: Record<ViewportColorMode, string> = {
  spectrum: "linear-gradient(90deg, hsl(0,78%,58%), hsl(120,78%,58%), hsl(300,78%,58%))",
  cone_depth: "linear-gradient(90deg, rgb(10,98,91), rgb(153,240,220))",
  epistemic: "linear-gradient(90deg, rgb(44,49,66), rgb(236,165,58))",
  aleatoric: "linear-gradient(90deg, rgb(44,49,66), rgb(220,92,233))",
  plasticity: "linear-gradient(90deg, rgb(40,80,220), rgb(255,140,30), rgb(255,50,20))",
  allosteric: "linear-gradient(90deg, rgb(20,80,40), rgb(0,220,220), rgb(220,40,255))",
  resistance: "linear-gradient(90deg, rgb(240,240,240), rgb(255,220,100), rgb(180,20,10))",
  pockets: "linear-gradient(90deg, rgb(40,200,80), rgb(240,220,30), rgb(255,60,30))",
  drug_candidates: "linear-gradient(90deg, rgb(40,160,180), rgb(220,200,40), rgb(255,200,30))",
};

const LEGEND_LABELS: Record<ViewportColorMode, { lo: string; hi: string }> = {
  spectrum: { lo: "N-term", hi: "C-term" },
  cone_depth: { lo: "shallow", hi: "deep" },
  epistemic: { lo: "low σ", hi: "high σ" },
  aleatoric: { lo: "low", hi: "high" },
  plasticity: { lo: "stable", hi: "risk" },
  allosteric: { lo: "peripheral", hi: "hub" },
  resistance: { lo: "stable", hi: "sensitive" },
  pockets: { lo: "low W", hi: "druggable" },
  drug_candidates: { lo: "weak", hi: "lead" },
};

export function ColorLegend({ mode }: { mode: ViewportColorMode }) {
  const labels = LEGEND_LABELS[mode];
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-[9px] text-text-muted">{labels.lo}</span>
      <div
        className="h-1.5 w-24 rounded-full"
        style={{ background: LEGEND_GRADIENTS[mode] }}
      />
      <span className="font-mono text-[9px] text-text-secondary">{labels.hi}</span>
    </div>
  );
}
