export interface MobiusStripProps {
  mobiusX: number;
  mobiusY: number;
  onReset: () => void;
}

export function MobiusStrip({ mobiusX, mobiusY, onReset }: MobiusStripProps) {
  return (
    <div className="flex h-[42px] shrink-0 items-center gap-4 border-t border-slate bg-bg-surface px-4">
      <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-text-muted">
        Möbius focus
      </span>
      <span className="text-[11px] text-text-secondary">
        Drag the disc background to recenter the geodesic
      </span>
      <div className="flex items-center gap-2 font-mono text-[10px]">
        <span className="text-text-muted">a =</span>
        <span className="w-12 text-right text-teal">{mobiusX.toFixed(3)}</span>
        <span className="text-text-muted">+</span>
        <span className="w-12 text-teal">{mobiusY.toFixed(3)}i</span>
      </div>
      <button
        type="button"
        onClick={onReset}
        className="rounded border border-slate-light bg-bg-elevated px-3 py-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-text-secondary hover:border-teal-dim"
      >
        Reset
      </button>
      <span className="ml-auto font-mono text-[9px] text-text-muted">conformal · angle-preserving</span>
    </div>
  );
}
