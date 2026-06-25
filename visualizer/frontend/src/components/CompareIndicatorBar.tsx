/**
 * CompareIndicatorBar — shows when compare mode is active.
 * Displays both PDB IDs and an "Exit Compare" button.
 * Requirements: 1.2, 1.3
 */
import { useDashboard } from "../lib/context";

export default function CompareIndicatorBar() {
  const { activeStructure, compareState, exitCompareMode } = useDashboard();

  if (!compareState.active || !compareState.secondaryStructure) return null;

  return (
    <div className="flex items-center gap-3 px-3 py-1.5 bg-amber-900/20 border border-amber-700/40 rounded-md text-xs">
      <span className="text-amber-300 font-medium">Compare Mode</span>
      <span className="text-zinc-300">
        <span className="font-mono text-cyan-300">
          {activeStructure?.pdb_id ?? "—"}
        </span>
        {" ⇔ "}
        <span className="font-mono text-cyan-300">
          {compareState.secondaryStructure.pdb_id}
        </span>
      </span>
      {compareState.loading && (
        <span className="text-zinc-500">Loading...</span>
      )}
      <button
        onClick={exitCompareMode}
        className="ml-auto px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:text-red-300 hover:bg-zinc-700 transition-colors"
      >
        Exit Compare
      </button>
    </div>
  );
}
