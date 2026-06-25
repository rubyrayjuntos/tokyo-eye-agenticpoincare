import { CheckCircle, AlertTriangle } from "lucide-react";

interface PersistenceIndicatorProps {
  /** Whether the data for this panel has been persisted */
  persisted: boolean;
  /** Optional label to display next to the indicator */
  label?: string;
}

/**
 * Shows a green checkmark when data is persisted, red warning when not.
 * Reads from hydration persistence_status via props.
 */
export default function PersistenceIndicator({
  persisted,
  label,
}: PersistenceIndicatorProps) {
  return (
    <span
      className="inline-flex items-center gap-1 text-xs"
      title={persisted ? "Data persisted" : "Data not persisted"}
      aria-label={persisted ? "Data persisted" : "Data not persisted"}
    >
      {persisted ? (
        <CheckCircle className="w-3.5 h-3.5 text-emerald-400" aria-hidden="true" />
      ) : (
        <AlertTriangle className="w-3.5 h-3.5 text-red-400" aria-hidden="true" />
      )}
      {label && (
        <span className={persisted ? "text-emerald-400" : "text-red-400"}>
          {label}
        </span>
      )}
    </span>
  );
}
