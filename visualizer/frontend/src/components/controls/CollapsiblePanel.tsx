import { useState } from "react";
import type { ReactNode } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

interface CollapsiblePanelProps {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
  icon?: ReactNode;
  /** Controlled open state (overrides internal state when provided) */
  open?: boolean;
  /** Callback when panel toggle is clicked (used with controlled mode) */
  onToggle?: () => void;
}

/**
 * Collapsible panel wrapper for tool panels in the sidebar.
 * Supports both uncontrolled (defaultOpen) and controlled (open + onToggle) modes.
 */
export default function CollapsiblePanel({
  title,
  children,
  defaultOpen = false,
  icon,
  open: controlledOpen,
  onToggle,
}: CollapsiblePanelProps) {
  const [internalOpen, setInternalOpen] = useState(defaultOpen);
  const isControlled = controlledOpen !== undefined;
  const isOpen = isControlled ? controlledOpen : internalOpen;

  const handleClick = () => {
    if (isControlled && onToggle) {
      onToggle();
    } else {
      setInternalOpen(!internalOpen);
    }
  };

  return (
    <div className="border-b border-zinc-800">
      <button
        onClick={handleClick}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs font-medium text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40 transition-colors"
        aria-expanded={isOpen}
      >
        {isOpen ? (
          <ChevronDown className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
        )}
        {icon && <span className="shrink-0">{icon}</span>}
        <span className="uppercase tracking-wider">{title}</span>
      </button>
      {isOpen && (
        <div className="max-h-[50vh] overflow-y-auto">{children}</div>
      )}
    </div>
  );
}
