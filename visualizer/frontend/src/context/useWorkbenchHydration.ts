import { useHydration } from "./HydrationProvider";

/** @deprecated Use useHydration — bus overlay is built into the unified hook. */
export function useWorkbenchHydration() {
  return useHydration();
}