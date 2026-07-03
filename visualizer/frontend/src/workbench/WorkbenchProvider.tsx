import {
  createContext,
  useContext,
  useMemo,
  type PropsWithChildren,
} from "react";

import { LayoutEngineAdapter } from "./LayoutEngineAdapter";
import { WorkbenchBus } from "./WorkbenchBus";

interface WorkbenchContextValue {
  bus: WorkbenchBus;
  layoutAdapter: LayoutEngineAdapter;
}

export const WorkbenchContext = createContext<WorkbenchContextValue | null>(null);

export function WorkbenchProvider({ children }: PropsWithChildren) {
  const value = useMemo(() => {
    const bus = new WorkbenchBus();
    const layoutAdapter = new LayoutEngineAdapter(bus);
    return { bus, layoutAdapter };
  }, []);

  return (
    <WorkbenchContext.Provider value={value}>{children}</WorkbenchContext.Provider>
  );
}

export function useWorkbenchBus(): WorkbenchBus {
  const context = useContext(WorkbenchContext);
  if (!context) {
    throw new Error("useWorkbenchBus must be used within WorkbenchProvider");
  }
  return context.bus;
}

export function useLayoutEngineAdapter(): LayoutEngineAdapter {
  const context = useContext(WorkbenchContext);
  if (!context) {
    throw new Error("useLayoutEngineAdapter must be used within WorkbenchProvider");
  }
  return context.layoutAdapter;
}

export function useWorkbench(): WorkbenchContextValue {
  const context = useContext(WorkbenchContext);
  if (!context) {
    throw new Error("useWorkbench must be used within WorkbenchProvider");
  }
  return context;
}
