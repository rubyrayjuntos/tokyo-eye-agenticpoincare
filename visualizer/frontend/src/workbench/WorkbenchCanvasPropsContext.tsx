import { createContext, useContext, type ReactNode } from "react";

import type { WorkbenchCanvasProps } from "./WorkbenchCanvas";

const WorkbenchCanvasPropsContext = createContext<WorkbenchCanvasProps | null>(null);

export function WorkbenchCanvasPropsProvider({
  value,
  children,
}: {
  value: WorkbenchCanvasProps;
  children: ReactNode;
}) {
  return (
    <WorkbenchCanvasPropsContext.Provider value={value}>
      {children}
    </WorkbenchCanvasPropsContext.Provider>
  );
}

export function useWorkbenchCanvasProps(): WorkbenchCanvasProps {
  const value = useContext(WorkbenchCanvasPropsContext);
  if (!value) {
    throw new Error("useWorkbenchCanvasProps must be used within WorkbenchCanvas");
  }
  return value;
}
