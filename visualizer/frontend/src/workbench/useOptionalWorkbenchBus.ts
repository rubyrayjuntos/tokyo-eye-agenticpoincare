import { useContext } from "react";

import { WorkbenchContext } from "./WorkbenchProvider";
import type { WorkbenchBus } from "./WorkbenchBus";

/** Returns the workbench bus when inside WorkbenchProvider; undefined in legacy shells. */
export function useOptionalWorkbenchBus(): WorkbenchBus | undefined {
  const context = useContext(WorkbenchContext);
  return context?.bus;
}