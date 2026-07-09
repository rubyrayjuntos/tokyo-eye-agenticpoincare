/** Dockview wrapper for Model Lifecycle (expanded). */
import ModelLifecyclePanel from "../panels/ModelLifecyclePanel";
import type { PanelManifest } from "../workbench/panelContract";

const MANIFEST: PanelManifest<"model_lifecycle"> = {
  id: "model_lifecycle",
  title: "Model Lifecycle",
  region: "activity-sidebar",
  ports: [
    { name: "open-panel", kind: "open-panel", direction: "output" },
  ],
  emits: ["panel.activated"],
};

export default function ModelLifecycleDockPanel({
  structureId,
}: {
  structureId?: string | null;
}) {
  return (
    <div className="h-full overflow-auto">
      <ModelLifecyclePanel manifest={MANIFEST} structureId={structureId} />
    </div>
  );
}
