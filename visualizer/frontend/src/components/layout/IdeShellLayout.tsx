import type { ReactNode } from "react";
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
} from "react-resizable-panels";
import { GripHorizontal, GripVertical } from "lucide-react";

export interface IdeShellPane {
  id: string;
  title: string;
  content: ReactNode;
  accent?: "teal" | "magenta" | "slate";
}

export interface IdeShellLayoutModel {
  workspace?: number[];
  editor?: number[];
  content?: number[];
}

export interface IdeShellLayoutProps {
  titleBar?: ReactNode;
  header?: ReactNode;
  activityBar?: ReactNode;
  workspaceBar?: ReactNode;
  editorTabs?: ReactNode;
  sidebarPane?: IdeShellPane | null;
  leftPane: IdeShellPane;
  centerPane: IdeShellPane;
  rightPane?: IdeShellPane | null;
  bottomPane?: IdeShellPane | null;
  statusBar?: ReactNode;
  onLayoutModelChange?: (model: IdeShellLayoutModel) => void;
}

const accentClasses: Record<NonNullable<IdeShellPane["accent"]>, string> = {
  teal: "text-teal border-teal-dim/40",
  magenta: "text-magenta border-magenta-dim/40",
  slate: "text-text-secondary border-slate-light",
};

function ResizeHandle({
  direction,
}: {
  direction: "horizontal" | "vertical";
}) {
  return (
    <PanelResizeHandle
      className={
        direction === "horizontal"
          ? "group relative w-1 bg-slate/40 hover:bg-teal-dim/30 transition-colors"
          : "group relative h-1 bg-slate/40 hover:bg-teal-dim/30 transition-colors"
      }
    >
      <div
        className={
          direction === "horizontal"
            ? "absolute inset-y-0 left-1/2 -translate-x-1/2 flex items-center justify-center"
            : "absolute inset-x-0 top-1/2 -translate-y-1/2 flex items-center justify-center"
        }
      >
        {direction === "horizontal" ? (
          <GripVertical size={12} className="text-text-muted opacity-0 group-hover:opacity-100 transition-opacity" />
        ) : (
          <GripHorizontal size={12} className="text-text-muted opacity-0 group-hover:opacity-100 transition-opacity" />
        )}
      </div>
    </PanelResizeHandle>
  );
}

function ShellPane({
  pane,
  className = "",
}: {
  pane: IdeShellPane;
  className?: string;
}) {
  const accent = accentClasses[pane.accent ?? "slate"];

  return (
    <section
      className={`min-h-0 min-w-0 overflow-hidden rounded-[var(--radius-panel)] border bg-bg-surface ${className}`}
    >
      <div className="flex h-full min-h-0 flex-col">
        <div className={`flex shrink-0 items-center gap-2 border-b px-3 py-2 ${accent}`}>
          <div className="h-2 w-2 rounded-full bg-current/80" />
          <span className="text-[10px] font-display uppercase tracking-[0.22em]">
            {pane.title}
          </span>
        </div>
        <div className="flex-1 min-h-0 min-w-0 overflow-hidden">{pane.content}</div>
      </div>
    </section>
  );
}

export function IdeShellLayout({
  titleBar,
  header,
  activityBar,
  workspaceBar,
  editorTabs,
  sidebarPane,
  leftPane,
  centerPane,
  rightPane,
  bottomPane,
  statusBar,
  onLayoutModelChange,
}: IdeShellLayoutProps) {
  const emitContentLayout = (sizes: number[]) => {
    onLayoutModelChange?.({ content: sizes });
  };

  const emitEditorLayout = (sizes: number[]) => {
    onLayoutModelChange?.({ editor: sizes });
  };

  const emitWorkspaceLayout = (sizes: number[]) => {
    onLayoutModelChange?.({ workspace: sizes });
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-bg text-text-primary">
      {activityBar ? (
        <aside className="shrink-0 border-r border-slate bg-bg-surface/95">
          {activityBar}
        </aside>
      ) : null}

      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        {titleBar ? <div className="shrink-0">{titleBar}</div> : null}
        {header ? <header className="shrink-0">{header}</header> : null}
        {workspaceBar ? (
          <div className="shrink-0 border-b border-slate bg-bg-surface/70 px-2 py-1.5">
            {workspaceBar}
          </div>
        ) : null}
        {editorTabs ? (
          <div className="shrink-0 border-b border-slate bg-bg-surface/90 px-2">
            {editorTabs}
          </div>
        ) : null}

        <PanelGroup
          direction="vertical"
          autoSaveId="tokyo-eye-shell-workspace"
          className="min-h-0 flex-1"
          onLayout={emitWorkspaceLayout}
        >
          <Panel defaultSize={78} minSize={45}>
            <PanelGroup
              direction="horizontal"
              autoSaveId="tokyo-eye-shell-editor"
              className="min-h-0"
              onLayout={emitEditorLayout}
            >
              {sidebarPane ? (
                <>
                  <Panel defaultSize={18} minSize={12} maxSize={30}>
                    <ShellPane pane={sidebarPane} />
                  </Panel>
                  <ResizeHandle direction="horizontal" />
                </>
              ) : null}

              <Panel defaultSize={sidebarPane ? 82 : 100} minSize={35}>
                <PanelGroup
                  direction="horizontal"
                  autoSaveId="tokyo-eye-shell-content"
                  className="min-h-0"
                  onLayout={emitContentLayout}
                >
                  <Panel defaultSize={24} minSize={18} maxSize={40}>
                    <ShellPane pane={leftPane} />
                  </Panel>
                  <ResizeHandle direction="horizontal" />
                  <Panel defaultSize={rightPane ? 48 : 76} minSize={28}>
                    <ShellPane pane={centerPane} />
                  </Panel>
                  {rightPane ? (
                    <>
                      <ResizeHandle direction="horizontal" />
                      <Panel defaultSize={28} minSize={18} maxSize={40}>
                        <ShellPane pane={rightPane} />
                      </Panel>
                    </>
                  ) : null}
                </PanelGroup>
              </Panel>
            </PanelGroup>
          </Panel>

          {bottomPane ? (
            <>
              <ResizeHandle direction="vertical" />
              <Panel defaultSize={22} minSize={12} maxSize={45}>
                <ShellPane pane={bottomPane} className="rounded-none border-x-0 border-b-0" />
              </Panel>
            </>
          ) : null}
        </PanelGroup>
        {statusBar ? <div className="shrink-0">{statusBar}</div> : null}
      </div>
    </div>
  );
}
