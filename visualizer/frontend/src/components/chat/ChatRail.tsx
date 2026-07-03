import { useState, useRef, useEffect, useCallback } from "react";
import type React from "react";
import { Send, MessageSquare } from "lucide-react";
import { api } from "../../lib/api";
import type {
  AgentChatRequest,
  AgentChatResponse,
  ViewportState,
  ViewportDirective,
  SelectedResidueInfo,
  PoincareColorMode,
  StructureColorModeType,
} from "../../lib/types";
import type { DiscoveryPhase } from "../../lib/discoveryPhaseMachine";
import type { HypothesisLifecycleState } from "../../lib/hypothesisLifecycleMachine";

/**
 * ChatRail — Discovery Cockpit chat panel with viewport context enrichment
 *
 * Sends viewport_state with each message. Renders directive annotations inline.
 * Shows discovery phase and lifecycle as contextual badges.
 * Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
 */

export interface ChatRailProps {
  structureId: string | null;
  pdbId: string | null;
  discoveryPhase: DiscoveryPhase;
  hypothesisLifecycle: HypothesisLifecycleState;
  viewportStateBuilder: () => ViewportState;
  sessionId?: string | null;
  onDirective?: (directive: ViewportDirective) => void;
  onSessionId?: (sessionId: string) => void;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  directives?: ViewportDirective[];
}

/** Simple markdown rendering for agent responses */
function renderMarkdown(text: string): string {
  return text
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="bg-bg-elevated rounded-[var(--radius-card)] p-2 text-xs overflow-x-auto my-1 font-mono"><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code class="bg-bg-elevated px-1 rounded text-xs font-mono">$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br/>");
}

/** Format directive action for display */
function directiveLabel(d: ViewportDirective): string {
  switch (d.action) {
    case "highlight":
      return `Highlight ${d.highlight_groups?.flatMap((g) => g.residue_ids).length ?? 0} residues`;
    case "focus":
      return `Focus on ${d.focus_residues?.join(", ") ?? "residue"}`;
    case "set_metric":
      return `Set metric → ${d.metric ?? d.color_mode ?? "unknown"}`;
    case "clear":
      return "Clear highlights";
    default:
      return `${d.action}`;
  }
}

export function ChatRail({
  structureId,
  pdbId,
  discoveryPhase,
  hypothesisLifecycle,
  viewportStateBuilder,
  sessionId: externalSessionId,
  onDirective,
  onSessionId,
}: ChatRailProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const fallbackSessionIdRef = useRef(crypto.randomUUID());
  const sessionId = externalSessionId ?? fallbackSessionIdRef.current;
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const publishSessionId = useCallback(
    (nextSessionId: string) => {
      onSessionId?.(nextSessionId);
    },
    [onSessionId],
  );

  // Bootstrap parent session only when the parent does not already own one.
  useEffect(() => {
    if (externalSessionId == null) {
      publishSessionId(sessionId);
    }
  }, [externalSessionId, publishSessionId, sessionId]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = useCallback(async () => {
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setInput("");
    setLoading(true);

    try {
      const viewportState = viewportStateBuilder();
      const req: AgentChatRequest = {
        message: trimmed,
        session_id: sessionId,
        context: viewportState,
      };

      const res: AgentChatResponse = await api.chat(req);
      publishSessionId(res.session_id ?? sessionId);

      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: res.response,
        directives: res.viewport_directives,
      };
      setMessages((prev) => [...prev, assistantMsg]);

      // Apply viewport directives (inspect-preview-commit: shown in chat before execution)
      if (res.viewport_directives?.length && onDirective) {
        for (const directive of res.viewport_directives) {
          onDirective(directive);
        }
      }
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `⚠️ ${e.message || "Failed to get response"}` },
      ]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, sessionId, viewportStateBuilder, onDirective, publishSessionId]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header with phase/lifecycle badges */}
      <div className="shrink-0 flex items-center justify-between px-3 py-2 border-b border-slate">
        <div className="flex items-center gap-2">
          <MessageSquare size={14} className="text-teal" />
          <span className="text-xs font-body text-text-primary">Agent</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[9px] uppercase tracking-wide px-1.5 py-0.5 rounded-[var(--radius-badge)] bg-bg-elevated text-text-muted border border-slate-light">
            {discoveryPhase}
          </span>
          <span className="text-[9px] uppercase tracking-wide px-1.5 py-0.5 rounded-[var(--radius-badge)] bg-bg-elevated text-text-muted border border-slate-light">
            {hypothesisLifecycle}
          </span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3 min-h-0">
        {messages.length === 0 && (
          <div className="text-xs text-text-muted text-center mt-8 px-4">
            {structureId
              ? `Ask about ${pdbId?.toUpperCase() ?? structureId}. Viewport context is sent with each message.`
              : "Load a structure to enable context-aware conversation."}
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i}>
            <div className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[90%] rounded-[var(--radius-card)] px-3 py-2 text-xs leading-relaxed ${
                  msg.role === "user"
                    ? "bg-teal-dim/20 text-teal-bright border border-teal-dim/40"
                    : "bg-bg-elevated text-text-primary border border-slate-light"
                }`}
                dangerouslySetInnerHTML={
                  msg.role === "assistant"
                    ? { __html: renderMarkdown(msg.content) }
                    : undefined
                }
              >
                {msg.role === "user" ? msg.content : undefined}
              </div>
            </div>

            {/* Directive annotations (inspect-preview-commit) */}
            {msg.directives && msg.directives.length > 0 && (
              <div className="ml-2 mt-1 space-y-0.5">
                {msg.directives.map((d, di) => (
                  <div
                    key={di}
                    className="flex items-center gap-1.5 text-[10px] text-magenta px-2 py-0.5 rounded bg-magenta-dim/10 border border-magenta-dim/30 w-fit"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-magenta" />
                    {directiveLabel(d)}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        {/* Loading indicator */}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-bg-elevated border border-slate-light rounded-[var(--radius-card)] px-3 py-2">
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-teal animate-pulse" />
                <span className="w-1.5 h-1.5 rounded-full bg-teal animate-pulse [animation-delay:0.2s]" />
                <span className="w-1.5 h-1.5 rounded-full bg-teal animate-pulse [animation-delay:0.4s]" />
                <span className="text-[10px] text-text-muted ml-1">Analyzing…</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="shrink-0 p-2 border-t border-slate">
        <div className="flex items-center gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={structureId ? "Ask about this structure…" : "Load a structure first…"}
            disabled={loading}
            className="flex-1 bg-bg-elevated border border-slate-light rounded-[var(--radius-button)] px-3 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-teal disabled:opacity-50 font-body"
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="p-1.5 rounded-[var(--radius-button)] bg-teal-dim/30 text-teal hover:bg-teal-dim/50 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            aria-label="Send message"
          >
            <Send size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
