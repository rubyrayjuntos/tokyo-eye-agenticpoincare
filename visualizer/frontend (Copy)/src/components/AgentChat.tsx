import { useState, useRef, useEffect, useCallback } from "react";
import type React from "react";
import { MessageSquare, X, Send } from "lucide-react";
import { useDashboard } from "../lib/context";
import { useHydration } from "../context/HydrationProvider";
import { useState as useReactState } from "react"; // if needed for local, but we'll pull from dashboard
import { api } from "../lib/api";
import type {
  AgentChatRequest,
  AgentChatResponse,
  ViewportState,
  ViewportStateProps,
} from "../lib/types";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

/** Parse residue references from agent response text (e.g. "residue A123", "Res 45", "GLY-78") */
function parseResidueReferences(text: string): string[] {
  const patterns = [
    /\b(?:residue|res|resi)\s*[A-Z]?(\d+)/gi,
    /\b[A-Z]{3}[- ]?(\d+)\b/g,
    /\b(?:position|pos)\s*(\d+)\b/gi,
  ];
  const residues = new Set<string>();
  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(text)) !== null) {
      residues.add(match[1]);
    }
  }
  return Array.from(residues);
}

/** Simple markdown renderer for agent responses */
function renderMarkdown(text: string): string {
  return text
    // Code blocks
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="bg-zinc-800 rounded p-2 text-xs overflow-x-auto my-1"><code>$2</code></pre>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code class="bg-zinc-800 px-1 rounded text-xs">$1</code>')
    // Bold
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    // Italic
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    // Line breaks
    .replace(/\n/g, "<br/>");
}

export default function AgentChat(props: Partial<ViewportStateProps>) {
  const {
    poincareColorMode = "cone_depth",
    poincareSelectedResidue = null,
    mobiusFocus = false,
    brushSelection = [],
    viewerColorMode = "spectrum",
    riskThreshold = 0,
    activePanel = null,
  } = props;

  const {
    activeStructure,
    chatOpen,
    setChatOpen,
    setHighlightedResidues,
    emitDirective,
    highlightedResidues,
    compareState,
    selectedPocketId,
    isRadarActive,
    setLatestAgentTelemetry,
    setAgentSessionId,
  } =
    useDashboard();
  const {
    hydration,
    sourceLeaks,
    hypotheses,
    provenanceRuns,
    annotations,
    resistanceData,
    persistenceStatus,
    embeddings,
    pharmacophorePockets,
    drugCandidates,
  } = useHydration();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(() => crypto.randomUUID());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setAgentSessionId(sessionId);
  }, [sessionId, setAgentSessionId]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Focus input when chat opens
  useEffect(() => {
    if (chatOpen) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [chatOpen]);

  const buildDataInspectorContext = useCallback(() => {
    const phasesComputed: string[] = [];
    const phaseCounts: Record<string, number> = {};

    if (embeddings?.residues?.length) {
      phasesComputed.push("embeddings");
      phaseCounts.embeddings = embeddings.residues.length;
    }
    if (sourceLeaks?.leaks?.length) {
      phasesComputed.push("source_leaks");
      phaseCounts.source_leaks = sourceLeaks.leaks.length;
    }
    if (resistanceData?.residues?.length) {
      phasesComputed.push("resistance");
      phaseCounts.resistance = resistanceData.residues.length;
    }
    if (pharmacophorePockets?.pockets?.length) {
      phasesComputed.push("phase5");
      phaseCounts.phase5 = pharmacophorePockets.pockets.length;
    }
    if (drugCandidates?.candidates?.length) {
      phasesComputed.push("phase6");
      phaseCounts.phase6 = drugCandidates.candidates.length;
    }

    // Top druggability pocket score
    const topDruggabilityPocket = pharmacophorePockets?.pockets?.length
      ? Math.max(...pharmacophorePockets.pockets.map((p) => p.druggability_score))
      : null;

    // Top drug candidate score
    const topDrugCandidateScore = drugCandidates?.candidates?.length
      ? Math.max(...drugCandidates.candidates.map((c) => c.combined_druggability))
      : null;

    // ADMET pass rate
    let admetPassRate: number | null = null;
    if (drugCandidates?.candidates?.length) {
      const passed = drugCandidates.candidates.filter((c) => c.admet_pass).length;
      admetPassRate = passed / drugCandidates.candidates.length;
    }

    return {
      phases_computed: phasesComputed,
      phase_counts: phaseCounts,
      top_druggability_pocket: topDruggabilityPocket,
      top_drug_candidate_score: topDrugCandidateScore,
      admet_pass_rate: admetPassRate,
    };
  }, [embeddings, sourceLeaks, resistanceData, pharmacophorePockets, drugCandidates]);

  const buildContext = useCallback((): ViewportState => {
    // Build top uncertainty residues from embeddings
    const topUncertaintyResidues = embeddings?.residues
      ? [...embeddings.residues]
          .sort((a, b) => b.epistemic_uncertainty - a.epistemic_uncertainty)
          .slice(0, 5)
          .map((r) => ({
            residue_id: r.residue_id,
            epistemic_uncertainty: r.epistemic_uncertainty,
          }))
      : [];

    // Build hypothesis status distribution
    let hypothesisStatusDistribution: Record<string, number> | null = null;
    if (hypotheses && hypotheses.length > 0) {
      hypothesisStatusDistribution = {};
      for (const h of hypotheses) {
        hypothesisStatusDistribution[h.status] =
          (hypothesisStatusDistribution[h.status] || 0) + 1;
      }
    }

    // Build resistance summary
    let resistanceSummary: ViewportState["data_summary"] extends { resistance_summary: infer R } ? R : never = null;
    if (resistanceData) {
      resistanceSummary = {
        lambda_2: resistanceData.spectral.lambda_2,
        hinge_count: resistanceData.spectral.hinge_count,
        high_sensitivity_count: resistanceData.residues.filter(
          (r) => r.classification === "high_sensitivity"
        ).length,
        moderate_count: resistanceData.residues.filter(
          (r) => r.classification === "moderate"
        ).length,
        stable_count: resistanceData.residues.filter(
          (r) => r.classification === "stable"
        ).length,
      };
    }

    // Build persistence status from hydration
    const persistenceStatusMap: Record<string, boolean> = persistenceStatus
      ? {
          embeddings: persistenceStatus.embeddings_persisted,
          graph: persistenceStatus.graph_persisted,
          sites: persistenceStatus.sites_persisted,
          resistance: resistanceData != null,
        }
      : {};

    // Build data_summary (null when hydration hasn't loaded)
    const dataSummary: ViewportState["data_summary"] = hydration
      ? {
          residue_count: embeddings?.residues?.length ?? 0,
          source_leak_count: sourceLeaks?.leaks?.length ?? 0,
          hypothesis_count: hypotheses?.length ?? 0,
          hypothesis_status_distribution: hypothesisStatusDistribution,
          provenance_run_count: provenanceRuns?.length ?? 0,
          annotation_count: annotations?.length ?? 0,
          top_uncertainty_residues: topUncertaintyResidues,
          persistence_status: persistenceStatusMap,
          resistance_summary: resistanceSummary,
        }
      : null;

    return {
      structure_id: activeStructure?.structure_id ?? null,
      structure_title: activeStructure?.title ?? null,
      is_radar_active: isRadarActive ?? false,
      selected_pocket_id: selectedPocketId ?? null,

      poincare: {
        color_mode: poincareColorMode,
        mobius_focus_enabled: mobiusFocus,
        mobius_focus_residue: mobiusFocus && poincareSelectedResidue
          ? poincareSelectedResidue.residue_id
          : null,
        selected_residue: poincareSelectedResidue,
        brush_selected_ids: brushSelection,
      },

      viewer_3d: {
        color_mode: viewerColorMode,
        risk_threshold: riskThreshold,
        highlighted_residue_ids: highlightedResidues,
      },

      data_summary: dataSummary,

      active_panel: activePanel,

      pipeline: {
        status: activeStructure?.last_run_id
          ? "complete"
          : activeStructure?.has_embeddings
            ? "complete"
            : "never_run",
        current_step: null,
        progress: null,
      },

      compare: compareState.active && compareState.secondaryStructure
        ? {
            secondary_structure_id: compareState.secondaryStructure.structure_id,
            top_movers: compareState.displacements
              ? compareState.displacements.displacements
                  .slice(0, 10)
                  .map((d) => ({ residue_id: d.residue_id, displacement: d.displacement }))
              : [],
            edge_diff: compareState.graphDiff
              ? {
                  gained: compareState.graphDiff.edge_diff.gained_count,
                  lost: compareState.graphDiff.edge_diff.lost_count,
                  changed: compareState.graphDiff.edge_diff.changed_count,
                }
              : { gained: 0, lost: 0, changed: 0 },
          }
        : null,

      data_inspector: activePanel === "data_inspector"
        ? buildDataInspectorContext()
        : null,
    };
  }, [
    activeStructure,
    poincareColorMode,
    poincareSelectedResidue,
    mobiusFocus,
    brushSelection,
    viewerColorMode,
    riskThreshold,
    activePanel,
    highlightedResidues,
    hydration,
    embeddings,
    sourceLeaks,
    hypotheses,
    provenanceRuns,
    annotations,
    resistanceData,
    persistenceStatus,
    compareState,
    pharmacophorePockets,
    drugCandidates,
  ]);

  const sendMessage = useCallback(async () => {
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    const userMsg: ChatMessage = { role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const req: AgentChatRequest = {
        message: trimmed,
        session_id: sessionId,
        context: buildContext(),
      };

      const res: AgentChatResponse = await api.chat(req);
      setLatestAgentTelemetry(res.telemetry ?? null);
      setAgentSessionId(res.session_id ?? sessionId);

      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: res.response,
      };
      setMessages((prev) => [...prev, assistantMsg]);

      // Apply viewport directives from agent response
      if (res.viewport_directives && res.viewport_directives.length > 0) {
        for (const directive of res.viewport_directives) {
          emitDirective(directive);
        }
      }

      // Parse residue references and emit highlight events
      const residuesFromResponse = res.referenced_residues ?? [];
      const parsedResidues = parseResidueReferences(res.response);
      const allResidues = [
        ...new Set([...residuesFromResponse, ...parsedResidues]),
      ];
      if (allResidues.length > 0) {
        setHighlightedResidues(allResidues);
      }
    } catch (e: any) {
      const errorMsg: ChatMessage = {
        role: "assistant",
        content: `⚠️ Error: ${e.message || "Failed to get response"}`,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  }, [
    input,
    loading,
    sessionId,
    buildContext,
    setHighlightedResidues,
    emitDirective,
    setLatestAgentTelemetry,
    setAgentSessionId,
  ]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  if (!chatOpen) return null;

  return (
    <div className="w-80 border-l border-zinc-800 bg-zinc-900/95 backdrop-blur-sm flex flex-col shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <MessageSquare size={14} className="text-cyan-400" />
          <span className="text-xs font-medium text-zinc-200">
            Agent Analyst
          </span>
        </div>
        <button
          onClick={() => setChatOpen(false)}
          className="p-1 rounded hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 transition-colors"
          aria-label="Close chat"
        >
          <X size={14} />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3 min-h-0">
        {messages.length === 0 && (
          <div className="text-xs text-zinc-500 text-center mt-8">
            Ask questions about your active structure and results. The agent
            provides scientific interpretation without executing pipelines.
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[90%] rounded-lg px-3 py-2 text-xs leading-relaxed ${
                msg.role === "user"
                  ? "bg-cyan-900/40 text-cyan-100 border border-cyan-800/30"
                  : "bg-zinc-800/60 text-zinc-200 border border-zinc-700/30"
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
        ))}

        {/* Thinking indicator */}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-zinc-800/60 border border-zinc-700/30 rounded-lg px-3 py-2">
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse [animation-delay:0.2s]" />
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse [animation-delay:0.4s]" />
                <span className="text-[10px] text-zinc-500 ml-1">
                  Analyzing…
                </span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Context indicator */}
      {activeStructure && (
        <div className="px-3 py-1 border-t border-zinc-800/50">
          <span className="text-[10px] text-zinc-500">
            Context: {activeStructure.pdb_id.toUpperCase()} —{" "}
            {activeStructure.residue_count} residues
          </span>
        </div>
      )}

      {/* Input */}
      <div className="p-2 border-t border-zinc-800">
        <div className="flex items-center gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              activeStructure
                ? "Ask about this structure…"
                : "Select a structure first…"
            }
            disabled={loading}
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded-md px-3 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-500 focus:outline-none focus:border-cyan-700 disabled:opacity-50"
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="p-1.5 rounded-md bg-cyan-800/50 text-cyan-300 hover:bg-cyan-700/50 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            aria-label="Send message"
          >
            <Send size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
