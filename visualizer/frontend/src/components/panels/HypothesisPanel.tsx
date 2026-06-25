/**
 * HypothesisPanel — Propose, test, and track scientific hypotheses.
 * Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
 */
import { useState, useCallback } from "react";
import { api } from "../../lib/api";
import { useDashboard } from "../../lib/context";
import { useHydration } from "../../context/HydrationProvider";
import type { Hypothesis, Prediction } from "../../lib/types";

type HypothesisStatus = Hypothesis["status"];

/** Parse residue references from hypothesis text (e.g. "G12", "Res 45", "residue A123") */
function parseResidueRefs(text: string): string[] {
  const patterns = [
    /\b[A-Z]{1,3}(\d+)\b/g,
    /\b(?:residue|res|resi)\s*[A-Z]?(\d+)/gi,
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

const STATUS_COLORS: Record<HypothesisStatus, string> = {
  proposed: "bg-blue-600/30 text-blue-300 border-blue-500/40",
  gathering: "bg-yellow-600/30 text-yellow-300 border-yellow-500/40",
  supported: "bg-emerald-600/30 text-emerald-300 border-emerald-500/40",
  contradicted: "bg-red-600/30 text-red-300 border-red-500/40",
  inconclusive: "bg-zinc-600/30 text-zinc-300 border-zinc-500/40",
};

function ConfidenceBar({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  const color =
    confidence >= 0.7
      ? "bg-emerald-500"
      : confidence >= 0.4
        ? "bg-yellow-500"
        : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-zinc-700 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] text-zinc-400 w-8 text-right">{pct}%</span>
    </div>
  );
}

function StatusBadge({ status }: { status: HypothesisStatus }) {
  return (
    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium border ${STATUS_COLORS[status]}`}>
      {status}
    </span>
  );
}

function PredictionSummary({ predictions }: { predictions: Prediction[] }) {
  if (predictions.length === 0) return null;
  return (
    <div className="mt-1.5 space-y-0.5">
      {predictions.map((p) => (
        <div key={p.prediction_id} className="flex items-center gap-1.5 text-[10px]">
          <span
            className={
              p.passed === null
                ? "text-zinc-500"
                : p.passed
                  ? "text-emerald-400"
                  : "text-red-400"
            }
          >
            {p.passed === null ? "○" : p.passed ? "✓" : "✗"}
          </span>
          <span className="text-zinc-400 truncate">{p.statement}</span>
        </div>
      ))}
    </div>
  );
}

interface NewPredictionInput {
  statement: string;
  test_tool: string;
  threshold: string;
}

export default function HypothesisPanel() {
  const { activeStructure, emitDirective } = useDashboard();
  const { hypotheses, loading, refresh } = useHydration();

  // UI state
  const [showForm, setShowForm] = useState(false);
  const [showEvidenceFor, setShowEvidenceFor] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);

  // New hypothesis form state
  const [statement, setStatement] = useState("");
  const [mechanism, setMechanism] = useState("");
  const [predictions, setPredictions] = useState<NewPredictionInput[]>([
    { statement: "", test_tool: "", threshold: "" },
  ]);
  const [submitting, setSubmitting] = useState(false);

  // Evidence form state
  const [evidenceDesc, setEvidenceDesc] = useState("");
  const [evidenceSupports, setEvidenceSupports] = useState(true);
  const [evidenceStrength, setEvidenceStrength] = useState(0.5);
  const [evidenceSource, setEvidenceSource] = useState("");
  const [evidenceSubmitting, setEvidenceSubmitting] = useState(false);

  const structureId = activeStructure?.structure_id ?? null;

  const addPrediction = () => {
    setPredictions((prev) => [...prev, { statement: "", test_tool: "", threshold: "" }]);
  };

  const removePrediction = (idx: number) => {
    setPredictions((prev) => prev.filter((_, i) => i !== idx));
  };

  const updatePrediction = (idx: number, field: keyof NewPredictionInput, value: string) => {
    setPredictions((prev) =>
      prev.map((p, i) => (i === idx ? { ...p, [field]: value } : p))
    );
  };

  const handleSubmitHypothesis = useCallback(async () => {
    if (!structureId || !statement.trim()) return;
    const validPredictions = predictions.filter((p) => p.statement.trim());
    if (validPredictions.length === 0) {
      setError("At least one prediction is required (falsifiability guardrail).");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await api.createHypothesis({
        structure_id: structureId,
        statement: statement.trim(),
        predictions: validPredictions.map((p) => ({
          statement: p.statement.trim(),
          test_tool: p.test_tool.trim() || undefined,
          threshold: p.threshold.trim() || undefined,
        })),
        mechanism: mechanism.trim() || undefined,
      });
      // Reset form
      setStatement("");
      setMechanism("");
      setPredictions([{ statement: "", test_tool: "", threshold: "" }]);
      setShowForm(false);
      refresh();
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Failed to create hypothesis";
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  }, [structureId, statement, mechanism, predictions, refresh]);

  const handleTest = useCallback(
    async (hypothesisId: string) => {
      setTestingId(hypothesisId);
      setError(null);
      try {
        await api.testHypothesis(hypothesisId);
        refresh();
      } catch (err: unknown) {
        const msg =
          err && typeof err === "object" && "message" in err
            ? (err as { message: string }).message
            : "Test failed";
        setError(msg);
      } finally {
        setTestingId(null);
      }
    },
    [refresh]
  );

  const handleAddEvidence = useCallback(
    async (hypothesisId: string) => {
      if (!evidenceDesc.trim() || !evidenceSource.trim()) return;
      setEvidenceSubmitting(true);
      setError(null);
      try {
        await api.addEvidence(hypothesisId, {
          source_tool: evidenceSource.trim(),
          supports: evidenceSupports,
          strength: evidenceStrength,
          description: evidenceDesc.trim(),
        });
        // Reset evidence form
        setEvidenceDesc("");
        setEvidenceSource("");
        setEvidenceSupports(true);
        setEvidenceStrength(0.5);
        setShowEvidenceFor(null);
        refresh();
      } catch (err: unknown) {
        const msg =
          err && typeof err === "object" && "message" in err
            ? (err as { message: string }).message
            : "Failed to add evidence";
        setError(msg);
      } finally {
        setEvidenceSubmitting(false);
      }
    },
    [evidenceDesc, evidenceSource, evidenceSupports, evidenceStrength, refresh]
  );

  // No structure selected
  if (!activeStructure) {
    return (
      <div className="flex items-center justify-center h-full px-4">
        <p className="text-xs text-zinc-600 text-center">Select a structure to view hypotheses.</p>
      </div>
    );
  }

  // Loading state
  if (loading.hypotheses) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-xs text-zinc-500 animate-pulse">Loading hypotheses...</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Error banner */}
      {error && (
        <div className="px-3 py-2 bg-red-900/20 border-b border-red-800/40">
          <p className="text-[11px] text-red-400">{error}</p>
        </div>
      )}

      {/* Header with New Hypothesis button */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800">
        <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wide">
          Hypotheses ({hypotheses?.length ?? 0})
        </p>
        <button
          onClick={() => setShowForm(!showForm)}
          className="px-2 py-1 bg-cyan-700/30 border border-cyan-600/40 rounded text-[11px] font-medium text-cyan-300 hover:bg-cyan-700/50"
        >
          {showForm ? "Cancel" : "New Hypothesis"}
        </button>
      </div>

      {/* New Hypothesis Form */}
      {showForm && (
        <div className="px-3 py-3 border-b border-zinc-800 space-y-2 bg-zinc-900/50">
          <div>
            <label className="text-[10px] text-zinc-500 block mb-0.5">Statement *</label>
            <textarea
              value={statement}
              onChange={(e) => setStatement(e.target.value)}
              placeholder="e.g., Residue G12 acts as an allosteric switch..."
              className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600 resize-none"
              rows={2}
            />
          </div>

          <div>
            <label className="text-[10px] text-zinc-500 block mb-0.5">Mechanism (optional)</label>
            <input
              type="text"
              value={mechanism}
              onChange={(e) => setMechanism(e.target.value)}
              placeholder="Proposed mechanism..."
              className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-[10px] text-zinc-500">Predictions *</label>
              <button
                onClick={addPrediction}
                className="text-[10px] text-cyan-400 hover:text-cyan-300"
              >
                + Add
              </button>
            </div>
            {predictions.map((pred, idx) => (
              <div key={idx} className="flex gap-1.5 mb-1.5">
                <input
                  type="text"
                  value={pred.statement}
                  onChange={(e) => updatePrediction(idx, "statement", e.target.value)}
                  placeholder="Prediction statement"
                  className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
                />
                <input
                  type="text"
                  value={pred.test_tool}
                  onChange={(e) => updatePrediction(idx, "test_tool", e.target.value)}
                  placeholder="Tool"
                  className="w-20 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
                />
                <input
                  type="text"
                  value={pred.threshold}
                  onChange={(e) => updatePrediction(idx, "threshold", e.target.value)}
                  placeholder="Threshold"
                  className="w-20 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600"
                />
                {predictions.length > 1 && (
                  <button
                    onClick={() => removePrediction(idx)}
                    className="text-[10px] text-red-400 hover:text-red-300 px-1"
                  >
                    ✗
                  </button>
                )}
              </div>
            ))}
          </div>

          <button
            onClick={handleSubmitHypothesis}
            disabled={submitting || !statement.trim()}
            className="w-full px-2 py-1.5 bg-cyan-700/40 border border-cyan-600/50 rounded text-[11px] font-medium text-cyan-200 hover:bg-cyan-700/60 disabled:opacity-40"
          >
            {submitting ? "Submitting..." : "Submit Hypothesis"}
          </button>
        </div>
      )}

      {/* Hypothesis cards */}
      <div className="flex-1 overflow-auto">
        {(!hypotheses || hypotheses.length === 0) ? (
          <div className="flex items-center justify-center h-full px-4">
            <p className="text-xs text-zinc-600 text-center">No hypotheses yet. Click "New Hypothesis" to propose one.</p>
          </div>
        ) : (
          <div className="divide-y divide-zinc-800">
            {hypotheses.map((h) => (
              <div key={h.hypothesis_id} className="px-3 py-2.5 space-y-1.5">
                {/* Header: status + confidence */}
                <div className="flex items-center justify-between">
                  <StatusBadge status={h.status} />
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-zinc-500">
                      +{h.evidence_supporting} / −{h.evidence_contradicting}
                    </span>
                  </div>
                </div>

                {/* Statement */}
                <p className="text-xs text-zinc-200 leading-relaxed">{h.statement}</p>

                {/* Confidence bar */}
                <ConfidenceBar confidence={h.confidence} />

                {/* Predictions */}
                <PredictionSummary predictions={h.predictions} />

                {/* Actions */}
                <div className="flex gap-2 pt-1">
                  <button
                    onClick={() => handleTest(h.hypothesis_id)}
                    disabled={testingId === h.hypothesis_id}
                    className="px-2 py-1 bg-emerald-700/30 border border-emerald-600/40 rounded text-[10px] text-emerald-300 hover:bg-emerald-700/50 disabled:opacity-40"
                  >
                    {testingId === h.hypothesis_id ? "Testing..." : "Test"}
                  </button>
                  <button
                    onClick={() =>
                      setShowEvidenceFor(
                        showEvidenceFor === h.hypothesis_id ? null : h.hypothesis_id
                      )
                    }
                    className="px-2 py-1 bg-violet-700/30 border border-violet-600/40 rounded text-[10px] text-violet-300 hover:bg-violet-700/50"
                  >
                    {showEvidenceFor === h.hypothesis_id ? "Cancel" : "Add Evidence"}
                  </button>
                  <button
                    onClick={() => {
                      const refs = parseResidueRefs(
                        h.statement + " " + h.predictions.map((p) => p.statement).join(" ")
                      );
                      if (refs.length > 0) {
                        emitDirective({
                          action: "highlight",
                          structure_id: structureId ?? undefined,
                          highlight_groups: [
                            {
                              residue_ids: refs,
                              color: "#60a5fa",
                              style: "glow",
                              label: "Hypothesis residues",
                            },
                          ],
                        });
                      }
                    }}
                    className="px-2 py-1 bg-blue-700/30 border border-blue-600/40 rounded text-[10px] text-blue-300 hover:bg-blue-700/50"
                  >
                    Highlight
                  </button>
                </div>

                {/* Evidence form (inline) */}
                {showEvidenceFor === h.hypothesis_id && (
                  <div className="mt-2 p-2 bg-zinc-800/50 rounded border border-zinc-700 space-y-1.5">
                    <input
                      type="text"
                      value={evidenceSource}
                      onChange={(e) => setEvidenceSource(e.target.value)}
                      placeholder="Source tool (e.g., graph_metrics)"
                      className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-violet-600"
                    />
                    <textarea
                      value={evidenceDesc}
                      onChange={(e) => setEvidenceDesc(e.target.value)}
                      placeholder="Evidence description..."
                      className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-violet-600 resize-none"
                      rows={2}
                    />
                    <div className="flex items-center gap-3">
                      {/* Supports/Contradicts toggle */}
                      <div className="flex items-center gap-1.5">
                        <button
                          onClick={() => setEvidenceSupports(true)}
                          className={`px-1.5 py-0.5 rounded text-[10px] border ${
                            evidenceSupports
                              ? "bg-emerald-700/40 border-emerald-600/50 text-emerald-300"
                              : "bg-zinc-800 border-zinc-700 text-zinc-500"
                          }`}
                        >
                          Supports
                        </button>
                        <button
                          onClick={() => setEvidenceSupports(false)}
                          className={`px-1.5 py-0.5 rounded text-[10px] border ${
                            !evidenceSupports
                              ? "bg-red-700/40 border-red-600/50 text-red-300"
                              : "bg-zinc-800 border-zinc-700 text-zinc-500"
                          }`}
                        >
                          Contradicts
                        </button>
                      </div>
                      {/* Strength slider */}
                      <div className="flex items-center gap-1.5 flex-1">
                        <span className="text-[10px] text-zinc-500">Strength:</span>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={evidenceStrength}
                          onChange={(e) => setEvidenceStrength(parseFloat(e.target.value))}
                          className="flex-1 h-1 accent-violet-500"
                        />
                        <span className="text-[10px] text-zinc-400 w-6 text-right">
                          {evidenceStrength.toFixed(1)}
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={() => handleAddEvidence(h.hypothesis_id)}
                      disabled={evidenceSubmitting || !evidenceDesc.trim() || !evidenceSource.trim()}
                      className="w-full px-2 py-1 bg-violet-700/40 border border-violet-600/50 rounded text-[10px] text-violet-200 hover:bg-violet-700/60 disabled:opacity-40"
                    >
                      {evidenceSubmitting ? "Adding..." : "Submit Evidence"}
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
