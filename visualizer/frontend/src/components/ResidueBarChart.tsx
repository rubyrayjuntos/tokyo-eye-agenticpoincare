import { useEffect, useRef, useState } from "react";
import {
  Chart,
  BarController,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
} from "chart.js";
import { useDashboard } from "../lib/context";
import { api } from "../lib/api";
import type { GraphMetrics } from "../lib/types";

Chart.register(BarController, BarElement, CategoryScale, LinearScale, Tooltip);

type MetricMode = "dehydron_score" | "centrality";

export default function ResidueBarChart() {
  const { activeStructure, refreshKey } = useDashboard();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart | null>(null);
  const [metricMode, setMetricMode] = useState<MetricMode>("dehydron_score");
  const [data, setData] = useState<GraphMetrics | null>(null);

  useEffect(() => {
    if (!activeStructure) {
      setData(null);
      return;
    }
    let cancelled = false;
    api.getMetrics(activeStructure.structure_id).then((result) => {
      if (!cancelled) setData(result);
    }).catch(() => {
      if (!cancelled) setData(null);
    });
    return () => { cancelled = true; };
  }, [activeStructure, refreshKey]);

  useEffect(() => {
    if (!canvasRef.current || !data) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const residues = data.residues;
    const labels = residues.map((r) => r.residue_id);
    const values = residues.map((r) =>
      metricMode === "dehydron_score" ? r.dehydron_score : r.centrality
    );

    chartRef.current = new Chart(canvasRef.current, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: metricMode === "dehydron_score" ? "Dehydron Score" : "Centrality",
            data: values,
            backgroundColor: "rgba(34,211,238,0.6)",
            borderColor: "rgba(34,211,238,0.9)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const r = residues[ctx.dataIndex];
                return `${r.residue_id}: ${metricMode === "dehydron_score" ? r.dehydron_score.toFixed(3) : r.centrality.toFixed(4)}`;
              },
            },
          },
        },
        scales: {
          x: {
            display: false, // too many labels for residues
          },
          y: {
            grid: { color: "rgba(63,63,70,0.4)" },
            ticks: { color: "#a1a1aa", font: { size: 9 } },
          },
        },
      },
    });

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [data, metricMode]);

  if (!activeStructure) {
    return (
      <div className="h-full flex items-center justify-center text-zinc-600 text-xs">
        Select a structure to view metrics
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between px-2 py-1 shrink-0">
        <span className="text-[10px] uppercase tracking-wider text-zinc-500">
          Residue Metrics
        </span>
        <select
          value={metricMode}
          onChange={(e) => setMetricMode(e.target.value as MetricMode)}
          className="text-[10px] bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-zinc-300"
        >
          <option value="dehydron_score">Dehydron Score</option>
          <option value="centrality">Centrality</option>
        </select>
      </div>
      <div className="flex-1 min-h-0 p-1">
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}
