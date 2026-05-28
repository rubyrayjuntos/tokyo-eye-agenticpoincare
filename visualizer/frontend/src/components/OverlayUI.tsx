import React from 'react';
import * as THREE from 'three';
import { Download, Maximize2, Columns2, Square, Circle } from 'lucide-react';
import { cn } from '../lib/utils';

interface OverlayUIProps {
  curvature: number;
  setCurvature: (v: number) => void;
  mobiusOffset: THREE.Vector3;
  setMobiusOffset: (v: THREE.Vector3) => void;
  mobiusOffset2D: [number, number];
  setMobiusOffset2D: (v: [number, number]) => void;
  showLabels: boolean;
  setShowLabels: (v: boolean) => void;
  highlightOutliers: boolean;
  setHighlightOutliers: (v: boolean) => void;
  viewMode: 'dual' | '3d' | '2d';
  setViewMode: (v: 'dual' | '3d' | '2d') => void;
  onExport: () => void;
  hoveredNode: string | null;
  selectedNodes: Set<string>;
  dataSource: 'mock' | 'static' | 'live' | 'agent';
  serverAvailable: boolean;
  proteins: Array<{ pdb_id: string; gene: string; desc: string }>;
  activePdb: string;
  onLoadProtein: (pdbId: string) => void;
  stats: {
    totalNodes: number;
    outliers: number;
    regressionSlope: number;
    correlation: number;
  };
}

export default function OverlayUI({
  curvature, setCurvature,
  mobiusOffset, setMobiusOffset,
  mobiusOffset2D, setMobiusOffset2D,
  showLabels, setShowLabels,
  highlightOutliers, setHighlightOutliers,
  viewMode, setViewMode,
  onExport,
  hoveredNode, selectedNodes,
  dataSource, serverAvailable, proteins, activePdb, onLoadProtein,
  stats
}: OverlayUIProps) {
  return (
    <div className="absolute inset-0 pointer-events-none flex flex-col overflow-hidden text-slate-100 font-sans">
      
      {/* Header */}
      <header className="h-14 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-900/60 backdrop-blur-md pointer-events-auto">
        <div className="flex items-center gap-4">
          <div className="w-7 h-7 rounded-full border-2 border-sky-500 flex items-center justify-center">
            <div className="w-1.5 h-1.5 bg-sky-500 rounded-full animate-pulse" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight uppercase">Tokyo Eyes <span className="text-sky-400">V5</span></h1>
            <p className="text-[9px] text-slate-400 font-mono">DECOUPLED RADIAL-ANGULAR HYPERBOLIC GNN</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {/* View mode toggle */}
          <div className="flex border border-slate-700 rounded overflow-hidden">
            <button 
              onClick={() => setViewMode('dual')}
              className={cn("px-3 py-1.5 text-[10px] font-bold uppercase flex items-center gap-1.5 transition-colors",
                viewMode === 'dual' ? "bg-sky-600 text-white" : "bg-slate-800 text-slate-400 hover:bg-slate-700")}
            >
              <Columns2 size={12} /> Dual
            </button>
            <button 
              onClick={() => setViewMode('3d')}
              className={cn("px-3 py-1.5 text-[10px] font-bold uppercase flex items-center gap-1.5 transition-colors",
                viewMode === '3d' ? "bg-sky-600 text-white" : "bg-slate-800 text-slate-400 hover:bg-slate-700")}
            >
              <Square size={12} /> 3D
            </button>
            <button 
              onClick={() => setViewMode('2d')}
              className={cn("px-3 py-1.5 text-[10px] font-bold uppercase flex items-center gap-1.5 transition-colors",
                viewMode === '2d' ? "bg-sky-600 text-white" : "bg-slate-800 text-slate-400 hover:bg-slate-700")}
            >
              <Circle size={12} /> 2D
            </button>
          </div>

          <div className="flex flex-col items-end mr-2">
            <span className="text-[9px] text-slate-500 uppercase font-bold">Nodes</span>
            <span className="text-xs font-mono text-emerald-400">{stats.totalNodes}</span>
          </div>
          <button onClick={onExport} className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-[10px] font-bold uppercase transition-colors flex items-center gap-1.5">
            <Download size={12} /> Export
          </button>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 flex justify-between pointer-events-none overflow-hidden">
        
        {/* Left Sidebar */}
        <aside className="w-64 border-r border-slate-800 p-5 flex flex-col gap-6 bg-slate-900/50 backdrop-blur-sm pointer-events-auto overflow-y-auto">
          <div>
            <h2 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Hyper-Parameters</h2>
            <div className="space-y-4">
              <div className="space-y-1.5">
                <div className="flex justify-between text-[10px] font-mono">
                  <span>CURVATURE (c)</span>
                  <span className="text-sky-400">{curvature.toFixed(3)}</span>
                </div>
                <input 
                  type="range" min="0.1" max="2.0" step="0.01" 
                  value={curvature}
                  onChange={e => setCurvature(parseFloat(e.target.value))}
                  className="w-full accent-sky-500 h-1 bg-slate-800 rounded-full appearance-none outline-none"
                />
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Display</h2>
            <div className="space-y-2">
              <label className="flex items-center justify-between p-2 bg-slate-900 border border-slate-800 rounded cursor-pointer hover:bg-slate-800 transition-colors">
                <span className="text-[11px]">Show Labels</span>
                <div className={cn("w-7 h-3.5 rounded-full flex items-center px-0.5 transition-colors", showLabels ? "bg-sky-600 justify-end" : "bg-slate-700")}>
                  <div className={cn("w-2 h-2 rounded-full", showLabels ? "bg-white" : "bg-slate-400")} />
                </div>
                <input type="checkbox" className="hidden" checked={showLabels} onChange={e => setShowLabels(e.target.checked)} />
              </label>
              
              <label className="flex items-center justify-between p-2 bg-slate-900 border border-slate-800 rounded cursor-pointer hover:bg-slate-800 transition-colors">
                <span className="text-[11px]">Alert Outliers</span>
                <div className={cn("w-7 h-3.5 rounded-full flex items-center px-0.5 transition-colors", highlightOutliers ? "bg-rose-600 justify-end" : "bg-slate-700")}>
                  <div className={cn("w-2 h-2 rounded-full", highlightOutliers ? "bg-white" : "bg-slate-400")} />
                </div>
                <input type="checkbox" className="hidden" checked={highlightOutliers} onChange={e => setHighlightOutliers(e.target.checked)} />
              </label>
            </div>
          </div>

          {/* Hovered/Selected info */}
          {(hoveredNode || selectedNodes.size > 0) && (
            <div>
              <h2 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Selection</h2>
              {hoveredNode && (
                <div className="p-2 bg-slate-900 border border-sky-900/30 rounded mb-2">
                  <div className="text-[9px] text-slate-400 uppercase">Hover</div>
                  <div className="text-[11px] text-sky-400 font-mono">{hoveredNode}</div>
                </div>
              )}
              {selectedNodes.size > 0 && (
                <div className="p-2 bg-slate-900 border border-slate-800 rounded">
                  <div className="text-[9px] text-slate-400 uppercase">Selected ({selectedNodes.size})</div>
                  <div className="text-[10px] text-slate-300 font-mono mt-1 max-h-20 overflow-y-auto">
                    {Array.from(selectedNodes).join(', ')}
                  </div>
                </div>
              )}
            </div>
          )}
        </aside>

        {/* Center — empty (viewport shows through) */}
        <div className="flex-1" />

        {/* Right Sidebar */}
        <aside className="w-64 border-l border-slate-800 p-5 flex flex-col gap-5 bg-slate-900/50 backdrop-blur-sm pointer-events-auto overflow-y-auto">
          <div>
            <h2 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Analysis</h2>
            <div className="space-y-2">
              <div className="p-2.5 bg-slate-900 border border-slate-800 rounded">
                <div className="text-[9px] text-slate-400 mb-1 uppercase">Depth-Value Correlation (ρ)</div>
                <div className="text-base font-mono text-indigo-400">{stats.correlation.toFixed(4)}</div>
                <div className="h-1 w-full bg-slate-800 rounded-full mt-1.5 overflow-hidden">
                  <div className="h-full bg-indigo-500" style={{ width: `${Math.min(Math.abs(stats.correlation) * 100, 100)}%`}} />
                </div>
              </div>
              
              <div className="p-2.5 bg-slate-900 border border-slate-800 rounded">
                <div className="text-[9px] text-slate-400 mb-1 uppercase">Radial Depth Slope</div>
                <div className="text-base font-mono text-sky-400">{stats.regressionSlope.toFixed(4)}</div>
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Outlier Detection</h2>
            <div className={cn("flex items-center gap-3 p-2.5 border rounded",
              highlightOutliers && stats.outliers > 0 ? "bg-rose-950/20 border-rose-900/30" : "bg-slate-900 border-slate-800")}>
              <div className={cn("w-2 h-2 rounded-full",
                highlightOutliers && stats.outliers > 0 ? "bg-rose-500 animate-pulse" : "bg-slate-600")} />
              <div className="flex-1">
                <div className="text-[9px] font-mono text-slate-400 uppercase">State</div>
                <div className={cn("text-[12px]",
                  highlightOutliers && stats.outliers > 0 ? "text-rose-400 font-bold" : "text-slate-500 font-mono")}>
                  {stats.outliers > 0 ? `${stats.outliers} Anomalies` : 'Stable'}
                </div>
              </div>
            </div>
          </div>

          <div className="mt-auto border-t border-slate-800 pt-4">
            <div className="flex items-center gap-2 mb-2">
              <div className={cn("w-1.5 h-1.5 rounded-full", 
                dataSource === 'live' ? "bg-emerald-500 animate-pulse" :
                dataSource === 'static' ? "bg-amber-500" : "bg-slate-500")} />
              <span className="text-[9px] font-bold uppercase tracking-widest">
                {dataSource === 'live' ? 'Live Server' : dataSource === 'static' ? 'Static Export' : 'Mock Data'}
              </span>
            </div>
            {serverAvailable && proteins.length > 0 && (
              <div className="mt-2">
                <div className="text-[9px] text-slate-500 uppercase mb-1">Load Protein</div>
                <div className="flex flex-wrap gap-1">
                  {proteins.slice(0, 6).map(p => (
                    <button
                      key={p.pdb_id}
                      onClick={() => onLoadProtein(p.pdb_id)}
                      className={cn(
                        "px-1.5 py-0.5 text-[9px] font-mono rounded border transition-colors",
                        p.pdb_id === activePdb
                          ? "bg-sky-900/50 border-sky-700 text-sky-300"
                          : "bg-slate-900 border-slate-700 text-slate-400 hover:bg-slate-800"
                      )}
                    >
                      {p.pdb_id}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {!serverAvailable && (
              <p className="text-[9px] text-slate-500 mt-1">
                Drop viewer_data.json or start serve_live.py
              </p>
            )}
            <p className="text-[10px] text-slate-400 leading-relaxed mt-2">
              v5 Decoupled radial-angular. Gradient isolation verified.
            </p>
          </div>
        </aside>
      </main>

      {/* Footer: Möbius Scrubber */}
      <footer className="h-24 border-t border-slate-800 bg-slate-900/80 backdrop-blur-sm flex flex-col justify-center px-6 gap-2 pointer-events-auto shrink-0 z-10">
        <h2 className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Möbius Transformation Scrubber</h2>
        
        <div className="flex gap-6 items-center w-full">
          {/* 3D scrubbers */}
          <div className="flex-1 space-y-1">
            <div className="flex justify-between text-[10px] font-mono uppercase text-slate-500">
              <span>3D X</span>
              <span className="text-sky-400">{mobiusOffset.x.toFixed(2)}</span>
            </div>
            <input 
              type="range" min="-0.99" max="0.99" step="0.01" 
              value={mobiusOffset.x}
              onChange={e => setMobiusOffset(new THREE.Vector3(parseFloat(e.target.value), mobiusOffset.y, mobiusOffset.z))}
              className="w-full h-1 bg-slate-700 rounded-full appearance-none outline-none accent-sky-500"
            />
          </div>
          <div className="flex-1 space-y-1">
            <div className="flex justify-between text-[10px] font-mono uppercase text-slate-500">
              <span>3D Y</span>
              <span className="text-sky-400">{mobiusOffset.y.toFixed(2)}</span>
            </div>
            <input 
              type="range" min="-0.99" max="0.99" step="0.01" 
              value={mobiusOffset.y}
              onChange={e => setMobiusOffset(new THREE.Vector3(mobiusOffset.x, parseFloat(e.target.value), mobiusOffset.z))}
              className="w-full h-1 bg-slate-700 rounded-full appearance-none outline-none accent-sky-500"
            />
          </div>
          <div className="flex-1 space-y-1">
            <div className="flex justify-between text-[10px] font-mono uppercase text-slate-500">
              <span>3D Z</span>
              <span className="text-sky-400">{mobiusOffset.z.toFixed(2)}</span>
            </div>
            <input 
              type="range" min="-0.99" max="0.99" step="0.01" 
              value={mobiusOffset.z}
              onChange={e => setMobiusOffset(new THREE.Vector3(mobiusOffset.x, mobiusOffset.y, parseFloat(e.target.value)))}
              className="w-full h-1 bg-slate-700 rounded-full appearance-none outline-none accent-sky-500"
            />
          </div>

          <div className="w-px h-8 bg-slate-700" />

          {/* 2D scrubbers */}
          <div className="flex-1 space-y-1">
            <div className="flex justify-between text-[10px] font-mono uppercase text-slate-500">
              <span>2D X</span>
              <span className="text-emerald-400">{mobiusOffset2D[0].toFixed(2)}</span>
            </div>
            <input 
              type="range" min="-0.99" max="0.99" step="0.01" 
              value={mobiusOffset2D[0]}
              onChange={e => setMobiusOffset2D([parseFloat(e.target.value), mobiusOffset2D[1]])}
              className="w-full h-1 bg-slate-700 rounded-full appearance-none outline-none accent-emerald-500"
            />
          </div>
          <div className="flex-1 space-y-1">
            <div className="flex justify-between text-[10px] font-mono uppercase text-slate-500">
              <span>2D Y</span>
              <span className="text-emerald-400">{mobiusOffset2D[1].toFixed(2)}</span>
            </div>
            <input 
              type="range" min="-0.99" max="0.99" step="0.01" 
              value={mobiusOffset2D[1]}
              onChange={e => setMobiusOffset2D([mobiusOffset2D[0], parseFloat(e.target.value)])}
              className="w-full h-1 bg-slate-700 rounded-full appearance-none outline-none accent-emerald-500"
            />
          </div>
          
          <button 
            onClick={() => {
              setMobiusOffset(new THREE.Vector3(0, 0, 0));
              setMobiusOffset2D([0, 0]);
            }}
            className="ml-2 px-3 py-1.5 text-[9px] uppercase font-bold tracking-widest bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded transition-colors text-slate-300 h-fit whitespace-nowrap"
          >
            Reset
          </button>
        </div>
      </footer>
    </div>
  );
}
