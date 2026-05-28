import React, { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import * as THREE from 'three';
import Visualizer3D from './components/Visualizer3D';
import Visualizer2D from './components/Visualizer2D';
import OverlayUI from './components/OverlayUI';
import ChatPanel from './components/ChatPanel';
import { generateMockGNNData, NodeData, EdgeData } from './lib/math';
import { loadGNNData, loadGNNDataFromFile, LoadedData } from './lib/dataLoader';
import { ViewportClient, checkAgent, fetchPoincareData, ViewportDirective, PoincareResidue } from './lib/agentClient';
import { adaptPoincareData, recolorNodes } from './lib/dataAdapter';
import { useDirectives, getHighlightColor, shouldPulse } from './lib/useDirectives';

export default function App() {
  // Data state
  const [data, setData] = useState<LoadedData>(() => ({ ...generateMockGNNData(), metadata: undefined }));
  const [residues, setResidues] = useState<PoincareResidue[]>([]);
  const [dataSource, setDataSource] = useState<'mock' | 'static' | 'live' | 'agent'>('mock');
  const [activePdb, setActivePdb] = useState('4OBE');
  const [proteins, setProteins] = useState<Array<{ pdb_id: string; gene: string; desc: string }>>([]);

  // Agent connection state
  const [agentAvailable, setAgentAvailable] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const viewportClientRef = useRef<ViewportClient | null>(null);
  const sessionIdRef = useRef(crypto.randomUUID());

  // Viewport directive state
  const {
    highlights, focusTargets, activeMetric, annotations, lastMessage,
    applyDirective, clearHighlights,
  } = useDirectives('cone_depth');

  // Shared interaction state
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set());

  // Hyper-parameters
  const [mobiusOffset, setMobiusOffset] = useState<THREE.Vector3>(new THREE.Vector3(0, 0, 0));
  const [mobiusOffset2D, setMobiusOffset2D] = useState<[number, number]>([0, 0]);
  const [curvature, setCurvature] = useState<number>(1.0);
  const [showLabels, setShowLabels] = useState(false);
  const [highlightOutliers, setHighlightOutliers] = useState(true);
  const [viewMode, setViewMode] = useState<'dual' | '3d' | '2d'>('dual');
  const [showChat, setShowChat] = useState(true);

  // ------------------------------------------------------------------
  // Agent connection + WebSocket
  // ------------------------------------------------------------------

  useEffect(() => {
    const init = async () => {
      const available = await checkAgent();
      setAgentAvailable(available);

      if (available) {
        // Load data from agent
        try {
          const response = await fetchPoincareData(activePdb);
          const adapted = adaptPoincareData(response, 'cone_depth');
          setData({ nodes: adapted.nodes, edges: adapted.edges, metadata: undefined });
          setResidues(response.residues);
          setCurvature(response.curvature_c);
          setDataSource('agent');
        } catch {
          // Fall back to legacy data loading
          const loaded = await loadGNNData(activePdb);
          setData(loaded);
          setDataSource(loaded.metadata ? 'static' : 'mock');
        }

        // Connect WebSocket
        const client = new ViewportClient({
          sessionId: sessionIdRef.current,
          onDirective: (directive) => applyDirective(directive),
          onConnection: (connected) => setWsConnected(connected),
        });
        client.connect();
        viewportClientRef.current = client;
      } else {
        // No agent — use legacy data loading
        const loaded = await loadGNNData(activePdb);
        setData(loaded);
        if (loaded.metadata) {
          setDataSource('static');
          if (loaded.metadata.curvature) setCurvature(loaded.metadata.curvature);
        }
      }
    };
    init();

    return () => {
      viewportClientRef.current?.disconnect();
    };
  }, []);

  // ------------------------------------------------------------------
  // Load protein (from agent or legacy)
  // ------------------------------------------------------------------

  const loadProtein = useCallback(async (pdbId: string) => {
    setActivePdb(pdbId);
    clearHighlights();

    if (agentAvailable) {
      try {
        const response = await fetchPoincareData(pdbId);
        const adapted = adaptPoincareData(response, activeMetric as any);
        setData({ nodes: adapted.nodes, edges: adapted.edges, metadata: undefined });
        setResidues(response.residues);
        setCurvature(response.curvature_c);
        setDataSource('agent');
        return;
      } catch { /* fall through */ }
    }

    const loaded = await loadGNNData(pdbId);
    setData(loaded);
    setDataSource(loaded.metadata ? 'static' : 'mock');
  }, [agentAvailable, activeMetric, clearHighlights]);

  // ------------------------------------------------------------------
  // Re-color when metric changes (from directives or UI)
  // ------------------------------------------------------------------

  useEffect(() => {
    if (residues.length > 0 && dataSource === 'agent') {
      const recolored = recolorNodes(data.nodes, residues, activeMetric as any);
      setData(prev => ({ ...prev, nodes: recolored }));
    }
  }, [activeMetric, residues]);

  // ------------------------------------------------------------------
  // Interaction handlers (send events to agent via WebSocket)
  // ------------------------------------------------------------------

  const handleNodeHover = useCallback((nodeId: string | null) => {
    setHoveredNode(nodeId);
    viewportClientRef.current?.sendHover(nodeId);
  }, []);

  const handleNodeSelect = useCallback((nodeId: string) => {
    setSelectedNodes(prev => {
      const next = new Set(prev);
      if (next.has(nodeId)) next.delete(nodeId);
      else next.add(nodeId);
      viewportClientRef.current?.sendSelection(Array.from(next));
      return next;
    });
  }, []);

  // Handle directives from chat panel
  const handleChatDirective = useCallback((directive: ViewportDirective) => {
    applyDirective(directive);
  }, [applyDirective]);

  // ------------------------------------------------------------------
  // File drop
  // ------------------------------------------------------------------

  useEffect(() => {
    const handleDrop = async (e: DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer?.files[0];
      if (file && file.name.endsWith('.json')) {
        const loaded = await loadGNNDataFromFile(file);
        setData(loaded);
        setDataSource('static');
        if (loaded.metadata?.curvature) setCurvature(loaded.metadata.curvature);
      }
    };
    const handleDragOver = (e: DragEvent) => e.preventDefault();
    window.addEventListener('drop', handleDrop);
    window.addEventListener('dragover', handleDragOver);
    return () => {
      window.removeEventListener('drop', handleDrop);
      window.removeEventListener('dragover', handleDragOver);
    };
  }, []);

  // ------------------------------------------------------------------
  // Stats
  // ------------------------------------------------------------------

  const stats = useMemo(() => {
    const outliers = data.nodes.filter(n => n.isOutlier).length;
    const depths = data.nodes.map(n => n.depth);
    const values = data.nodes.map(n => n.value);
    const n = depths.length;
    if (n === 0) return { totalNodes: 0, outliers: 0, regressionSlope: 0, correlation: 0 };

    const sumX = depths.reduce((a, b) => a + b, 0);
    const sumY = values.reduce((a, b) => a + b, 0);
    const sumXY = depths.reduce((a, b, i) => a + b * values[i], 0);
    const sumXX = depths.reduce((a, b) => a + b * b, 0);
    const sumYY = values.reduce((a, b) => a + b * b, 0);
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX) || 0;
    const correlation = (n * sumXY - sumX * sumY) /
      Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY)) || 0;

    return { totalNodes: n, outliers, regressionSlope: slope, correlation };
  }, [data]);

  const handleExport = () => {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tokyo_eyes_v5_${activePdb}.json`;
    a.click();
  };

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------

  return (
    <div className="w-full h-screen relative bg-slate-950 overflow-hidden font-sans text-slate-100">
      {/* Main layout */}
      <div className="absolute inset-0 flex">
        {/* Left sidebar (controls) */}
        <div className="w-64 shrink-0" />

        {/* Viewport area */}
        <div className="flex-1 flex relative">
          {(viewMode === 'dual' || viewMode === '3d') && (
            <div className={viewMode === 'dual' ? 'w-1/2 h-full relative' : 'w-full h-full relative'}>
              <Visualizer3D
                nodes={data.nodes}
                edges={data.edges}
                curvature={curvature}
                mobiusOffset={mobiusOffset}
                showLabels={showLabels}
                highlightOutliers={highlightOutliers}
                hoveredNode={hoveredNode}
                selectedNodes={selectedNodes}
                onNodeHover={handleNodeHover}
                onNodeSelect={handleNodeSelect}
              />
              <div className="absolute bottom-2 left-2 text-[9px] font-mono text-slate-500 uppercase tracking-widest pointer-events-none">
                Poincaré Ball (3D)
              </div>
            </div>
          )}
          {viewMode === 'dual' && <div className="w-px bg-slate-700/50 shrink-0" />}
          {(viewMode === 'dual' || viewMode === '2d') && (
            <div className={viewMode === 'dual' ? 'w-1/2 h-full relative' : 'w-full h-full relative'}>
              <Visualizer2D
                nodes={data.nodes}
                edges={data.edges}
                curvature={curvature}
                mobiusOffset2D={mobiusOffset2D}
                showLabels={showLabels}
                highlightOutliers={highlightOutliers}
                hoveredNode={hoveredNode}
                selectedNodes={selectedNodes}
                onNodeHover={handleNodeHover}
                onNodeSelect={handleNodeSelect}
              />
              <div className="absolute bottom-2 left-2 text-[9px] font-mono text-slate-500 uppercase tracking-widest pointer-events-none">
                Poincaré Disc (2D)
              </div>
            </div>
          )}
        </div>

        {/* Right sidebar: Chat panel */}
        {showChat && (
          <div className="w-72 shrink-0">
            <ChatPanel
              sessionId={sessionIdRef.current}
              structureId={activePdb}
              currentMetric={activeMetric}
              curvature={curvature}
              onDirective={handleChatDirective}
              agentConnected={agentAvailable}
            />
          </div>
        )}
      </div>

      {/* Directive status bar */}
      {lastMessage && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-slate-800/90 backdrop-blur-sm border border-slate-700/50 rounded-lg px-4 py-2 text-xs text-cyan-300 pointer-events-none animate-pulse">
          {lastMessage}
        </div>
      )}

      {/* Connection status */}
      <div className="absolute top-2 right-2 flex items-center gap-2 text-[10px] text-slate-500 pointer-events-none">
        <div className={`w-1.5 h-1.5 rounded-full ${wsConnected ? 'bg-emerald-400' : agentAvailable ? 'bg-amber-400' : 'bg-red-400'}`} />
        <span>{wsConnected ? 'live' : agentAvailable ? 'REST only' : dataSource}</span>
      </div>

      {/* UI Overlay (left sidebar controls) */}
      <OverlayUI
        curvature={curvature}
        setCurvature={setCurvature}
        mobiusOffset={mobiusOffset}
        setMobiusOffset={setMobiusOffset}
        mobiusOffset2D={mobiusOffset2D}
        setMobiusOffset2D={setMobiusOffset2D}
        showLabels={showLabels}
        setShowLabels={setShowLabels}
        highlightOutliers={highlightOutliers}
        setHighlightOutliers={setHighlightOutliers}
        viewMode={viewMode}
        setViewMode={setViewMode}
        stats={stats}
        onExport={handleExport}
        hoveredNode={hoveredNode}
        selectedNodes={selectedNodes}
        dataSource={dataSource}
        serverAvailable={agentAvailable}
        proteins={proteins}
        activePdb={activePdb}
        onLoadProtein={loadProtein}
      />
    </div>
  );
}
