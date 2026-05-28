import React, { useRef, useMemo, useCallback, useEffect, useState } from 'react';
import { NodeData, EdgeData, mobiusAdd2D, hyperbolicDistance2D } from '../lib/math';

interface Visualizer2DProps {
  nodes: NodeData[];
  edges: EdgeData[];
  curvature: number;
  mobiusOffset2D: [number, number];
  showLabels: boolean;
  highlightOutliers: boolean;
  hoveredNode: string | null;
  selectedNodes: Set<string>;
  onNodeHover: (id: string | null) => void;
  onNodeSelect: (id: string) => void;
}

export default function Visualizer2D({
  nodes, edges, curvature, mobiusOffset2D, showLabels, highlightOutliers,
  hoveredNode, selectedNodes, onNodeHover, onNodeSelect
}: Visualizer2DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });
  const animRef = useRef<number>(0);

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setSize({ width, height });
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Transform nodes with 2D Möbius
  const transformedNodes = useMemo(() => {
    return nodes.map(node => ({
      ...node,
      tPos2D: mobiusAdd2D(mobiusOffset2D, node.position2D, curvature)
    }));
  }, [nodes, mobiusOffset2D, curvature]);

  // Map from disc coords [-1, 1] to canvas pixels
  const toCanvas = useCallback((pos: [number, number]): [number, number] => {
    const cx = size.width / 2;
    const cy = size.height / 2;
    const radius = Math.min(cx, cy) * 0.9;
    return [cx + pos[0] * radius, cy + pos[1] * radius];
  }, [size]);

  // Find node under cursor
  const findNodeAt = useCallback((canvasX: number, canvasY: number): string | null => {
    const cx = size.width / 2;
    const cy = size.height / 2;
    const radius = Math.min(cx, cy) * 0.9;
    
    let closest: string | null = null;
    let closestDist = 12; // pixel threshold

    for (const node of transformedNodes) {
      const [nx, ny] = toCanvas(node.tPos2D);
      const dist = Math.sqrt((canvasX - nx) ** 2 + (canvasY - ny) ** 2);
      if (dist < closestDist) {
        closestDist = dist;
        closest = node.id;
      }
    }
    return closest;
  }, [transformedNodes, toCanvas, size]);

  // Mouse handlers
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const nodeId = findNodeAt(x, y);
    onNodeHover(nodeId);
  }, [findNodeAt, onNodeHover]);

  const handleMouseLeave = useCallback(() => {
    onNodeHover(null);
  }, [onNodeHover]);

  const handleClick = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const nodeId = findNodeAt(x, y);
    if (nodeId) onNodeSelect(nodeId);
  }, [findNodeAt, onNodeSelect]);

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = size.width * window.devicePixelRatio;
    canvas.height = size.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const cx = size.width / 2;
    const cy = size.height / 2;
    const discRadius = Math.min(cx, cy) * 0.9;

    // Clear
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, size.width, size.height);

    // Draw boundary circle
    ctx.beginPath();
    ctx.arc(cx, cy, discRadius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.15)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Draw concentric geodesic circles (depth contours)
    for (const r of [0.3, 0.6, 0.8]) {
      ctx.beginPath();
      ctx.arc(cx, cy, discRadius * r, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.05)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Draw radial lines (angular reference)
    for (let i = 0; i < 8; i++) {
      const angle = (Math.PI * 2 * i) / 8;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + discRadius * Math.cos(angle), cy + discRadius * Math.sin(angle));
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.03)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Build node map for edge drawing
    const nodeMap = new Map<string, typeof transformedNodes[0]>();
    transformedNodes.forEach(n => nodeMap.set(n.id, n));

    // Draw edges
    ctx.lineWidth = 0.5;
    edges.forEach(edge => {
      const src = nodeMap.get(edge.source);
      const tgt = nodeMap.get(edge.target);
      if (!src || !tgt) return;

      const [sx, sy] = toCanvas(src.tPos2D);
      const [tx, ty] = toCanvas(tgt.tPos2D);
      
      const d = hyperbolicDistance2D(src.tPos2D, tgt.tPos2D, curvature);
      const opacity = Math.max(0.05, 0.3 * (1 - Math.min(d, 4) / 4));

      // Highlight edges connected to hovered/selected nodes
      const isActive = hoveredNode === edge.source || hoveredNode === edge.target ||
                       selectedNodes.has(edge.source) || selectedNodes.has(edge.target);

      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(tx, ty);
      ctx.strokeStyle = isActive 
        ? `rgba(255, 255, 255, ${opacity * 3})`
        : `rgba(255, 255, 255, ${opacity})`;
      ctx.stroke();
    });

    // Draw nodes
    transformedNodes.forEach(node => {
      const [nx, ny] = toCanvas(node.tPos2D);
      const isHovered = hoveredNode === node.id;
      const isSelected = selectedNodes.has(node.id);
      const isHighlighted = highlightOutliers && node.isOutlier;

      let radius = 3;
      let color = node.color;

      if (isHovered || isSelected) {
        radius = 6;
        color = '#ffffff';
      } else if (isHighlighted) {
        radius = 5;
        color = '#ff0000';
      }

      // Glow for active nodes
      if (isHovered || isSelected || isHighlighted) {
        ctx.beginPath();
        ctx.arc(nx, ny, radius + 4, 0, Math.PI * 2);
        const gradient = ctx.createRadialGradient(nx, ny, radius, nx, ny, radius + 4);
        gradient.addColorStop(0, color + '40');
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.fill();
      }

      // Node circle
      ctx.beginPath();
      ctx.arc(nx, ny, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // Selection ring
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(nx, ny, radius + 2, 0, Math.PI * 2);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Label
      if (showLabels || isHovered) {
        ctx.font = '9px monospace';
        ctx.fillStyle = isHovered ? '#ffffff' : 'rgba(255,255,255,0.5)';
        ctx.textAlign = 'center';
        ctx.fillText(node.id, nx, ny - radius - 4);
        if (isHovered) {
          ctx.font = '8px monospace';
          ctx.fillStyle = 'rgba(56, 189, 248, 0.8)';
          ctx.fillText(node.domain, nx, ny + radius + 10);
        }
      }
    });

  }, [transformedNodes, edges, size, curvature, hoveredNode, selectedNodes, 
      showLabels, highlightOutliers, toCanvas]);

  return (
    <div ref={containerRef} className="w-full h-full relative bg-[#0f172a]">
      <canvas
        ref={canvasRef}
        style={{ width: size.width, height: size.height }}
        className="cursor-crosshair"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      />
    </div>
  );
}
