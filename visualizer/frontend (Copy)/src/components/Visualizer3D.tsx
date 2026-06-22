import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Html, Environment } from '@react-three/drei';
import * as THREE from 'three';
import { NodeData, EdgeData, mobiusAdd, hyperbolicDistance } from '../lib/math';

interface Visualizer3DProps {
  nodes: NodeData[];
  edges: EdgeData[];
  curvature: number;
  mobiusOffset: THREE.Vector3;
  showLabels: boolean;
  highlightOutliers: boolean;
  hoveredNode: string | null;
  selectedNodes: Set<string>;
  onNodeHover: (id: string | null) => void;
  onNodeSelect: (id: string) => void;
}

interface NetworkGraphProps extends Visualizer3DProps {}

const NetworkGraph: React.FC<NetworkGraphProps> = ({
  nodes, edges, curvature, mobiusOffset, showLabels, highlightOutliers,
  hoveredNode, selectedNodes, onNodeHover, onNodeSelect
}) => {
  // Apply Möbius transformation
  const transformedNodes = useMemo(() => {
    return nodes.map(node => ({
      ...node,
      tPos: mobiusAdd(mobiusOffset, node.position, curvature)
    }));
  }, [nodes, mobiusOffset, curvature]);

  // Edge geometry
  const edgeBuffer = useMemo(() => {
    const points: number[] = [];
    const colors: number[] = [];
    const nodeMap = new Map<string, typeof transformedNodes[0]>();
    transformedNodes.forEach(n => nodeMap.set(n.id, n));

    edges.forEach(edge => {
      const src = nodeMap.get(edge.source);
      const tgt = nodeMap.get(edge.target);
      if (src && tgt) {
        points.push(src.tPos.x, src.tPos.y, src.tPos.z);
        points.push(tgt.tPos.x, tgt.tPos.y, tgt.tPos.z);
        const d = hyperbolicDistance(src.tPos, tgt.tPos, curvature);
        const opacity = Math.max(0.1, 1 - Math.min(d, 4) / 4);
        colors.push(1, 1, 1, opacity);
        colors.push(1, 1, 1, opacity);
      }
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 4));
    return geometry;
  }, [transformedNodes, edges, curvature]);

  return (
    <group>
      <lineSegments geometry={edgeBuffer}>
        <lineBasicMaterial vertexColors transparent depthWrite={false} blending={THREE.AdditiveBlending} />
      </lineSegments>

      {transformedNodes.map(node => {
        const isHovered = hoveredNode === node.id;
        const isSelected = selectedNodes.has(node.id);
        const isHighlighted = highlightOutliers && node.isOutlier;
        
        let color = node.color;
        let scale = 0.02;
        let emissiveIntensity = 0.4;
        
        if (isHovered || isSelected) {
          color = '#ffffff';
          scale = 0.04;
          emissiveIntensity = 1.5;
        } else if (isHighlighted) {
          color = '#ff0000';
          scale = 0.04;
          emissiveIntensity = 1.0;
        }

        return (
          <mesh 
            key={node.id} 
            position={node.tPos}
            onPointerOver={(e) => { e.stopPropagation(); onNodeHover(node.id); }}
            onPointerOut={(e) => { e.stopPropagation(); onNodeHover(null); }}
            onClick={(e) => { e.stopPropagation(); onNodeSelect(node.id); }}
          >
            <sphereGeometry args={[scale, 32, 32]} />
            <meshPhysicalMaterial
              color={color}
              metalness={0.9}
              roughness={0.1}
              clearcoat={1.0}
              clearcoatRoughness={0.05}
              reflectivity={1.0}
              emissive={color}
              emissiveIntensity={emissiveIntensity}
              toneMapped={false}
              envMapIntensity={1.5}
            />
            {(isHovered || isSelected) && (
              <pointLight color={color} intensity={4} distance={0.6} decay={2} />
            )}
            {isHighlighted && !isHovered && !isSelected && (
              <pointLight color="#ff0000" intensity={2} distance={0.4} decay={2} />
            )}
            
            {isHovered && (
              <Html distanceFactor={2} position={[0, 0.06, 0]} zIndexRange={[100, 0]}>
                <div className="bg-slate-900/90 backdrop-blur-sm border border-slate-700 px-2 py-1.5 rounded-md flex flex-col gap-0.5 pointer-events-none whitespace-nowrap shadow-xl">
                  <span className="font-bold text-sky-400 text-[10px] uppercase tracking-wider">{node.id}</span>
                  <span className="text-slate-300 font-mono text-[9px] uppercase tracking-widest">{node.domain}</span>
                  <span className="text-slate-400 font-mono text-[8px]">depth: {node.depth.toFixed(3)}</span>
                </div>
              </Html>
            )}
            {!isHovered && showLabels && (
              <Html distanceFactor={2} position={[0, 0.04, 0]}>
                <div className="text-[8px] text-white opacity-60 pointer-events-none whitespace-nowrap">
                  {node.id}
                </div>
              </Html>
            )}
          </mesh>
        );
      })}

      {/* Poincaré boundary shells */}
      <mesh>
        <sphereGeometry args={[1 / Math.sqrt(curvature), 32, 32]} />
        <meshBasicMaterial color="#38bdf8" wireframe transparent opacity={0.06} side={THREE.BackSide} depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>
      <mesh>
        <sphereGeometry args={[(1 / Math.sqrt(curvature)) * 0.6, 24, 24]} />
        <meshBasicMaterial color="#38bdf8" wireframe transparent opacity={0.03} side={THREE.BackSide} depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>
    </group>
  );
};

export default function Visualizer3D(props: Visualizer3DProps) {
  return (
    <div className="w-full h-full bg-[radial-gradient(circle_at_center,_#1e293b_0%,_#0f172a_100%)] relative">
      <div 
        className="absolute inset-0 opacity-10 pointer-events-none" 
        style={{ backgroundImage: 'radial-gradient(#38bdf8 1px, transparent 1px)', backgroundSize: '30px 30px' }}
      />
      <Canvas camera={{ position: [0, 0, 3], fov: 45 }} gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.2 }} className="w-full h-full cursor-grab active:cursor-grabbing relative z-10">
        <fog attach="fog" args={['#0f172a', 3, 7]} />
        <ambientLight intensity={0.3} />
        <directionalLight position={[5, 5, 5]} intensity={0.6} color="#e0f2fe" />
        <directionalLight position={[-3, -2, 4]} intensity={0.3} color="#7dd3fc" />
        <pointLight position={[0, 0, 0]} intensity={0.5} color="#38bdf8" distance={3} decay={2} />
        <Environment preset="night" />
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        <NetworkGraph {...props} />
        <OrbitControls enablePan={false} enableDamping dampingFactor={0.05} />
      </Canvas>
    </div>
  );
}
