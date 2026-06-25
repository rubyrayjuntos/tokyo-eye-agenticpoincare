import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { useDashboard } from "../lib/context";
import { api } from "../lib/api";
import type { EmbeddingData, ResidueEmbedding } from "../lib/types";

type ColorMetric = "cone_depth" | "uncertainty";

function metricToColor(r: ResidueEmbedding, metric: ColorMetric, minVal: number, maxVal: number): THREE.Color {
  const raw = metric === "cone_depth" ? r.cone_depth : r.epistemic_uncertainty;
  // Normalize to [0, 1] using actual data range
  const range = maxVal - minVal || 1;
  const t = Math.min(1, Math.max(0, (raw - minVal) / range));
  // Shallow/low = cyan (0.5), Deep/high = magenta (0.0)
  return new THREE.Color().setHSL(0.5 - t * 0.5, 0.85, 0.5 + (1 - t) * 0.15);
}

export default function LatentSpace3D() {
  const { activeStructure, refreshKey } = useDashboard();
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const frameRef = useRef<number>(0);
  const [colorMetric, setColorMetric] = useState<ColorMetric>("cone_depth");
  const [data, setData] = useState<EmbeddingData | null>(null);
  const isDragging = useRef(false);
  const prevMouse = useRef({ x: 0, y: 0 });
  const rotation = useRef({ x: 0, y: 0 });

  useEffect(() => {
    if (!activeStructure) {
      setData(null);
      return;
    }
    let cancelled = false;
    api.getEmbeddings(activeStructure.structure_id).then((result) => {
      if (!cancelled) setData(result);
    }).catch(() => {
      if (!cancelled) setData(null);
    });
    return () => { cancelled = true; };
  }, [activeStructure, refreshKey]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x09090b);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.z = 3;
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    // Mouse controls for rotation
    const onMouseDown = (e: MouseEvent) => {
      isDragging.current = true;
      prevMouse.current = { x: e.clientX, y: e.clientY };
    };
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      const dx = e.clientX - prevMouse.current.x;
      const dy = e.clientY - prevMouse.current.y;
      rotation.current.y += dx * 0.005;
      rotation.current.x += dy * 0.005;
      prevMouse.current = { x: e.clientX, y: e.clientY };
    };
    const onMouseUp = () => { isDragging.current = false; };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      camera.position.z = Math.max(1, Math.min(10, camera.position.z + e.deltaY * 0.005));
    };

    renderer.domElement.addEventListener("mousedown", onMouseDown);
    renderer.domElement.addEventListener("mousemove", onMouseMove);
    renderer.domElement.addEventListener("mouseup", onMouseUp);
    renderer.domElement.addEventListener("mouseleave", onMouseUp);
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      cancelAnimationFrame(frameRef.current);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("mousedown", onMouseDown);
      renderer.domElement.removeEventListener("mousemove", onMouseMove);
      renderer.domElement.removeEventListener("mouseup", onMouseUp);
      renderer.domElement.removeEventListener("mouseleave", onMouseUp);
      renderer.domElement.removeEventListener("wheel", onWheel);
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  // Update point cloud when data or color metric changes
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    // Remove old points
    const toRemove = scene.children.filter((c) => c.type === "Points");
    toRemove.forEach((c) => scene.remove(c));

    if (!data || data.residues.length === 0) return;

    const residues = data.residues;

    // Compute data ranges for normalization
    let minDepth = Infinity, maxDepth = -Infinity;
    let minUnc = Infinity, maxUnc = -Infinity;
    let maxR = 0; // max radial distance from origin on disc
    for (const r of residues) {
      if (r.cone_depth < minDepth) minDepth = r.cone_depth;
      if (r.cone_depth > maxDepth) maxDepth = r.cone_depth;
      if (r.epistemic_uncertainty < minUnc) minUnc = r.epistemic_uncertainty;
      if (r.epistemic_uncertainty > maxUnc) maxUnc = r.epistemic_uncertainty;
      const rad = Math.sqrt(r.x * r.x + r.y * r.y);
      if (rad > maxR) maxR = rad;
    }
    // Avoid division by zero
    if (maxR < 0.001) maxR = 1;
    const depthRange = maxDepth - minDepth || 1;

    const [minVal, maxVal] = colorMetric === "cone_depth"
      ? [minDepth, maxDepth]
      : [minUnc, maxUnc];

    const positions = new Float32Array(residues.length * 3);
    const colors = new Float32Array(residues.length * 3);

    residues.forEach((r, i) => {
      // Spread the Poincaré disc coordinates to fill the view
      // Scale so the outermost point reaches ~1.2 radius
      const scale = 1.2 / maxR;
      positions[i * 3] = r.x * scale;
      positions[i * 3 + 1] = r.y * scale;
      // Use normalized cone depth as z-axis (shallow near z=0, deep at z=-1.5)
      positions[i * 3 + 2] = -((r.cone_depth - minDepth) / depthRange) * 1.5;

      const color = metricToColor(r, colorMetric, minVal, maxVal);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.04,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.85,
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    // Animate rotation
    const animateRotation = () => {
      points.rotation.x = rotation.current.x;
      points.rotation.y = rotation.current.y;
      requestAnimationFrame(animateRotation);
    };
    const rotFrame = requestAnimationFrame(animateRotation);

    return () => {
      cancelAnimationFrame(rotFrame);
      geometry.dispose();
      material.dispose();
    };
  }, [data, colorMetric]);

  if (!activeStructure) {
    return (
      <div className="h-full flex items-center justify-center text-zinc-600 text-xs">
        Select a structure for 3D view
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between px-2 py-1 shrink-0">
        <span className="text-[10px] uppercase tracking-wider text-zinc-500">
          Latent Space 3D
        </span>
        <select
          value={colorMetric}
          onChange={(e) => setColorMetric(e.target.value as ColorMetric)}
          className="text-[10px] bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-zinc-300"
        >
          <option value="cone_depth">Cone Depth</option>
          <option value="uncertainty">Uncertainty</option>
        </select>
      </div>
      <div ref={containerRef} className="flex-1 min-h-0" />
    </div>
  );
}
