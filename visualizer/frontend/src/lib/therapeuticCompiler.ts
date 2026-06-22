/**
 * Therapeutic Compiler Platform - Core Data Contracts & Schemas
 *
 * This module defines the canonical TypeScript interfaces and example payloads
 * for the multi-pathway (KRAS/NRAS/BRAF/MAP2K/ERK) extension of the
 * Buffering Atlas / Deterministic Therapeutic Compiler.
 *
 * These types power:
 * - Pathway Selector (checkbox layers)
 * - Atlas Engine (X=Core Frustration, Y=Relay Flux + hyperbolic Poincaré overlay)
 * - Vector Synthesizer (β-vector computation for therapeutic migration)
 * - Biochemical Translator (molecular design constraints)
 * - Manufacturing Interface (synthesis handoff)
 *
 * The design unifies Euclidean atlas (quantitative β reasoning) with
 * hyperbolic (Poincaré disc) view for resistance topology / redundancy shells.
 */

export type PathwayId =
  | 'KRAS'
  | 'NRAS'
  | 'BRAF'
  | 'MAP2K'   // MEK1/2
  | 'ERK'
  | 'HRAS';   // for completeness

export interface PathwayNode {
  id: string;                    // e.g. "KRAS_G12D"
  pathway: PathwayId;
  name: string;                  // display name
  x: number;                     // Core Frustration (X)
  y: number;                     // Relay Flux (Y)
  type?: string;                 // "Relay Inhibitor" | "Resistance Node" etc.
  color?: string;
  metadata?: {
    mutation?: string;
    ic50?: number;
    betaX?: number;
    betaY?: number;
    confidence?: number;
    source?: string;
  };
}

export interface PathwayEdge {
  source: string;
  target: string;
  relation: string;              // "Resistance Migration" | "Relay" | "Allosteric Coupling"
  weight?: number;
  style?: 'solid' | 'dashed';
}

export interface BufferingAtlasData {
  nodes: PathwayNode[];
  edges: PathwayEdge[];
  axes: {
    x_label: string;             // "Core Frustration (X)"
    y_label: string;             // "Relay Flux (Y)"
  };
  selectedPathways: PathwayId[];
  lastComputed: string;          // ISO timestamp
}

export interface TherapeuticGoal {
  id: string;
  label: string;
  description?: string;
}

export const THERAPEUTIC_GOALS: TherapeuticGoal[] = [
  { id: 'collapse_relay_redundancy', label: 'Collapse relay redundancy', description: 'Reduce Y (Bifurcador → Embudo)' },
  { id: 'increase_pocket_accessibility', label: 'Increase pocket accessibility', description: 'Raise X (more Embudo-like)' },
  { id: 'stabilize_effector_interface', label: 'Stabilize effector interface', description: 'Ancla-like: lower X and Y' },
  { id: 'prevent_resistance_migration', label: 'Prevent resistance migration', description: 'Block ΔX/ΔY escape vectors' },
];

export interface BetaVector {
  beta_x: number;                // pocket deformation requirement
  beta_y: number;                // relay-flux modulation requirement
}

export interface VectorSynthesisResult {
  goal: string;
  parameters: BetaVector;
  optimization?: {
    method: string;
    iterations: number;
    convergence: number;
  };
  result: {
    optimal_vector: BetaVector;
    score: number;
    migration?: string;          // e.g. "Bifurcador → Embudo"
  };
}

export interface MolecularFragment {
  id: string;
  structure: string;             // SMILES or InChI
  electrostatic_map?: string;    // url or base64 reference
  binding_energy?: number;
  properties?: Record<string, any>;
}

export interface MolecularDesignSpec {
  fragments: MolecularFragment[];
  assembly: {
    method: string;
    constraints: string[];
  };
  beta_vector_applied?: BetaVector;
}

export interface ManufacturingStep {
  step_id: number;
  description: string;
  temperature?: string;
  duration?: string;
  reagents?: string[];
}

export interface ManufacturingProtocol {
  protocol: {
    steps: ManufacturingStep[];
  };
  qc: {
    purity: number;
    yield: number;
    status: string;
  };
  export: {
    format: 'JSON' | 'SDF' | 'BLUEPRINT';
    destination?: string;
  };
}

/**
 * Full unified payload the frontend can request (e.g. from /api/therapeutic-compiler/state)
 */
export interface TherapeuticCompilerState {
  selectedPathways: PathwayId[];
  atlas: BufferingAtlasData;
  vector: VectorSynthesisResult | null;
  design: MolecularDesignSpec | null;
  manufacturing: ManufacturingProtocol | null;
  hyperbolic?: {
    // Poincaré disc coordinates for resistance topology / redundancy shells
    nodes: Array<{
      id: string;
      r: number;                 // radial: centrality / buffering depth
      theta: number;             // angular: pathway family / similarity
      branch_id?: string;
    }>;
    edges: Array<{ source: string; target: string; type: string }>;
  };
}

/* ------------------------------------------------------------------ */
/* Example payloads (for contract testing, mock data, and documentation) */
/* ------------------------------------------------------------------ */

export const EXAMPLE_PATHWAY_SELECTION = {
  pathways: [
    { id: 'KRAS_G12D', name: 'KRAS G12D', category: 'Oncogenic', active: true, metadata: { source: 'COSMIC', confidence: 0.92 } },
    { id: 'NRAS_Q61R', name: 'NRAS Q61R', category: 'Relay', active: true, metadata: { source: 'UniProt', confidence: 0.88 } },
    { id: 'BRAF_V600E', name: 'BRAF V600E', category: 'Amplifier', active: false },
    { id: 'MEK_K57N', name: 'MEK1 K57N', category: 'Modulator', active: true },
  ],
};

export const EXAMPLE_ATLAS = {
  nodes: [
    { id: 'KRAS_G12D', pathway: 'KRAS', name: 'KRAS G12D', x: 10.35, y: 0.087, type: 'Relay Inhibitor', color: '#00FF00' },
    { id: 'NRAS_Q61R', pathway: 'NRAS', name: 'NRAS Q61R', x: 9.8, y: 0.12, type: 'Minimalista/Bifurcador', color: '#0088FF' },
    { id: 'BRAF_V600E', pathway: 'BRAF', name: 'BRAF V600E', x: 11.4, y: 0.055, type: 'Embudo-like', color: '#FF8800' },
    { id: 'MEK_K57N', pathway: 'MAP2K', name: 'MEK1 K57N', x: 10.1, y: 0.19, type: 'Bifurcador-like', color: '#CC00CC' },
  ],
  edges: [
    { source: 'KRAS_G12D', target: 'BRAF_V600E', relation: 'Resistance Migration', style: 'dashed', weight: 0.7 },
  ],
  axes: { x_label: 'Core Frustration (X)', y_label: 'Relay Flux (Y)' },
  selectedPathways: ['KRAS', 'NRAS', 'MAP2K'],
  lastComputed: new Date().toISOString(),
};

export const EXAMPLE_VECTOR = {
  goal: 'Collapse Relay Redundancy',
  parameters: { beta_x: 0.42, beta_y: -0.18 },
  result: {
    optimal_vector: { beta_x: 0.42, beta_y: -0.18 },
    score: 0.94,
    migration: 'Bifurcador → Embudo',
  },
};

export const EXAMPLE_DESIGN: MolecularDesignSpec = {
  fragments: [
    { id: 'FRAG_001', structure: 'C1=CC=CC=C1', binding_energy: -7.2 },
    { id: 'FRAG_002', structure: 'C2H5OH', binding_energy: -5.9 },
  ],
  assembly: {
    method: 'Topological Stitching',
    constraints: ['Hydrogen Bonding', 'Steric Fit'],
  },
};

export const EXAMPLE_MANUFACTURING: ManufacturingProtocol = {
  protocol: {
    steps: [
      { step_id: 1, description: 'Combine fragments under inert atmosphere', temperature: '25°C', duration: '2h' },
      { step_id: 2, description: 'Purify by HPLC', duration: '1h' },
    ],
  },
  qc: { purity: 0.98, yield: 0.87, status: 'Ready for Execution' },
  export: { format: 'JSON', destination: 'Manufacturing API' },
};
