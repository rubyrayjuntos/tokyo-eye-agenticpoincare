import { setup, assign } from 'xstate';
import { toolPolicyForPhase } from './toolPolicy';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type DiscoveryPhase =
  | 'residue'
  | 'topology'
  | 'structure'
  | 'pocket'
  | 'screening'
  | 'report';

export interface DiscoveryPhaseContext {
  phase: DiscoveryPhase;

  // provenance
  phaseSource: 'default' | 'user' | 'inferred' | 'system';
  lastActor: 'user' | 'agent' | 'system' | null;
  lastTransitionAt: number | null;

  // current scientific target
  activeStructureId: string | null;
  selectedResidues: string[];
  selectedClusterId: string | null;
  selectedPocketIndex: number | null;

  // evidence flags
  hasResidueSelection: boolean;
  hasTopologySelection: boolean;
  hasStructureMapping: boolean;
  hasPocketExtraction: boolean;
  hasPharmacophoreMap: boolean;
  hasScreeningResults: boolean;
  hasDockingResults: boolean;
  hasReportDraft: boolean;

  // optional payload references
  lastScreeningRunId: string | null;
  lastDockingRunId: string | null;
  lastReportId: string | null;

  // orchestration hints (derived on entry)
  blockedTools: string[];
  allowedTools: string[];
  phaseRationale: string | null;
}

export type DiscoveryPhaseEvent =
  | { type: 'USER_SET_PHASE'; phase: DiscoveryPhase; rationale?: string }
  | { type: 'RESET_DISCOVERY_PHASE' }
  | { type: 'USER_SELECTED_RESIDUES'; residues: string[]; structureId?: string }
  | { type: 'USER_SELECTED_CLUSTER'; clusterId: string; residues?: string[] }
  | { type: 'STRUCTURE_MAPPED'; structureId: string; residues?: string[] }
  | { type: 'POCKET_EXTRACTED'; pocketIndex: number; structureId?: string }
  | { type: 'PHARMACOPHORE_MAPPED'; pocketIndex: number }
  | { type: 'SCREENING_COMPLETED'; runId: string; pocketIndex?: number }
  | { type: 'DOCKING_COMPLETED'; runId: string; pocketIndex?: number }
  | { type: 'REPORT_STARTED'; reportId?: string }
  | { type: 'REPORT_COMPLETED'; reportId: string }
  | { type: 'HYPOTHESIS_COLLAPSED'; reason: string }
  | { type: 'CLEAR_TARGET' }
  | { type: 'SYNC_FROM_VIEWPORT'; residues: string[] }
  | { type: 'SYNC_FROM_PIPELINE'; stage: string; status: 'pending' | 'active' | 'complete' | 'failed' };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const phaseRankMap: Record<DiscoveryPhase, number> = {
  residue: 0,
  topology: 1,
  structure: 2,
  pocket: 3,
  screening: 4,
  report: 5,
};

const phaseRank = (phase: DiscoveryPhase): number => phaseRankMap[phase];

// Guard helper for USER_SET_PHASE targeting a specific phase
const isUserSetTo = (target: DiscoveryPhase) =>
  ({ event }: { event: DiscoveryPhaseEvent }) =>
    event.type === 'USER_SET_PHASE' && event.phase === target;

// ---------------------------------------------------------------------------
// Machine
// ---------------------------------------------------------------------------

export const discoveryPhaseMachine = setup({
  types: {} as {
    context: DiscoveryPhaseContext;
    events: DiscoveryPhaseEvent;
  },
  guards: {
    hasResidues: ({ event }) =>
      (event.type === 'USER_SELECTED_RESIDUES' || event.type === 'SYNC_FROM_VIEWPORT') &&
      (event as { residues: string[] }).residues.length > 0,

    hasCluster: ({ event }) =>
      event.type === 'USER_SELECTED_CLUSTER' && !!event.clusterId,

    canEnterStructure: ({ context }) => {
      return context.hasResidueSelection || context.hasTopologySelection;
    },

    canEnterPocket: ({ context }) => {
      return context.hasStructureMapping;
    },

    canEnterScreening: ({ context }) => {
      return context.hasPocketExtraction || context.hasPharmacophoreMap;
    },

    canEnterReport: ({ context }) => {
      return (
        context.hasStructureMapping ||
        context.hasPocketExtraction ||
        context.hasScreeningResults ||
        context.hasDockingResults
      );
    },

    isUserSetResidue: isUserSetTo('residue'),
    isUserSetTopology: isUserSetTo('topology'),
    isUserSetStructure: isUserSetTo('structure'),
    isUserSetPocket: isUserSetTo('pocket'),
    isUserSetScreening: isUserSetTo('screening'),
    isUserSetReport: isUserSetTo('report'),
  },
  actions: {
    setUserPhase: assign(({ event }) => {
      if (event.type !== 'USER_SET_PHASE') return {};
      return {
        phase: event.phase,
        phaseSource: 'user' as const,
        lastActor: 'user' as const,
        lastTransitionAt: Date.now(),
        phaseRationale: event.rationale ?? null,
      };
    }),

    markResiduesSelected: assign(({ event }) => {
      if (event.type !== 'USER_SELECTED_RESIDUES' && event.type !== 'SYNC_FROM_VIEWPORT') return {};
      const residues = (event as { residues: string[] }).residues;
      return {
        selectedResidues: residues,
        hasResidueSelection: residues.length > 0,
        lastActor: (event.type === 'SYNC_FROM_VIEWPORT' ? 'system' : 'user') as 'user' | 'system',
        lastTransitionAt: Date.now(),
      };
    }),

    markClusterSelected: assign(({ event }) => {
      if (event.type !== 'USER_SELECTED_CLUSTER') return {};
      return {
        selectedClusterId: event.clusterId,
        selectedResidues: event.residues ?? [],
        hasTopologySelection: true,
        lastActor: 'user' as const,
        lastTransitionAt: Date.now(),
      };
    }),

    markStructureMapped: assign(({ event }) => {
      if (event.type !== 'STRUCTURE_MAPPED') return {};
      return {
        activeStructureId: event.structureId,
        selectedResidues: event.residues ?? [],
        hasStructureMapping: true,
        lastActor: 'agent' as const,
        lastTransitionAt: Date.now(),
      };
    }),

    markPocketExtracted: assign(({ event }) => {
      if (event.type !== 'POCKET_EXTRACTED') return {};
      return {
        selectedPocketIndex: event.pocketIndex,
        hasPocketExtraction: true,
        activeStructureId: event.structureId ?? null,
        lastActor: 'agent' as const,
        lastTransitionAt: Date.now(),
      };
    }),

    markPharmacophoreMapped: assign(({ event }) => {
      if (event.type !== 'PHARMACOPHORE_MAPPED') return {};
      return {
        selectedPocketIndex: event.pocketIndex,
        hasPharmacophoreMap: true,
        lastActor: 'agent' as const,
        lastTransitionAt: Date.now(),
      };
    }),

    markScreeningComplete: assign(({ event }) => {
      if (event.type !== 'SCREENING_COMPLETED') return {};
      return {
        hasScreeningResults: true,
        lastScreeningRunId: event.runId,
        selectedPocketIndex: event.pocketIndex ?? null,
        lastActor: 'agent' as const,
        lastTransitionAt: Date.now(),
      };
    }),

    markDockingComplete: assign(({ event }) => {
      if (event.type !== 'DOCKING_COMPLETED') return {};
      return {
        hasDockingResults: true,
        lastDockingRunId: event.runId,
        selectedPocketIndex: event.pocketIndex ?? null,
        lastActor: 'agent' as const,
        lastTransitionAt: Date.now(),
      };
    }),

    markReportStarted: assign(({ event }) => {
      if (event.type !== 'REPORT_STARTED') return {};
      return {
        hasReportDraft: true,
        lastReportId: event.reportId ?? null,
        lastActor: 'agent' as const,
        lastTransitionAt: Date.now(),
      };
    }),

    markReportCompleted: assign(({ event }) => {
      if (event.type !== 'REPORT_COMPLETED') return {};
      return {
        hasReportDraft: true,
        lastReportId: event.reportId,
        lastActor: 'agent' as const,
        lastTransitionAt: Date.now(),
      };
    }),

    regressPhase: assign(({ event }) => ({
      phaseSource: 'system' as const,
      lastActor: 'system' as const,
      lastTransitionAt: Date.now(),
      phaseRationale: event.type === 'HYPOTHESIS_COLLAPSED' ? event.reason : 'Target cleared',
      selectedPocketIndex: null,
      hasPocketExtraction: false,
      hasPharmacophoreMap: false,
      hasScreeningResults: false,
      hasDockingResults: false,
    })),

    clearTarget: assign(() => ({
      selectedResidues: [] as string[],
      selectedClusterId: null,
      selectedPocketIndex: null,
      hasResidueSelection: false,
      hasTopologySelection: false,
      hasStructureMapping: false,
      hasPocketExtraction: false,
      hasPharmacophoreMap: false,
      hasScreeningResults: false,
      hasDockingResults: false,
      hasReportDraft: false,
      phase: 'residue' as const,
      phaseSource: 'system' as const,
      lastActor: 'system' as const,
      lastTransitionAt: Date.now(),
      phaseRationale: 'Discovery target cleared',
    })),

    deriveToolPolicy: assign(({ context }) => {
      const policy = toolPolicyForPhase(context.phase);
      return {
        allowedTools: policy.allowed,
        blockedTools: policy.blocked,
      };
    }),
  },
}).createMachine({
  id: 'discoveryPhase',
  initial: 'residue',
  context: {
    phase: 'residue',
    phaseSource: 'default',
    lastActor: null,
    lastTransitionAt: null,
    activeStructureId: null,
    selectedResidues: [],
    selectedClusterId: null,
    selectedPocketIndex: null,
    hasResidueSelection: false,
    hasTopologySelection: false,
    hasStructureMapping: false,
    hasPocketExtraction: false,
    hasPharmacophoreMap: false,
    hasScreeningResults: false,
    hasDockingResults: false,
    hasReportDraft: false,
    lastScreeningRunId: null,
    lastDockingRunId: null,
    lastReportId: null,
    blockedTools: [],
    allowedTools: [],
    phaseRationale: 'Default session phase',
  },

  on: {
    RESET_DISCOVERY_PHASE: {
      target: '.residue',
      actions: ['clearTarget', 'deriveToolPolicy'],
    },
    CLEAR_TARGET: {
      target: '.residue',
      actions: ['clearTarget', 'deriveToolPolicy'],
    },
    SYNC_FROM_VIEWPORT: {
      guard: 'hasResidues',
      actions: 'markResiduesSelected',
    },
    SYNC_FROM_PIPELINE: {
      // Future hook for reacting to backend pipeline transitions
    },
  },

  states: {
    residue: {
      entry: [assign({ phase: () => 'residue' as const }), 'deriveToolPolicy'],
      on: {
        USER_SET_PHASE: [
          { target: 'topology', actions: 'setUserPhase', guard: 'isUserSetTopology' },
          { target: 'structure', actions: 'setUserPhase', guard: 'isUserSetStructure' },
          { target: 'pocket', actions: 'setUserPhase', guard: 'isUserSetPocket' },
          { target: 'screening', actions: 'setUserPhase', guard: 'isUserSetScreening' },
          { target: 'report', actions: 'setUserPhase', guard: 'isUserSetReport' },
        ],
        USER_SELECTED_RESIDUES: {
          target: 'residue',
          guard: 'hasResidues',
          actions: 'markResiduesSelected',
        },
        USER_SELECTED_CLUSTER: {
          target: 'topology',
          guard: 'hasCluster',
          actions: 'markClusterSelected',
        },
        STRUCTURE_MAPPED: {
          target: 'structure',
          guard: 'canEnterStructure',
          actions: 'markStructureMapped',
        },
      },
    },

    topology: {
      entry: [assign({ phase: () => 'topology' as const }), 'deriveToolPolicy'],
      on: {
        USER_SET_PHASE: [
          { target: 'residue', actions: 'setUserPhase', guard: 'isUserSetResidue' },
          { target: 'structure', actions: 'setUserPhase', guard: 'isUserSetStructure' },
          { target: 'pocket', actions: 'setUserPhase', guard: 'isUserSetPocket' },
          { target: 'screening', actions: 'setUserPhase', guard: 'isUserSetScreening' },
          { target: 'report', actions: 'setUserPhase', guard: 'isUserSetReport' },
        ],
        USER_SELECTED_RESIDUES: {
          target: 'residue',
          guard: 'hasResidues',
          actions: 'markResiduesSelected',
        },
        USER_SELECTED_CLUSTER: {
          target: 'topology',
          guard: 'hasCluster',
          actions: 'markClusterSelected',
        },
        STRUCTURE_MAPPED: {
          target: 'structure',
          guard: 'canEnterStructure',
          actions: 'markStructureMapped',
        },
        HYPOTHESIS_COLLAPSED: {
          target: 'residue',
          actions: 'regressPhase',
        },
      },
    },

    structure: {
      entry: [assign({ phase: () => 'structure' as const }), 'deriveToolPolicy'],
      on: {
        USER_SET_PHASE: [
          { target: 'residue', actions: 'setUserPhase', guard: 'isUserSetResidue' },
          { target: 'topology', actions: 'setUserPhase', guard: 'isUserSetTopology' },
          { target: 'pocket', actions: 'setUserPhase', guard: 'isUserSetPocket' },
          { target: 'screening', actions: 'setUserPhase', guard: 'isUserSetScreening' },
          { target: 'report', actions: 'setUserPhase', guard: 'isUserSetReport' },
        ],
        POCKET_EXTRACTED: {
          target: 'pocket',
          guard: 'canEnterPocket',
          actions: 'markPocketExtracted',
        },
        USER_SELECTED_CLUSTER: {
          target: 'topology',
          guard: 'hasCluster',
          actions: 'markClusterSelected',
        },
        HYPOTHESIS_COLLAPSED: {
          target: 'topology',
          actions: 'regressPhase',
        },
      },
    },

    pocket: {
      entry: [assign({ phase: () => 'pocket' as const }), 'deriveToolPolicy'],
      on: {
        USER_SET_PHASE: [
          { target: 'residue', actions: 'setUserPhase', guard: 'isUserSetResidue' },
          { target: 'topology', actions: 'setUserPhase', guard: 'isUserSetTopology' },
          { target: 'structure', actions: 'setUserPhase', guard: 'isUserSetStructure' },
          { target: 'screening', actions: 'setUserPhase', guard: 'isUserSetScreening' },
          { target: 'report', actions: 'setUserPhase', guard: 'isUserSetReport' },
        ],
        PHARMACOPHORE_MAPPED: {
          target: 'pocket',
          actions: 'markPharmacophoreMapped',
        },
        SCREENING_COMPLETED: {
          target: 'screening',
          guard: 'canEnterScreening',
          actions: 'markScreeningComplete',
        },
        HYPOTHESIS_COLLAPSED: {
          target: 'structure',
          actions: 'regressPhase',
        },
      },
    },

    screening: {
      entry: [assign({ phase: () => 'screening' as const }), 'deriveToolPolicy'],
      on: {
        USER_SET_PHASE: [
          { target: 'residue', actions: 'setUserPhase', guard: 'isUserSetResidue' },
          { target: 'topology', actions: 'setUserPhase', guard: 'isUserSetTopology' },
          { target: 'structure', actions: 'setUserPhase', guard: 'isUserSetStructure' },
          { target: 'pocket', actions: 'setUserPhase', guard: 'isUserSetPocket' },
          { target: 'report', actions: 'setUserPhase', guard: 'isUserSetReport' },
        ],
        DOCKING_COMPLETED: {
          target: 'screening',
          actions: 'markDockingComplete',
        },
        REPORT_STARTED: {
          target: 'report',
          guard: 'canEnterReport',
          actions: 'markReportStarted',
        },
        REPORT_COMPLETED: {
          target: 'report',
          guard: 'canEnterReport',
          actions: 'markReportCompleted',
        },
        HYPOTHESIS_COLLAPSED: {
          target: 'pocket',
          actions: 'regressPhase',
        },
      },
    },

    report: {
      entry: [assign({ phase: () => 'report' as const }), 'deriveToolPolicy'],
      on: {
        USER_SET_PHASE: [
          { target: 'residue', actions: 'setUserPhase', guard: 'isUserSetResidue' },
          { target: 'topology', actions: 'setUserPhase', guard: 'isUserSetTopology' },
          { target: 'structure', actions: 'setUserPhase', guard: 'isUserSetStructure' },
          { target: 'pocket', actions: 'setUserPhase', guard: 'isUserSetPocket' },
          { target: 'screening', actions: 'setUserPhase', guard: 'isUserSetScreening' },
        ],
        HYPOTHESIS_COLLAPSED: {
          target: 'screening',
          actions: 'regressPhase',
        },
      },
    },
  },
});
