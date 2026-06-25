import type { DiscoveryPhase } from './discoveryPhaseMachine';

export interface ToolPolicy {
  allowed: string[];
  blocked: string[];
}

export const toolPolicyForPhase = (phase: DiscoveryPhase): ToolPolicy => {
  switch (phase) {
    case 'residue':
      return {
        allowed: ['inspect_residues', 'get_residue_state', 'set_metric', 'highlight_residues'],
        blocked: ['extract_pockets', 'map_pharmacophore_features', 'screen_fragments', 'run_docking_surrogate', 'generate_report'],
      };
    case 'topology':
      return {
        allowed: ['search_pocket_vectors', 'inspect_topology', 'graph_metrics', 'highlight_clusters', 'map_to_structure'],
        blocked: ['screen_fragments', 'run_docking_surrogate'],
      };
    case 'structure':
      return {
        allowed: ['map_to_structure', 'inspect_secondary_structure', 'extract_pockets'],
        blocked: ['run_docking_surrogate'],
      };
    case 'pocket':
      return {
        allowed: ['extract_pockets', 'map_pharmacophore_features', 'search_pocket_vectors'],
        blocked: ['generate_report'],
      };
    case 'screening':
      return {
        allowed: ['screen_fragments', 'run_docking_surrogate', 'map_pharmacophore_features'],
        blocked: [],
      };
    case 'report':
      return {
        allowed: ['generate_report', 'export_results'],
        blocked: ['extract_pockets', 'screen_fragments', 'run_docking_surrogate'],
      };
  }
};

/**
 * Check whether a tool may be invoked given the current phase policy.
 * blockedTools takes priority; if allowedTools is non-empty only listed tools pass.
 */
export function canInvokeTool(toolName: string, allowedTools: string[], blockedTools: string[]): boolean {
  if (blockedTools.includes(toolName)) return false;
  if (allowedTools.length > 0) return allowedTools.includes(toolName);
  return true;
}
