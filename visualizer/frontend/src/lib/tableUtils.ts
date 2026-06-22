/**
 * tableUtils — Generic sort and filter utilities for the Data Inspector
 * residue table. Pure functions, easy to property-test.
 */

import type { MergedResidueRow } from "./mergeResidueData";

// ---------------------------------------------------------------------------
// Sort
// ---------------------------------------------------------------------------

export type SortDirection = "asc" | "desc";

export interface SortConfig {
  column: keyof MergedResidueRow;
  direction: SortDirection;
}

/**
 * Sort rows by the given column. Null values are always sorted to the end
 * regardless of direction.
 */
export function sortRows(
  rows: MergedResidueRow[],
  config: SortConfig,
): MergedResidueRow[] {
  const { column, direction } = config;
  const multiplier = direction === "asc" ? 1 : -1;

  return [...rows].sort((a, b) => {
    const aVal = a[column];
    const bVal = b[column];

    // Nulls always last
    if (aVal === null && bVal === null) return 0;
    if (aVal === null) return 1;
    if (bVal === null) return -1;

    // Boolean comparison
    if (typeof aVal === "boolean" && typeof bVal === "boolean") {
      return multiplier * (Number(aVal) - Number(bVal));
    }

    // Numeric comparison
    if (typeof aVal === "number" && typeof bVal === "number") {
      return multiplier * (aVal - bVal);
    }

    // String comparison
    if (typeof aVal === "string" && typeof bVal === "string") {
      return multiplier * aVal.localeCompare(bVal);
    }

    return 0;
  });
}

// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------

export interface FilterCriteria {
  /** Filter to specific chain(s) */
  chain?: string | null;
  /** Min epistemic uncertainty */
  minUncertainty?: number | null;
  /** Max epistemic uncertainty */
  maxUncertainty?: number | null;
  /** Min cone depth */
  minDepth?: number | null;
  /** Max cone depth */
  maxDepth?: number | null;
  /** Filter by resistance classification */
  classification?: "high_sensitivity" | "moderate" | "stable" | null;
}

/**
 * Filter rows by the given criteria. All criteria are AND-combined.
 * A null/undefined criterion means "no filter on this dimension".
 */
export function filterRows(
  rows: MergedResidueRow[],
  criteria: FilterCriteria,
): MergedResidueRow[] {
  return rows.filter((row) => {
    // Chain filter
    if (criteria.chain != null && criteria.chain !== "") {
      if (row.chain_label !== criteria.chain) return false;
    }

    // Uncertainty range (epistemic)
    if (criteria.minUncertainty != null) {
      if (row.epistemic_uncertainty === null) return false;
      if (row.epistemic_uncertainty < criteria.minUncertainty) return false;
    }
    if (criteria.maxUncertainty != null) {
      if (row.epistemic_uncertainty === null) return false;
      if (row.epistemic_uncertainty > criteria.maxUncertainty) return false;
    }

    // Depth range
    if (criteria.minDepth != null) {
      if (row.cone_depth === null) return false;
      if (row.cone_depth < criteria.minDepth) return false;
    }
    if (criteria.maxDepth != null) {
      if (row.cone_depth === null) return false;
      if (row.cone_depth > criteria.maxDepth) return false;
    }

    // Classification filter
    if (criteria.classification != null) {
      if (row.classification !== criteria.classification) return false;
    }

    return true;
  });
}
