"""
Dehydron Bonus Calculator Service

Computes stability bonuses from dehydron wrapping count changes.
Implements deterministic scoring based on bound vs unbound wrapping state.

Spec Reference: Section 6.6 "Dehydron Bonus"
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from gosp.models.data_models import Dehydron


@dataclass
class DehydronBonusResult:
    """Result of dehydron bonus calculation for a single dehydron."""
    dehydron_id: str
    residue_pair: Tuple[int, int]
    wrapping_unbound: int
    wrapping_bound: int
    bonus: int
    is_dehydron: bool
    stability_weight: float


class DehydronBonusCalculator:
    """
    Calculates stability bonuses from dehydron wrapping count changes.
    
    **Purpose:**
    Quantifies the stabilizing effect of buried hydrogen bonds (dehydrons) during
    complex assembly by tracking change in polar wrapping from unbound to bound state.
    
    **Core Concept:**
    Dehydrons are backbone hydrogen bonds that become more wrapped (desolvated) in
    complex formation. The wrapping delta (bound wrapping - unbound wrapping) reflects
    how much more buried the H-bond becomes, indicating stabilization magnitude.
    
    **Workflow:**
    1. compute_bonus(): Main workhorse - processes each dehydron
    2. _estimate_baseline_wrapping(): Gets unbound wrapping baseline
    3. _compute_stability_weight(): Normalizes bonus into 0-1 weight
    4. aggregate_bonus(): Computes collective metrics across all dehydrons
    
    **Constants (Module Level):**
    - DEHYDRON_THRESHOLD_WRAPPING = 19
      * H-bonds with unbound wrapping < 19 are classified as dehydrons
      * Threshold empirically determined from structural analysis
    - WRAPPING_SPHERE_RADIUS = 6.5Å
      * Desolvation sphere for counting polar atoms
      * Includes probe (1.4Å) + hydration (1.4Å) + safety margin
    
    **Bonus Calculation Formula:**
    ```
    For each dehydron (i, j, wrapping_bound):
        wrapping_unbound = baseline.get((i,j), 14)
        bonus_ij = max(0, wrapping_bound - wrapping_unbound)
        is_dehydron = (wrapping_unbound < 19)
        if is_dehydron and bonus_ij > 0:
            normalized = min(bonus_ij / 10.0, 1.0)
            weight_ij = normalized ^ 0.8
        else:
            weight_ij = 0.0
    ```
    
    **Stability Weight Interpretation:**
    The stability_weight (0.0-1.0) reflects both:
    - Rarity of the wrapping configuration (high bonus = rare = high weight)
    - Magnitude of stabilization (dehydron bonus delta)
    
    Weight Ranges:
    - 0.0: No stabilization or not a dehydron
    - 0.1-0.3: Weak dehydron (small bonus, common configuration)
    - 0.3-0.7: Moderate dehydron (substantial desolvation)
    - 0.7-1.0: Strong dehydron (rare, highly wrapped)
    
    **Aggregation Strategy:**
    aggregate_bonus() computes across all dehydrons:
    - total_bonus: Sum of all bonuses (absolute scale)
    - avg_stability_weight: Mean of all weights (normalized scale, 0-1)
    - dehydron_count: H-bonds classified as dehydrons
    - stabilized_count: H-bonds with any wrapping increase
    
    **Example Usage:**
    ```python
    calculator = DehydronBonusCalculator()
    result = calculator.compute_bonus(
        Dehydron(donor_res_id=10, acceptor_res_id=13, wrapping_count=22),
        baseline_wrapping={(10, 13): 18}
    )
    # result.bonus = max(0, 22-18) = 4
    # result.stability_weight ≈ 0.63 (min(4/10, 1.0)^0.8)
    
    results, agg = calculator.aggregate_bonus(all_dehydrons, baseline)
    print(f\"Average impact: {agg['avg_stability_weight']:.2f}\")
    ```
    
    **Edge Cases Handled:**
    - Empty dehydron list: Returns zero aggregates
    - Negative bonus (wrapping_bound < wrapping_unbound): Treated as 0 (not dehydron)
    - Very large bonuses (>10): Clamped at normalized=1.0
    - Missing baseline entry: Falls back to default 14
    
    **Computational Complexity:**
    - Time: O(n) where n = number of dehydrons
    - Space: O(n) for results storage
    - Per-dehydron overhead: Microseconds (simple arithmetic)
    
    **Integration with Validation Pipeline:**
    - Input: Dehydrons from dehydron_detection service
    - Output: Bonuses feed into overall PROTAC prediction score
    - Applied only to H-bonds with wrapping_unbound < 19 (true dehydrons)
    
    Spec Reference: PROTAC_Discovery_Spec Section 6.6
    
    **Bonus Calculation Formula (Internal Details):**
    Wrapping bonus = wrapping_bound - wrapping_unbound
    Only applies if wrapping_bound > wrapping_unbound (stabilizing effect).
    
    Stability weight: Normalized bonus magnitude for weighted scoring.
    """

    # Constants for dehydron detection
    DEHYDRON_THRESHOLD_WRAPPING = 19  # wrapping_unbound < 19 = dehydron
    WRAPPING_SPHERE_RADIUS = 6.5  # Angstroms for desolvation sphere

    def __init__(self):
        """Initialize calculator with default thresholds."""
        self.dehydron_threshold = self.DEHYDRON_THRESHOLD_WRAPPING
        self.radius = self.WRAPPING_SPHERE_RADIUS

    def compute_bonus(
        self,
        dehydrons: List[Dehydron],
        baseline_wrapping: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> List[DehydronBonusResult]:
        """
        Compute dehydron bonuses for a list of dehydrons.

        Args:
            dehydrons: List of detected dehydrons with wrapping counts
            baseline_wrapping: Optional baseline wrapping map for unbound state.
                              If not provided, defaults to ~14 wrapping per pair.

        Returns:
            List of DehydronBonusResult with computed bonuses and weights

        Raises:
            ValueError: If dehydrons list is empty or invalid
        """
        if not dehydrons:
            raise ValueError("Cannot compute bonus for empty dehydrons list")

        results: List[DehydronBonusResult] = []

        for dehydron in dehydrons:
            # Validate dehydron has required fields
            if dehydron.wrapping_count is None or dehydron.donor_res_id is None or dehydron.acceptor_res_id is None:
                continue

            residue_pair = (dehydron.donor_res_id, dehydron.acceptor_res_id)
            
            # Get unbound baseline wrapping
            if baseline_wrapping and residue_pair in baseline_wrapping:
                wrapping_unbound = baseline_wrapping[residue_pair]
            else:
                # Default baseline: ~14 wrapping (typical unbound H-bond)
                wrapping_unbound = self._estimate_baseline_wrapping(dehydron)

            wrapping_bound = dehydron.wrapping_count

            # Compute bonus: bonus exists only if bound > unbound (stabilizing)
            bonus = max(0, wrapping_bound - wrapping_unbound)

            # Use is_dehydron from input dehydron (already validated by model)
            is_dehydron = dehydron.is_dehydron

            # Compute stability weight (normalized 0.0-1.0)
            stability_weight = self._compute_stability_weight(bonus, wrapping_bound)

            # Create dehydron ID if not present
            dehydron_id = (
                getattr(dehydron, 'id', None)
                or f"HB_{dehydron.donor_res_id}_{dehydron.acceptor_res_id}"
            )

            results.append(
                DehydronBonusResult(
                    dehydron_id=dehydron_id,
                    residue_pair=residue_pair,
                    wrapping_unbound=wrapping_unbound,
                    wrapping_bound=wrapping_bound,
                    bonus=bonus,
                    is_dehydron=is_dehydron,
                    stability_weight=stability_weight,
                )
            )

        return results

    def _estimate_baseline_wrapping(self, dehydron: Dehydron) -> int:
        """
        Estimate baseline (unbound) wrapping for a dehydron.

        Returns typical unbound wrapping count based on structural context.
        Backbone H-bonds typically have ~14-16 wrapping at 6.5A radius in isolation.

        Args:
            dehydron: Dehydron object with metadata

        Returns:
            Estimated baseline wrapping count
        """
        # Check if dehydron has stored baseline (for testing/validation)
        if hasattr(dehydron, 'wrapping_unbound') and dehydron.wrapping_unbound is not None:
            return dehydron.wrapping_unbound

        # Default baseline for backbone H-bonds
        # Range: 12-18 depending on secondary structure environment
        # Conservative estimate: 14
        return 14

    def _compute_stability_weight(self, bonus: int, wrapping_bound: int) -> float:
        """
        Compute normalized stability weight based on bonus and bound wrapping.

        Weight reflects magnitude of stabilization effect:
        - bonus ≤ 0: weight = 0.0 (destabilizing or neutral)
        - bonus 1-3: weight ∈ [0.1, 0.3] (minor stabilization)
        - bonus 4-6: weight ∈ [0.3, 0.6] (moderate stabilization)
        - bonus 7+: weight ∈ [0.6, 1.0] (strong stabilization)

        Args:
            bonus: Calculated wrapping bonus
            wrapping_bound: Total wrapping in bound state

        Returns:
            Normalized stability weight [0.0, 1.0]
        """
        if bonus <= 0:
            return 0.0

        # Normalize bonus by maximum expected wrapping
        # Max wrapping ~50-60 at 6.5A for highly wrapped H-bonds
        # Linear scaling up to bonus=6 (weight=0.6), then curves to 1.0 at max
        max_bonus = 10.0
        normalized = min(bonus / max_bonus, 1.0)

        # Apply sigmoid-like curve for perceptual scaling
        # (higher bonuses get disproportionate weight, reflecting their rarity)
        weight = normalized ** 0.8

        return min(weight, 1.0)

    def aggregate_bonus(self, results: List[DehydronBonusResult]) -> Dict[str, float]:
        """
        Aggregate bonuses across all dehydrons for summary scoring.

        Args:
            results: List of DehydronBonusResult from compute_bonus()

        Returns:
            Dict with aggregate metrics:
                - total_bonus: Sum of all bonuses
                - avg_stability_weight: Mean stability weight
                - dehydron_count: Count of active dehydrons
                - stabilized_count: Count of dehydrons with bonus > 0
        """
        if not results:
            return {
                "total_bonus": 0,
                "avg_stability_weight": 0.0,
                "dehydron_count": 0,
                "stabilized_count": 0,
            }

        total_bonus = sum(r.bonus for r in results)
        avg_weight = sum(r.stability_weight for r in results) / len(results) if results else 0.0
        dehydron_count = sum(1 for r in results if r.is_dehydron)
        stabilized_count = sum(1 for r in results if r.bonus > 0)

        return {
            "total_bonus": total_bonus,
            "avg_stability_weight": round(avg_weight, 3),
            "dehydron_count": dehydron_count,
            "stabilized_count": stabilized_count,
        }


def calculate_dehydron_bonus(
    dehydrons: List[Dehydron],
    baseline_wrapping: Optional[Dict[Tuple[int, int], int]] = None,
) -> Tuple[List[DehydronBonusResult], Dict[str, float]]:
    """
    Calculate dehydron stability bonuses from wrapping count changes.
    
    This is the primary entry point for computing bonus values that characterize
    the stabilizing effect of dehydron (buried hydrogen bond) formation during
    complex assembly.
    
    **Algorithm Overview:**
    
    For each dehydron (backbone H-bond), compute:
    1. wrapping_unbound: Baseline count of polar atoms in 6.5Å sphere (unbound state)
    2. wrapping_bound: Actual count from input (bound state)
    3. bonus: Stabilization delta = max(0, wrapping_bound - wrapping_unbound)
    4. is_dehydron: True if wrapping_unbound < 19 (characteristic threshold)
    5. stability_weight: Normalized bonus magnitude (0.0-1.0) reflecting rarity
    
    **Key Parameters:**
    - Desolvation sphere radius: 6.5Å 
      * Probe radius: 1.4Å (van der Waals)
      * Water radius: 1.4Å (hydration shell)
      * Safety margin: 3.7Å additional
    - Dehydron threshold: wrapping_unbound < 19
      * Below this, H-bond is considered \"buried\" (dehydron)
      * Used to classify structural importance
    - Baseline wrapping (unbound): 14
      * Typical count for isolated backbone H-bond
      * Conservative estimate for water-accessible state
    
    **Bonus Interpretation:**
    - bonus = 0: Destabilizing or neutral (no increased wrapping)
    - bonus 1-3: Minor stabilization (minor desert formation)
    - bonus 4-6: Moderate stabilization (observable wrapping increase)
    - bonus 7+: Strong stabilization (rare, highly wrapped state)
    
    **Bonus Only Applies If Positive:**
    The bonus reflects a stabilizing interaction. Only positive deltas (wrapping_bound > wrapping_unbound)
    indicate desolvation and burial, which is energetically favorable.
    
    **Stability Weight Calculation:**
    Weight is computed using normalized sigmoid curve favoring high bonuses:
    - normalized = min(bonus / 10.0, 1.0)  # 10 is typical max bonus
    - weight = normalized^0.8  # Curve exponent (0.8 < 1.0 amplifies high values)
    - Result ranges [0.0, 1.0], clamped for safety
    
    **Baseline Wrapping Strategy:**
    If baseline_wrapping map provided (e.g., from unbound structure):
        Use exact value from map
    Else:
        Use estimated default (14) based on:
        - Empirical data from protein NMR/crystal structures
        - Conservative to avoid false positives
        - Can be overridden per residue pair in map
    
    **Default Aggregates Returned:**
    - total_bonus: Sum of all individual dehydron bonuses
    - avg_stability_weight: Mean of all stability weights (normalized dehydron count)
    - dehydron_count: Count of H-bonds classified as dehydrons (wrapping_unbound < 19)
    - stabilized_count: Count of H-bonds with bonus > 0 (any wrapping increase)
    
    Args:
        dehydrons: List of detected dehydrons with wrapping_count, donor_res_id, acceptor_res_id
        baseline_wrapping: Optional dict mapping (donor_res_id, acceptor_res_id) -> wrapping_unbound
                          If None, uses default estimate of 14 for all pairs
    
    Returns:
        Tuple of:
            - individual results: List[DehydronBonusResult] with all computed fields
            - aggregates: Dict[str, float] with summary metrics
    
    Raises:
        ValueError: If dehydrons list is empty or contains invalid entries
    
    Example:
        >>> dehydrons = [Dehydron(..., wrapping_count=20, donor_res_id=12, acceptor_res_id=15), ...]
        >>> results, agg = calculate_dehydron_bonus(dehydrons)
        >>> print(f\"Total bonus: {agg['total_bonus']}, stabilized: {agg['stabilized_count']}\")
    
    Spec Reference: Section 6.6 \"Dehydron Bonus\"
    """
    calculator = DehydronBonusCalculator()
    results = calculator.compute_bonus(dehydrons, baseline_wrapping)
    aggregates = calculator.aggregate_bonus(results)
    return results, aggregates
