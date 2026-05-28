# Migrated from: SRC_VIZ/src/components/TokyoEyesv4/ on 2026-05-27
"""V4 GNN training scripts.

Per ADR-003: Training runs are strictly separated from inference.
These scripts produce checkpoints; they do NOT write governed data directly.
Training outputs should be clearly labeled with run_type='training'.
"""
