# Migrated from: SRC_VIZ/src/components/TokyoEyesv5/ on 2026-05-27
"""V5 GNN training scripts.

V5 training uses the same loss functions as v4 but routes them
through the decoupled heads:
- Cone loss → RadialHead only
- Domain sep + angular diversity + neighborhood → AngularHead only
- Backbone receives gradients from both
"""
