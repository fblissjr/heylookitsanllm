"""J-space (Jacobian lens) interpretability feature.

Post-hoc read-out of a model's "verbalizable workspace": per-layer, which
vocabulary tokens a residual-stream activation is disposed toward. See
docs/jspace_integration_plan.md for the design + verifier plan.
"""
from .capture import ModelAdapter, capture_residuals
from .lens import JSpaceLens

__all__ = ["ModelAdapter", "capture_residuals", "JSpaceLens"]
