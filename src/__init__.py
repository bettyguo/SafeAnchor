"""
SafeAnchor: Preventing Cumulative Safety Erosion in Continual Domain Adaptation
of Large Language Models.

SafeAnchor anchors safety alignment in place throughout continual domain adaptation
by combining three components:
  - Safety Subspace Identification (SSI): Fisher Information-based subspace identification
  - Orthogonal Safety-Constrained Adaptation (OSCA): gradient projection
  - Cumulative Safety Monitoring (CSM): threshold-triggered corrective replay
"""

from safeanchor.__version__ import __version__
from safeanchor.models.safeanchor import SafeAnchor
from safeanchor.models.safety_subspace import SafetySubspaceIdentifier
from safeanchor.models.osca import OrthogonalSafetyConstrainedAdapter
from safeanchor.models.csm import CumulativeSafetyMonitor


__all__ = [
    "__version__",
    "SafeAnchor",
    "SafetySubspaceIdentifier",
    "OrthogonalSafetyConstrainedAdapter",
    "CumulativeSafetyMonitor",
]
