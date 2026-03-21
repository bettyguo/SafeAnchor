"""
Orthogonal Safety-Constrained Adaptation (OSCA)

Constrains LoRA gradient updates to the orthogonal complement of the safety
subspace, ensuring domain-specific learning occurs only in directions orthogonal
to safety-critical parameters.

For domain t and layer i, the projected gradient is:

    g̃ᵢᵗ = gᵢᵗ − Πᵢ^safe gᵢᵗ = (I − Vᵢ^safe (Vᵢ^safe)ᵀ) gᵢᵗ

An adaptive relaxation coefficient α_i reduces constraint strictness for layers
where safety and task subspaces have low overlap:

    α_i = max(0, 1 − λ · tr(F_i))

The final gradient is:

    ĝᵢᵗ = g̃ᵢᵗ + α_i · Πᵢ^safe gᵢᵗ
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from safeanchor.models.safety_subspace import SubspaceState


log = logging.getLogger(__name__)


class OrthogonalSafetyConstrainedAdapter:
    """
    Applies orthogonal safety-constrained gradient projection during training.

    OSCA hooks into the backward pass and modifies gradients of LoRA parameters
    to remove their safety-subspace component. An adaptive relaxation coefficient
    α_i = max(0, 1 − λ · tr(F_i)) reduces constraint strictness for layers with
    low safety concentration, preventing over-constraint.

    Strict projection (α=0 everywhere) recovers 79.6 ± 1.2 safety at a 4.4-point
    domain cost. Adaptive projection (default) achieves 82.4 ± 1.0 at 60.8 domain
    performance — a more favorable trade-off.

    Args:
        lambda_adaptive: λ controlling strictness of adaptive projection.
            Higher values enforce stricter constraints. Default: 0.5.

    Example:
        >>> osca = OrthogonalSafetyConstrainedAdapter(lambda_adaptive=0.5)
        >>> osca.register_hooks(model, subspaces)
        >>> # Normal training loop — gradients are projected automatically
        >>> loss.backward()
        >>> optimizer.step()
        >>> osca.remove_hooks()
    """

    def __init__(self, lambda_adaptive: float = 0.5) -> None:
        if lambda_adaptive < 0:
            raise ValueError(f"lambda_adaptive must be ≥ 0, got {lambda_adaptive}")
        self.lambda_adaptive = lambda_adaptive
        self._hooks: list[Any] = []
        self._subspaces: dict[str, SubspaceState] = {}

    def register_hooks(
        self,
        model: nn.Module,
        subspaces: dict[str, SubspaceState],
    ) -> None:
        """
        Register backward hooks on LoRA parameters.

        Each hook projects the gradient onto the orthogonal complement of the
        safety subspace when a backward pass completes on the parameter.

        Args:
            model: The model with LoRA parameters.
            subspaces: Safety subspaces keyed by parameter name.
        """
        self._subspaces = subspaces
        self._hooks.clear()

        for name, param in model.named_parameters():
            if name in subspaces and param.requires_grad:
                hook = param.register_hook(
                    self._make_projection_hook(name, subspaces[name])
                )
                self._hooks.append(hook)

        log.debug(f"Registered OSCA hooks on {len(self._hooks)} LoRA parameters")

    def remove_hooks(self) -> None:
        """Remove all registered backward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        log.debug("Removed all OSCA hooks")

    def project_gradient(
        self,
        gradient: Tensor,
        subspace: SubspaceState,
    ) -> Tensor:
        """
        Apply orthogonal projection to a single gradient tensor.

        Computes the projected gradient:
            ĝ = g̃ + α_i · Π^safe g    where g̃ = g − Π^safe g

        Equivalently:
            ĝ = g − (1 − α_i) · Π^safe g

        When α_i = 0 (strict projection): ĝ = (I − Π^safe) g
        When α_i = 1 (no projection): ĝ = g

        Args:
            gradient: Raw gradient tensor for the parameter.
            subspace: SubspaceState containing Π^safe and tr(F_i).

        Returns:
            Projected gradient with the same shape as input.
        """
        g_flat = gradient.flatten()
        proj = subspace.projection_matrix.to(g_flat.device)

        # Safety-subspace component of the gradient
        g_safety = proj @ g_flat                   # Π^safe g

        # Orthogonal component
        g_ortho = g_flat - g_safety                # (I − Π^safe) g

        # Adaptive relaxation: layers with high safety importance get α → 0
        alpha_i = max(0.0, 1.0 - self.lambda_adaptive * subspace.fisher_trace)
        alpha_i = min(alpha_i, 1.0)

        # Final projected gradient
        g_projected = g_ortho + alpha_i * g_safety

        return g_projected.reshape(gradient.shape)

    def _make_projection_hook(
        self,
        name: str,
        subspace: SubspaceState,
    ) -> Any:
        """Create a gradient hook closure for a specific LoRA layer."""
        def hook(gradient: Tensor) -> Tensor:
            return self.project_gradient(gradient, subspace)
        return hook
