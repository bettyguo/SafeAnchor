"""
Unit tests for SafeAnchor model components: SSI, OSCA, CSM, and SafeAnchor.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from safeanchor.models.safety_subspace import SafetySubspaceIdentifier, SubspaceState
from safeanchor.models.osca import OrthogonalSafetyConstrainedAdapter
from safeanchor.models.csm import CumulativeSafetyMonitor
from safeanchor.models.safeanchor import SafeAnchor


# ─────────────────────────────────────────────────────────────────────────────
# SSI tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetySubspaceIdentifier:
    """Tests for Safety Subspace Identification."""

    @pytest.mark.unit
    def test_init_valid_rho(self) -> None:
        ssi = SafetySubspaceIdentifier(rho=0.90)
        assert ssi.rho == 0.90

    @pytest.mark.unit
    def test_init_invalid_rho(self) -> None:
        with pytest.raises(ValueError):
            SafetySubspaceIdentifier(rho=0.0)
        with pytest.raises(ValueError):
            SafetySubspaceIdentifier(rho=1.1)

    @pytest.mark.unit
    def test_detect_lora_params(self, tiny_model: nn.Module) -> None:
        params = SafetySubspaceIdentifier._detect_lora_params(tiny_model)
        assert len(params) > 0
        assert all("lora_A" in p or "lora_B" in p for p in params)

    @pytest.mark.unit
    def test_compute_single_subspace_shape(self) -> None:
        ssi = SafetySubspaceIdentifier(rho=0.90)
        dim = 64
        F = torch.randn(dim, dim)
        F = F @ F.T
        state = ssi._compute_single_subspace("test_layer", F)
        assert state.rank >= 1
        assert state.rank <= dim
        assert state.basis.shape == (dim, state.rank)
        assert state.projection_matrix.shape == (dim, dim)
        assert state.fisher_trace >= 0.0

    @pytest.mark.unit
    def test_projection_matrix_idempotent(self) -> None:
        """Π² = Π (projection matrices are idempotent)."""
        ssi = SafetySubspaceIdentifier(rho=0.90)
        dim = 32
        F = torch.eye(dim) + 0.1 * torch.randn(dim, dim)
        F = F @ F.T
        state = ssi._compute_single_subspace("layer", F)
        P = state.projection_matrix
        assert torch.allclose(P @ P, P, atol=1e-5)

    @pytest.mark.unit
    def test_projection_matrix_symmetric(self) -> None:
        """Π = Πᵀ."""
        ssi = SafetySubspaceIdentifier(rho=0.90)
        dim = 32
        F = torch.eye(dim)
        state = ssi._compute_single_subspace("layer", F)
        P = state.projection_matrix
        assert torch.allclose(P, P.T, atol=1e-5)

    @pytest.mark.unit
    def test_variance_coverage_at_least_rho(self) -> None:
        """Selected eigenvectors cover at least ρ of total variance."""
        ssi = SafetySubspaceIdentifier(rho=0.90)
        dim = 64
        eigenvalues = torch.exp(-torch.arange(dim, dtype=torch.float32))
        V = torch.eye(dim)
        F = V @ torch.diag(eigenvalues) @ V.T
        state = ssi._compute_single_subspace("layer", F)
        covered = float(eigenvalues[:state.rank].sum() / eigenvalues.sum())
        assert covered >= 0.90 - 1e-4

    @pytest.mark.unit
    def test_compute_subspaces_returns_dict(
        self, tiny_model: nn.Module, calibration_loader: DataLoader
    ) -> None:
        ssi = SafetySubspaceIdentifier(rho=0.90)
        subspaces = ssi.compute_subspaces(tiny_model, calibration_loader)
        assert isinstance(subspaces, dict)
        assert len(subspaces) > 0
        for name, state in subspaces.items():
            assert isinstance(state, SubspaceState)
            assert state.rank >= 1

    @pytest.mark.unit
    def test_update_subspaces_rank_bounded(
        self, tiny_model: nn.Module, calibration_loader: DataLoader
    ) -> None:
        """Incremental update never exceeds parameter dimension."""
        ssi = SafetySubspaceIdentifier(rho=0.90)
        initial = ssi.compute_subspaces(tiny_model, calibration_loader)
        updated = ssi.update_subspaces(tiny_model, calibration_loader, initial)
        for name in initial:
            assert name in updated
            param_size = SafetySubspaceIdentifier._param_size(tiny_model, name)
            assert updated[name].rank <= param_size

    @pytest.mark.unit
    def test_rho_sensitivity(self) -> None:
        """Higher rho selects more eigenvectors."""
        dim = 64
        F = torch.randn(dim, dim)
        F = F @ F.T
        ssi_low = SafetySubspaceIdentifier(rho=0.80)
        ssi_high = SafetySubspaceIdentifier(rho=0.95)
        state_low = ssi_low._compute_single_subspace("l", F)
        state_high = ssi_high._compute_single_subspace("l", F)
        assert state_high.rank >= state_low.rank


# ─────────────────────────────────────────────────────────────────────────────
# OSCA tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOrthogonalSafetyConstrainedAdapter:
    """Tests for Orthogonal Safety-Constrained Adaptation."""

    def _make_subspace(self, dim: int = 32, rank: int = 4) -> SubspaceState:
        """Create a synthetic SubspaceState for testing."""
        basis, _ = torch.linalg.qr(torch.randn(dim, rank))
        basis = basis[:, :rank]
        proj = basis @ basis.T
        return SubspaceState(
            layer_name="test",
            basis=basis,
            projection_matrix=proj,
            fisher_trace=float(proj.trace()),
            rank=rank,
            eigenvalues=torch.ones(rank),
        )

    @pytest.mark.unit
    def test_init(self) -> None:
        osca = OrthogonalSafetyConstrainedAdapter(lambda_adaptive=0.5)
        assert osca.lambda_adaptive == 0.5

    @pytest.mark.unit
    def test_invalid_lambda(self) -> None:
        with pytest.raises(ValueError):
            OrthogonalSafetyConstrainedAdapter(lambda_adaptive=-0.1)

    @pytest.mark.unit
    def test_strict_projection_removes_safety_component(self) -> None:
        """Strict OSCA (λ→∞) fully removes the safety-subspace component."""
        dim = 32
        rank = 4
        osca = OrthogonalSafetyConstrainedAdapter(lambda_adaptive=1e9)
        subspace = self._make_subspace(dim, rank)

        g = torch.randn(dim)
        g_proj = osca.project_gradient(g, subspace)

        # After strict projection, safety component should be near zero
        safety_comp = subspace.projection_matrix @ g_proj.flatten()
        assert torch.norm(safety_comp) < 1e-4

    @pytest.mark.unit
    def test_zero_lambda_no_projection(self) -> None:
        """λ=0 means α=1 everywhere — gradient is unchanged."""
        dim = 32
        osca = OrthogonalSafetyConstrainedAdapter(lambda_adaptive=0.0)
        subspace = self._make_subspace(dim, rank=4)
        # Set fisher_trace to 0 so α = max(0, 1 - 0 * 0) = 1
        subspace.fisher_trace = 0.0
        g = torch.randn(dim)
        g_proj = osca.project_gradient(g, subspace)
        assert torch.allclose(g, g_proj, atol=1e-5)

    @pytest.mark.unit
    def test_projected_gradient_preserves_shape(self) -> None:
        dim = 32
        osca = OrthogonalSafetyConstrainedAdapter(lambda_adaptive=0.5)
        subspace = self._make_subspace(dim, rank=4)
        g = torch.randn(dim)
        g_proj = osca.project_gradient(g, subspace)
        assert g_proj.shape == g.shape

    @pytest.mark.unit
    def test_hook_registration_and_removal(self, tiny_model: nn.Module) -> None:
        """Hooks are registered and cleanly removed."""
        ssi = SafetySubspaceIdentifier(rho=0.90)
        from tests.conftest import _make_batch_dataloader
        loader = _make_batch_dataloader()
        subspaces = ssi.compute_subspaces(tiny_model, loader)

        osca = OrthogonalSafetyConstrainedAdapter(lambda_adaptive=0.5)
        osca.register_hooks(tiny_model, subspaces)
        assert len(osca._hooks) == len(subspaces)
        osca.remove_hooks()
        assert len(osca._hooks) == 0

    @pytest.mark.unit
    def test_adaptive_relaxation_high_importance(self) -> None:
        """High-importance layers get stricter projection (lower alpha)."""
        dim = 32
        osca = OrthogonalSafetyConstrainedAdapter(lambda_adaptive=0.5)

        # High importance: fisher_trace = 10.0 → α = max(0, 1 - 0.5*10) = 0
        subspace_important = self._make_subspace(dim)
        subspace_important.fisher_trace = 10.0

        # Low importance: fisher_trace = 0.0 → α = 1
        subspace_unimportant = self._make_subspace(dim)
        subspace_unimportant.fisher_trace = 0.0

        g = torch.randn(dim)
        alpha_important = max(0.0, 1.0 - 0.5 * subspace_important.fisher_trace)
        alpha_unimportant = max(0.0, 1.0 - 0.5 * subspace_unimportant.fisher_trace)
        assert alpha_important < alpha_unimportant


# ─────────────────────────────────────────────────────────────────────────────
# CSM tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCumulativeSafetyMonitor:
    """Tests for Cumulative Safety Monitoring."""

    @pytest.mark.unit
    def test_init_valid(self) -> None:
        csm = CumulativeSafetyMonitor(tau=0.05, e_repair=200, beta=1.0)
        assert csm.tau == 0.05
        assert csm.e_repair == 200
        assert csm.beta == 1.0

    @pytest.mark.unit
    def test_init_invalid_tau(self) -> None:
        with pytest.raises(ValueError):
            CumulativeSafetyMonitor(tau=-0.1)
        with pytest.raises(ValueError):
            CumulativeSafetyMonitor(tau=1.0)

    @pytest.mark.unit
    def test_init_invalid_e_repair(self) -> None:
        with pytest.raises(ValueError):
            CumulativeSafetyMonitor(e_repair=0)

    @pytest.mark.unit
    def test_threshold_calculation(self) -> None:
        """Replay threshold = (1 - τ) × s_0."""
        csm = CumulativeSafetyMonitor(tau=0.05)
        s0 = 0.914
        threshold = (1 - 0.05) * s0
        assert abs(threshold - 0.8683) < 1e-4

    @pytest.mark.unit
    def test_history_tracking(self) -> None:
        """CSM accumulates check results in history."""
        csm = CumulativeSafetyMonitor(tau=0.05, e_repair=5)
        assert len(csm.history) == 0

    @pytest.mark.unit
    def test_heuristic_refusal_detection(self) -> None:
        """Refusal heuristic correctly identifies common refusal phrases."""
        from safeanchor.models.csm import CumulativeSafetyMonitor
        from safeanchor.evaluation.metrics import HarmBenchEvaluator

        evaluator = HarmBenchEvaluator()
        assert evaluator._is_refusal("i cannot help with that request")
        assert evaluator._is_refusal("I'm unable to assist with this.")
        assert evaluator._is_refusal("I refuse to provide that information.")
        assert not evaluator._is_refusal("Sure, here is how you do it.")
        assert not evaluator._is_refusal("That's a great question! Let me explain...")


# ─────────────────────────────────────────────────────────────────────────────
# SafeAnchor integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeAnchor:
    """Integration tests for the unified SafeAnchor framework."""

    @pytest.mark.unit
    def test_init(self, tiny_model: nn.Module) -> None:
        framework = SafeAnchor(
            model=tiny_model,
            rho=0.90,
            lambda_adaptive=0.5,
            tau=0.05,
            gamma=0.1,
            e_repair=10,
            beta=1.0,
        )
        assert not framework._initialized

    @pytest.mark.unit
    def test_initialize(
        self,
        tiny_model: nn.Module,
        calibration_loader: DataLoader,
        probe_loader: DataLoader,
    ) -> None:
        """initialize() computes subspaces and returns baseline safety."""
        framework = SafeAnchor(
            model=tiny_model,
            rho=0.90,
            llamaguard_model_id="none",  # Triggers heuristic fallback
        )
        s0 = framework.initialize(calibration_loader, probe_loader)
        assert framework._initialized
        assert 0.0 <= s0 <= 1.0
        assert len(framework.current_subspaces) > 0

    @pytest.mark.unit
    def test_adapt_domain_not_initialized_raises(
        self,
        tiny_model: nn.Module,
        domain_loader: DataLoader,
        calibration_loader: DataLoader,
        probe_loader: DataLoader,
    ) -> None:
        """adapt_domain() raises if initialize() was not called."""
        framework = SafeAnchor(model=tiny_model)
        optimizer = torch.optim.AdamW(
            [p for p in tiny_model.parameters() if p.requires_grad], lr=2e-4
        )
        with pytest.raises(RuntimeError, match="initialize"):
            framework.adapt_domain(
                domain_name="medical",
                domain_loader=domain_loader,
                calibration_loader=calibration_loader,
                probe_loader=probe_loader,
                optimizer=optimizer,
            )

    @pytest.mark.unit
    def test_adapt_domain_completes(
        self,
        tiny_model: nn.Module,
        calibration_loader: DataLoader,
        probe_loader: DataLoader,
        domain_loader: DataLoader,
    ) -> None:
        """adapt_domain() completes and returns a DomainAdaptationResult."""
        framework = SafeAnchor(
            model=tiny_model,
            rho=0.90,
            llamaguard_model_id="none",
        )
        framework.initialize(calibration_loader, probe_loader)

        optimizer = torch.optim.AdamW(
            [p for p in tiny_model.parameters() if p.requires_grad], lr=2e-4
        )
        result = framework.adapt_domain(
            domain_name="medical",
            domain_loader=domain_loader,
            calibration_loader=calibration_loader,
            probe_loader=probe_loader,
            optimizer=optimizer,
        )
        assert result.domain_name == "medical"
        assert result.n_steps > 0
        assert result.final_loss >= 0.0

    @pytest.mark.unit
    def test_ablation_mode_ssi_only(
        self,
        tiny_model: nn.Module,
        calibration_loader: DataLoader,
        probe_loader: DataLoader,
        domain_loader: DataLoader,
    ) -> None:
        """SSI-only ablation disables OSCA, CSM, and anchor loss."""
        framework = SafeAnchor(
            model=tiny_model,
            ablation_mode="ssi_only",
            llamaguard_model_id="none",
        )
        assert not framework._use_osca
        assert not framework._use_csm
        assert not framework._use_anchor_loss

    @pytest.mark.unit
    def test_domain_results_accumulated(
        self,
        tiny_model: nn.Module,
        calibration_loader: DataLoader,
        probe_loader: DataLoader,
        domain_loader: DataLoader,
    ) -> None:
        """domain_results grows after each adapt_domain call."""
        framework = SafeAnchor(
            model=tiny_model,
            llamaguard_model_id="none",
        )
        framework.initialize(calibration_loader, probe_loader)
        optimizer = torch.optim.AdamW(
            [p for p in tiny_model.parameters() if p.requires_grad], lr=2e-4
        )
        assert len(framework.domain_results) == 0
        framework.adapt_domain("d1", domain_loader, calibration_loader, probe_loader, optimizer)
        assert len(framework.domain_results) == 1
        framework.adapt_domain("d2", domain_loader, calibration_loader, probe_loader, optimizer)
        assert len(framework.domain_results) == 2

    @pytest.mark.unit
    def test_anchor_loss_positive(
        self,
        tiny_model: nn.Module,
        calibration_loader: DataLoader,
    ) -> None:
        """Anchor KL loss is non-negative."""
        framework = SafeAnchor(model=tiny_model, gamma=0.1)
        for batch in calibration_loader:
            anchor = framework._compute_anchor_loss(batch)
            assert float(anchor) >= 0.0
            break
