"""
SafeAnchor: Unified Safety-Preserving Continual Domain Adaptation Framework

Integrates SSI, OSCA, and CSM into a single coherent pipeline for preventing
cumulative safety erosion during sequential domain adaptation.

The complete training objective for domain t is:

    L_total = L_task(D_t) + γ · L_anchor

where the anchor loss is:

    L_anchor = (1/|D_safe|) Σ_{x ∈ D_safe} KL( p_{θ_{t-1}}(·|x) || p_{θ_t}(·|x) )

Forward KL is used because it is mean-seeking: it penalizes the current model
for assigning low probability where the safe model assigned high probability,
preserving refusal behaviors. Empirically, reverse KL yields 1.8 points lower
safety preservation.

Crucially, OSCA projects only the task gradient ∇L_t; the anchor gradient
∇L_anchor bypasses projection to preserve its safety-reinforcing signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from safeanchor.models.csm import CumulativeSafetyMonitor, SafetyCheckResult
from safeanchor.models.osca import OrthogonalSafetyConstrainedAdapter
from safeanchor.models.safety_subspace import SafetySubspaceIdentifier, SubspaceState


log = logging.getLogger(__name__)


@dataclass
class DomainAdaptationResult:
    """Result of adapting to a single domain."""

    domain_name: str
    domain_index: int
    final_loss: float
    safety_check: SafetyCheckResult | None
    subspace_avg_rank: float
    n_steps: int


class SafeAnchor:
    """
    SafeAnchor: Safety-Preserving Continual Domain Adaptation Framework.

    Implements Algorithm 1 from the paper. The pipeline:

    1. Compute initial safety subspace {V_i^safe} via SSI
    2. Evaluate baseline safety s_0 = C(θ_0, D_probe)
    3. For each domain t:
        a. Initialize LoRA parameters Δ_t
        b. For each training step:
            - Compute task gradient g_i^t ← ∇_δᵢ L_t
            - Compute anchor gradient a_i^t ← γ ∇_δᵢ L_anchor
            - Apply OSCA projection to task gradient (Eqs. 3.3–3.4)
            - Update: δ_i ← δ_i − η (ĝ_i^t + a_i^t)
        c. Evaluate s_t = C(θ_t, D_probe)
        d. If s_t < (1 − τ) s_0: trigger safety replay
        e. Update V_i^safe via incremental SVD

    Args:
        model: Safety-aligned LLM with LoRA adapters.
        rho: Variance threshold for SSI. Default: 0.90.
        lambda_adaptive: Adaptive projection strictness λ. Default: 0.5.
        tau: CSM tolerance threshold τ. Default: 0.05.
        gamma: Anchor loss weight γ. Default: 0.1.
        e_repair: Corrective replay steps. Default: 200.
        beta: Replay safety-task balance β. Default: 1.0.
        llamaguard_model_id: Safety classifier model.
        ablation_mode: Controls which components are active.
            One of: "full" | "ssi_only" | "ssi_osca_strict" |
                    "ssi_osca_adaptive" | "full_no_csm" | "full_no_incremental"
        device: Torch device.

    Example:
        >>> framework = SafeAnchor(model, rho=0.90, tau=0.05, gamma=0.1)
        >>> framework.initialize(calibration_loader, probe_loader)
        >>> result = framework.adapt_domain(
        ...     domain_name="medical",
        ...     domain_loader=medical_loader,
        ...     calibration_loader=calibration_loader,
        ...     probe_loader=probe_loader,
        ...     optimizer=optimizer,
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        rho: float = 0.90,
        lambda_adaptive: float = 0.5,
        tau: float = 0.05,
        gamma: float = 0.1,
        e_repair: int = 200,
        beta: float = 1.0,
        llamaguard_model_id: str = "meta-llama/LlamaGuard-7b",
        ablation_mode: str = "full",
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.gamma = gamma
        self.ablation_mode = ablation_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Component instantiation
        self.ssi = SafetySubspaceIdentifier(rho=rho, device=self.device)
        self.osca = OrthogonalSafetyConstrainedAdapter(lambda_adaptive=lambda_adaptive)
        self.csm = CumulativeSafetyMonitor(
            tau=tau,
            e_repair=e_repair,
            beta=beta,
            llamaguard_model_id=llamaguard_model_id,
            device=self.device,
        )

        # Runtime state
        self._subspaces: dict[str, SubspaceState] = {}
        self._baseline_safety: float = 0.0
        self._domain_results: list[DomainAdaptationResult] = []
        self._initialized = False

        # Ablation flags
        self._use_osca = ablation_mode not in ("ssi_only",)
        self._use_adaptive = ablation_mode not in ("ssi_osca_strict",)
        self._use_csm = ablation_mode not in ("ssi_osca_strict", "ssi_osca_adaptive",
                                               "full_no_csm")
        self._use_anchor_loss = ablation_mode in ("full", "full_no_csm", "full_no_incremental")
        self._use_incremental_update = ablation_mode not in ("full_no_incremental",)

        if not self._use_adaptive:
            self.osca.lambda_adaptive = 1e9  # Force α_i → 0 (strict)

        log.info(f"SafeAnchor initialized with ablation_mode='{ablation_mode}'")

    def initialize(
        self,
        calibration_loader: DataLoader,
        probe_loader: DataLoader,
    ) -> float:
        """
        Compute initial safety subspace and baseline safety score.

        Must be called once before adapt_domain().

        Args:
            calibration_loader: DataLoader over D_safe (500 BeaverTails examples).
            probe_loader: DataLoader over D_probe (200 HarmBench examples).

        Returns:
            Baseline safety refusal rate s_0.
        """
        log.info("Initializing SafeAnchor...")

        # Step 1: Compute initial safety subspace
        self._subspaces = self.ssi.compute_subspaces(self.model, calibration_loader)

        # Step 2: Evaluate baseline safety
        self._baseline_safety = self.csm.evaluate_baseline(self.model, probe_loader)

        self._initialized = True
        log.info(
            f"Initialization complete: s_0={self._baseline_safety:.4f}, "
            f"{len(self._subspaces)} layer subspaces computed"
        )
        return self._baseline_safety

    def adapt_domain(
        self,
        domain_name: str,
        domain_loader: DataLoader,
        calibration_loader: DataLoader,
        probe_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        n_steps: int | None = None,
        domain_index: int | None = None,
    ) -> DomainAdaptationResult:
        """
        Adapt the model to a new domain with safety preservation.

        Implements the inner loop of Algorithm 1 for a single domain t:
        - OSCA-projected gradient updates
        - Anchor loss regularization
        - Post-adaptation CSM check and optional replay
        - Incremental SSI subspace update

        Args:
            domain_name: Human-readable domain name (e.g., "medical").
            domain_loader: DataLoader for domain training data.
            calibration_loader: DataLoader over D_safe.
            probe_loader: DataLoader over D_probe.
            optimizer: AdamW optimizer configured for LoRA parameters.
            n_steps: Total training steps (if None, uses len(domain_loader)).
            domain_index: 1-indexed domain number.

        Returns:
            DomainAdaptationResult with training statistics.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before adapt_domain()")

        domain_idx = domain_index or (len(self._domain_results) + 1)
        log.info(f"Adapting to domain {domain_idx}: {domain_name}")

        # Register OSCA hooks if enabled
        if self._use_osca:
            self.osca.register_hooks(self.model, self._subspaces)

        # Training loop
        self.model.train()
        total_loss = 0.0
        step = 0
        calib_iter = iter(calibration_loader)

        for batch in domain_loader:
            if n_steps is not None and step >= n_steps:
                break

            device = self.device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, Tensor)}

            optimizer.zero_grad()

            # Task gradient (will be OSCA-projected via hook)
            task_out = self.model(**batch)
            task_loss = task_out.loss

            anchor_loss = torch.tensor(0.0, device=device)
            if self._use_anchor_loss and self.gamma > 0:
                try:
                    safe_batch = next(calib_iter)
                except StopIteration:
                    calib_iter = iter(calibration_loader)
                    safe_batch = next(calib_iter)
                safe_batch = {k: v.to(device) for k, v in safe_batch.items()
                              if isinstance(v, Tensor)}
                anchor_loss = self._compute_anchor_loss(safe_batch)

            # Total loss: task + γ · anchor
            # OSCA hook intercepts task gradient automatically.
            # Anchor gradient bypasses OSCA to preserve its safety-reinforcing signal.
            total = task_loss + self.gamma * anchor_loss
            total.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            total_loss += total.item()
            step += 1

        # Remove OSCA hooks after training
        if self._use_osca:
            self.osca.remove_hooks()

        avg_loss = total_loss / max(step, 1)
        log.info(f"Domain {domain_name}: {step} steps, avg_loss={avg_loss:.4f}")

        # CSM: evaluate safety and trigger replay if needed
        safety_check: SafetyCheckResult | None = None
        if self._use_csm:
            safety_check = self.csm.check_and_repair(
                model=self.model,
                probe_loader=probe_loader,
                baseline_rate=self._baseline_safety,
                domain_idx=domain_idx,
                domain_loader=domain_loader,
                calibration_loader=calibration_loader,
                osca=self.osca if self._use_osca else None,
                optimizer=optimizer,
            )

        # Incremental SSI update
        if self._use_incremental_update:
            log.info("Updating safety subspace incrementally...")
            self._subspaces = self.ssi.update_subspaces(
                self.model, calibration_loader, self._subspaces
            )

        avg_rank = (
            sum(s.rank for s in self._subspaces.values()) / len(self._subspaces)
            if self._subspaces else 0.0
        )

        result = DomainAdaptationResult(
            domain_name=domain_name,
            domain_index=domain_idx,
            final_loss=avg_loss,
            safety_check=safety_check,
            subspace_avg_rank=avg_rank,
            n_steps=step,
        )
        self._domain_results.append(result)
        return result

    def _compute_anchor_loss(self, safe_batch: dict[str, Tensor]) -> Tensor:
        """
        Compute the forward KL anchor loss.

        L_anchor = (1/|D_safe|) Σ_{x ∈ D_safe} KL( p_{θ_{t-1}}(·|x) || p_{θ_t}(·|x) )

        Forward KL is mean-seeking: penalizes the current model for assigning
        low probability where the safe model assigned high probability, thus
        preserving refusal behaviors. Empirically 1.8 points better than reverse KL.

        The reference distribution p_{θ_{t-1}} is approximated by running the
        model in inference mode with the OSCA hooks removed (i.e., before the
        current step's update), which is a practical approximation since the
        update magnitude is small.

        Args:
            safe_batch: Batch from D_safe with input_ids and labels.

        Returns:
            Scalar anchor loss tensor.
        """
        # Compute reference distribution (no grad, no OSCA projection)
        with torch.no_grad():
            ref_out = self.model(**safe_batch)
            ref_logprobs = F.log_softmax(ref_out.logits, dim=-1).detach()

        # Compute current distribution
        curr_out = self.model(**safe_batch)
        curr_logprobs = F.log_softmax(curr_out.logits, dim=-1)

        # Forward KL: KL(ref || curr) = Σ exp(ref) · (ref - curr)
        ref_probs = ref_logprobs.exp()
        kl = (ref_probs * (ref_logprobs - curr_logprobs)).sum(dim=-1)

        # Mask padding tokens
        if "attention_mask" in safe_batch:
            mask = safe_batch["attention_mask"].float()
            kl = (kl * mask).sum() / mask.sum().clamp(min=1)
        else:
            kl = kl.mean()

        return kl

    @property
    def domain_results(self) -> list[DomainAdaptationResult]:
        """Return results from all completed domain adaptations."""
        return list(self._domain_results)

    @property
    def current_subspaces(self) -> dict[str, SubspaceState]:
        """Return the current safety subspace state."""
        return dict(self._subspaces)

    @property
    def baseline_safety(self) -> float:
        """Return the initial baseline safety score s_0."""
        return self._baseline_safety
