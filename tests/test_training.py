"""
Unit tests for the SafeAnchor training components and baselines.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from safeanchor.models.baselines import (
    StandardLoRABaseline,
    EWCLoRABaseline,
    SafetyInterleavingBaseline,
)
from safeanchor.utils.checkpoint import CheckpointManager


class TestStandardLoRABaseline:
    """Tests for Standard LoRA baseline."""

    @pytest.mark.unit
    def test_adapt_domain_returns_loss(
        self, tiny_model: nn.Module, domain_loader: DataLoader
    ) -> None:
        baseline = StandardLoRABaseline(tiny_model)
        optimizer = torch.optim.AdamW(
            [p for p in tiny_model.parameters() if p.requires_grad], lr=2e-4
        )
        loss = baseline.adapt_domain(domain_loader, optimizer)
        assert loss >= 0.0

    @pytest.mark.unit
    def test_parameters_updated(
        self, tiny_model: nn.Module, domain_loader: DataLoader
    ) -> None:
        """Training actually updates LoRA parameters."""
        baseline = StandardLoRABaseline(tiny_model)
        optimizer = torch.optim.AdamW(
            [p for p in tiny_model.parameters() if p.requires_grad], lr=1e-2
        )
        before = {n: p.clone() for n, p in tiny_model.named_parameters() if p.requires_grad}
        baseline.adapt_domain(domain_loader, optimizer)
        after = {n: p for n, p in tiny_model.named_parameters() if p.requires_grad}
        changed = any(not torch.allclose(before[n], after[n]) for n in before)
        assert changed, "No parameters were updated during training"


class TestEWCLoRABaseline:
    """Tests for EWC + LoRA baseline."""

    @pytest.mark.unit
    def test_adapt_domain_returns_loss(
        self, tiny_model: nn.Module, domain_loader: DataLoader
    ) -> None:
        baseline = EWCLoRABaseline(tiny_model, ewc_lambda=1.0)
        optimizer = torch.optim.AdamW(
            [p for p in tiny_model.parameters() if p.requires_grad], lr=2e-4
        )
        loss = baseline.adapt_domain(domain_loader, optimizer)
        assert loss >= 0.0

    @pytest.mark.unit
    def test_fisher_updated_after_domain(
        self, tiny_model: nn.Module, domain_loader: DataLoader
    ) -> None:
        """Fisher is computed after each domain."""
        baseline = EWCLoRABaseline(tiny_model, ewc_lambda=1.0)
        optimizer = torch.optim.AdamW(
            [p for p in tiny_model.parameters() if p.requires_grad], lr=2e-4
        )
        assert len(baseline._fisher) == 0
        baseline.adapt_domain(domain_loader, optimizer)
        assert len(baseline._fisher) > 0


class TestSafetyInterleavingBaseline:
    """Tests for Safety Interleaving baseline."""

    @pytest.mark.unit
    def test_adapt_domain_with_calibration(
        self,
        tiny_model: nn.Module,
        domain_loader: DataLoader,
        calibration_loader: DataLoader,
    ) -> None:
        baseline = SafetyInterleavingBaseline(tiny_model)
        optimizer = torch.optim.AdamW(
            [p for p in tiny_model.parameters() if p.requires_grad], lr=2e-4
        )
        loss = baseline.adapt_domain(domain_loader, optimizer, calibration_loader=calibration_loader)
        assert loss >= 0.0

    @pytest.mark.unit
    def test_mix_batches(self) -> None:
        """Batch mixing produces combined batch with correct proportions."""
        domain_batch = {"input_ids": torch.ones(8, 16, dtype=torch.long)}
        safe_batch = {"input_ids": torch.zeros(8, 16, dtype=torch.long)}
        mixed = SafetyInterleavingBaseline._mix_batches(domain_batch, safe_batch, 0.10)
        total = mixed["input_ids"].shape[0]
        assert total > 8  # Domain + at least 1 safety example

    @pytest.mark.unit
    def test_safety_fraction_constant(self) -> None:
        """Safety fraction is 10% as described in the paper."""
        assert SafetyInterleavingBaseline.SAFETY_FRACTION == 0.10


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.mark.unit
    def test_save_and_load(self, tmp_path, tiny_model: nn.Module) -> None:
        from omegaconf import OmegaConf

        manager = CheckpointManager(checkpoint_dir=tmp_path)
        optimizer = torch.optim.AdamW(
            [p for p in tiny_model.parameters() if p.requires_grad], lr=2e-4
        )
        cfg = OmegaConf.create({"seed": 42, "model": {"name": "test"}})

        saved_path = manager.save(
            model=tiny_model,
            optimizer=optimizer,
            epoch=1,
            global_step=100,
            config=cfg,
            metric=85.2,
            domain_index=1,
            domain_name="medical",
            seed=42,
        )

        assert saved_path.exists()
        assert (tmp_path / "best.pt").exists()
        assert (tmp_path / "latest.pt").exists()

    @pytest.mark.unit
    def test_best_checkpoint_updated(self, tmp_path, tiny_model: nn.Module) -> None:
        from omegaconf import OmegaConf

        manager = CheckpointManager(checkpoint_dir=tmp_path)
        optimizer = torch.optim.AdamW(
            [p for p in tiny_model.parameters() if p.requires_grad], lr=2e-4
        )
        cfg = OmegaConf.create({"seed": 42})

        manager.save(tiny_model, optimizer, 1, 100, cfg, metric=70.0)
        assert manager.best_metric == 70.0
        manager.save(tiny_model, optimizer, 2, 200, cfg, metric=85.2)
        assert manager.best_metric == 85.2

    @pytest.mark.unit
    def test_load_nonexistent_raises(self, tmp_path, tiny_model: nn.Module) -> None:
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            manager.load(tmp_path / "nonexistent.pt", tiny_model)
