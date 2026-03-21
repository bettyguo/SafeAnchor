"""
Unit tests for SafeAnchor data loading utilities.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader
from unittest.mock import MagicMock, patch

from safeanchor.data.dataset import (
    DomainDataset,
    SafetyCalibrationDataset,
    SafetyProbeDataset,
    create_dataloader,
    DOMAIN_DATASET_IDS,
    SAFETY_DATASET_IDS,
)


class TestDomainDataset:
    """Tests for DomainDataset."""

    def _make_samples(self, n: int = 8) -> list[dict]:
        return [
            {"input": f"Question {i}", "output": f"Answer {i}"}
            for i in range(n)
        ]

    def _make_tokenizer(self) -> MagicMock:
        tok = MagicMock()
        seq_len = 32
        tok.return_value = {
            "input_ids": torch.randint(1, 100, (1, seq_len)),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        tok.pad_token = "[PAD]"
        tok.pad_token_id = 0
        return tok

    @pytest.mark.unit
    def test_len(self) -> None:
        tok = self._make_tokenizer()
        ds = DomainDataset(self._make_samples(8), tok, max_length=32)
        assert len(ds) == 8

    @pytest.mark.unit
    def test_max_samples_truncation(self) -> None:
        tok = self._make_tokenizer()
        ds = DomainDataset(self._make_samples(100), tok, max_length=32, max_samples=10)
        assert len(ds) == 10

    @pytest.mark.unit
    def test_output_keys(self) -> None:
        tok = self._make_tokenizer()
        ds = DomainDataset(self._make_samples(4), tok, max_length=32)
        sample = ds[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

    @pytest.mark.unit
    def test_labels_mask_padding(self) -> None:
        tok = self._make_tokenizer()
        ds = DomainDataset(self._make_samples(4), tok, max_length=32)
        sample = ds[0]
        # Padding positions should have label = -100
        padding_positions = sample["attention_mask"] == 0
        if padding_positions.any():
            assert (sample["labels"][padding_positions] == -100).all()

    @pytest.mark.unit
    def test_field_name_normalization(self) -> None:
        """Handles both 'input'/'output' and 'prompt'/'response' field names."""
        tok = self._make_tokenizer()
        samples_v2 = [{"prompt": "Q", "response": "A"} for _ in range(4)]
        ds = DomainDataset(samples_v2, tok, max_length=32)
        _ = ds[0]  # Should not raise


class TestSafetyCalibrationDataset:
    """Tests for SafetyCalibrationDataset (D_safe)."""

    def _make_safety_samples(self, n: int = 20) -> list[dict]:
        return [{"prompt": f"Harmful prompt {i}", "response": f"Refusal {i}"} for i in range(n)]

    def _make_tokenizer(self) -> MagicMock:
        tok = MagicMock()
        tok.return_value = {
            "input_ids": torch.randint(1, 100, (1, 32)),
            "attention_mask": torch.ones(1, 32, dtype=torch.long),
        }
        tok.pad_token = "[PAD]"
        return tok

    @pytest.mark.unit
    def test_n_samples_truncation(self) -> None:
        tok = self._make_tokenizer()
        ds = SafetyCalibrationDataset(self._make_safety_samples(100), tok, n_samples=500)
        assert len(ds) == min(100, 500)

    @pytest.mark.unit
    def test_default_n_samples_500(self) -> None:
        """Default N_s = 500 (from paper)."""
        tok = self._make_tokenizer()
        ds = SafetyCalibrationDataset(self._make_safety_samples(600), tok)
        assert len(ds) == 500


class TestSafetyProbeDataset:
    """Tests for SafetyProbeDataset (D_probe)."""

    def _make_probe_samples(self, n: int = 250) -> list[dict]:
        return [{"prompt": f"Attack prompt {i}"} for i in range(n)]

    def _make_tokenizer(self) -> MagicMock:
        tok = MagicMock()
        tok.return_value = {
            "input_ids": torch.randint(1, 100, (1, 64)),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
        }
        return tok

    @pytest.mark.unit
    def test_default_n_probe_200(self) -> None:
        """Default N_probe = 200 (from paper)."""
        tok = self._make_tokenizer()
        ds = SafetyProbeDataset(self._make_probe_samples(300), tok)
        assert len(ds) == 200

    @pytest.mark.unit
    def test_output_has_prompt_key(self) -> None:
        tok = self._make_tokenizer()
        ds = SafetyProbeDataset(self._make_probe_samples(10), tok, n_samples=10)
        sample = ds[0]
        assert "prompt" in sample


class TestCreateDataloader:
    """Tests for the create_dataloader factory."""

    @pytest.mark.unit
    def test_returns_dataloader(self, calibration_loader: DataLoader) -> None:
        assert isinstance(calibration_loader, DataLoader)

    @pytest.mark.unit
    def test_batch_size(self) -> None:
        from tests.conftest import _make_batch_dataloader
        loader = _make_batch_dataloader(n_samples=16, batch_size=4)
        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 4

    @pytest.mark.unit
    def test_dataset_config_coverage(self) -> None:
        """All expected domains are in DOMAIN_DATASET_IDS."""
        for domain in ["medical", "legal", "code"]:
            assert domain in DOMAIN_DATASET_IDS

    @pytest.mark.unit
    def test_safety_dataset_config_coverage(self) -> None:
        """Calibration and probe datasets are configured."""
        assert "calibration" in SAFETY_DATASET_IDS
        assert "probe" in SAFETY_DATASET_IDS
