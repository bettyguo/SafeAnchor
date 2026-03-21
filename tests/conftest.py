"""
Shared pytest fixtures for SafeAnchor test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: mock model
# ─────────────────────────────────────────────────────────────────────────────

class TinyLoRAModel(nn.Module):
    """
    Minimal LoRA-like model for fast unit testing.
    Has lora_A and lora_B parameters matching the naming convention
    used by PEFT, so SSI and OSCA can detect them automatically.
    """

    def __init__(self, vocab_size: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        # Simulate LoRA attention projections
        self.q_proj_lora_A = nn.Parameter(torch.randn(hidden // 2, hidden) * 0.01)
        self.q_proj_lora_B = nn.Parameter(torch.zeros(hidden, hidden // 2))
        self.v_proj_lora_A = nn.Parameter(torch.randn(hidden // 2, hidden) * 0.01)
        self.v_proj_lora_B = nn.Parameter(torch.zeros(hidden, hidden // 2))
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        x = self.embedding(input_ids)
        logits = self.lm_head(x)

        output = MagicMock()
        output.logits = logits

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.shape[-1])
            shift_labels = labels[..., 1:].contiguous().view(-1)
            output.loss = self.loss_fct(shift_logits, shift_labels)
        else:
            output.loss = torch.tensor(0.1)

        return output


@pytest.fixture
def tiny_model() -> TinyLoRAModel:
    """A minimal model with LoRA-named parameters for fast testing."""
    return TinyLoRAModel(vocab_size=32, hidden=16)


@pytest.fixture
def device() -> torch.device:
    """CPU device (GPU tests are marked with @pytest.mark.gpu)."""
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _make_batch_dataloader(
    n_samples: int = 16,
    seq_len: int = 32,
    vocab_size: int = 32,
    batch_size: int = 4,
) -> DataLoader:
    """Create a DataLoader of random token batches."""
    input_ids = torch.randint(1, vocab_size, (n_samples, seq_len))
    attention_mask = torch.ones(n_samples, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    dataset = TensorDataset(input_ids, attention_mask, labels)

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, ds: TensorDataset) -> None:
            self.ds = ds

        def __len__(self) -> int:
            return len(self.ds)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            ii, am, lb = self.ds[idx]
            return {"input_ids": ii, "attention_mask": am, "labels": lb}

    return DataLoader(DictDataset(dataset), batch_size=batch_size, shuffle=False)


@pytest.fixture
def calibration_loader() -> DataLoader:
    """Simulated D_safe calibration DataLoader (16 examples)."""
    return _make_batch_dataloader(n_samples=16, seq_len=32, batch_size=4)


@pytest.fixture
def probe_loader() -> DataLoader:
    """Simulated D_probe DataLoader (16 examples)."""
    return _make_batch_dataloader(n_samples=16, seq_len=32, batch_size=4)


@pytest.fixture
def domain_loader() -> DataLoader:
    """Simulated domain training DataLoader (32 examples)."""
    return _make_batch_dataloader(n_samples=32, seq_len=32, batch_size=4)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: configs
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Load the sample configuration fixture."""
    config_path = Path(__file__).parent / "fixtures" / "sample_config.yaml"
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Markers
# ─────────────────────────────────────────────────────────────────────────────

def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")
