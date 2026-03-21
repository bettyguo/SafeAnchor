"""
Data loading and preprocessing for SafeAnchor continual domain adaptation.

Handles three domain datasets and two safety datasets:
    Domain datasets:
        - Medical: MedQA (bigbio/med_qa)
        - Legal: LegalBench (nguha/legalbench)
        - Code: CodeAlpaca (sahil2801/CodeAlpaca-20k)
    Safety datasets:
        - Calibration (D_safe): BeaverTails (PKU-Alignment/BeaverTails), N_s=500
        - Probe (D_probe): HarmBench (walledai/HarmBench), N_probe=200
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


log = logging.getLogger(__name__)


DOMAIN_DATASET_IDS: dict[str, str] = {
    "medical": "bigbio/med_qa",
    "legal": "nguha/legalbench",
    "code": "sahil2801/CodeAlpaca-20k",
}

SAFETY_DATASET_IDS: dict[str, str] = {
    "calibration": "PKU-Alignment/BeaverTails",
    "probe": "walledai/HarmBench",
}


class DomainDataset(Dataset):
    """
    Wrapper dataset for a single domain's training data.

    Tokenizes prompt-response pairs and returns input_ids, attention_mask,
    and labels (input_ids shifted by 1 for causal LM training).

    Args:
        data: List of dict with 'input' and 'output' (or 'prompt' and 'response') keys.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length after tokenization.
        max_samples: If set, truncate to this many samples.
    """

    def __init__(
        self,
        data: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        max_samples: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        if max_samples is not None:
            data = data[:max_samples]
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        item = self.data[idx]

        # Normalize field names
        prompt = item.get("input") or item.get("prompt") or item.get("question") or ""
        response = item.get("output") or item.get("response") or item.get("answer") or ""

        text = f"{prompt}\n{response}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Causal LM labels: same as input_ids, -100 for padding
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SafetyCalibrationDataset(Dataset):
    """
    Safety calibration dataset D_safe for SSI Fisher computation.

    Contains harmful prompts with correct refusal responses AND benign prompts
    with helpful responses, sampled from BeaverTails.

    Args:
        data: List of safety examples with 'prompt' and 'response' keys.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        n_samples: Number of samples to use (default: 500, i.e., N_s).
    """

    def __init__(
        self,
        data: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        n_samples: int = 500,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data[:n_samples]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        item = self.data[idx]
        prompt = item.get("prompt", "")
        response = item.get("response", "") or item.get("output", "")
        text = f"{prompt}\n{response}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SafetyProbeDataset(Dataset):
    """
    Safety probe dataset D_probe for CSM evaluation.

    Contains 200 harmful prompts from HarmBench. Disjoint from D_safe.
    Used by LlamaGuard to evaluate the safety refusal rate s_t after each
    domain adaptation.

    Args:
        data: List with 'prompt' keys (harmful queries).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum prompt length.
        n_samples: Number of samples (default: 200, i.e., N_probe).
    """

    def __init__(
        self,
        data: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        n_samples: int = 200,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data[:n_samples]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        item = self.data[idx]
        prompt = item.get("prompt") or item.get("input") or item.get("behavior") or ""

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "prompt": prompt,
        }


def load_domain_dataset(
    domain: str,
    tokenizer: PreTrainedTokenizerBase,
    max_samples: int = 5000,
    max_length: int = 512,
    split: str = "train",
) -> DomainDataset:
    """
    Load a domain dataset from HuggingFace Hub.

    Args:
        domain: Domain name — one of "medical", "legal", "code".
        tokenizer: HuggingFace tokenizer.
        max_samples: Maximum training examples (default: 5000 per domain).
        max_length: Maximum tokenized sequence length.
        split: Dataset split to use.

    Returns:
        DomainDataset ready for DataLoader.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets>=2.18.0")

    dataset_id = DOMAIN_DATASET_IDS.get(domain)
    if dataset_id is None:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Valid domains: {list(DOMAIN_DATASET_IDS.keys())}"
        )

    log.info(f"Loading {domain} dataset from {dataset_id}...")
    raw = load_dataset(dataset_id, split=split, trust_remote_code=True)
    data = [dict(x) for x in raw]

    return DomainDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
    )


def load_safety_calibration_dataset(
    tokenizer: PreTrainedTokenizerBase,
    n_samples: int = 500,
    max_length: int = 512,
) -> SafetyCalibrationDataset:
    """
    Load BeaverTails as the safety calibration set D_safe.

    Samples N_s = 500 examples containing both harmful prompts with refusal
    responses and benign prompts with helpful responses.

    Args:
        tokenizer: HuggingFace tokenizer.
        n_samples: N_s — calibration set size (default: 500).
        max_length: Maximum sequence length.

    Returns:
        SafetyCalibrationDataset.
    """
    from datasets import load_dataset

    log.info(f"Loading BeaverTails safety calibration set (N_s={n_samples})...")
    raw = load_dataset("PKU-Alignment/BeaverTails", split="train", trust_remote_code=True)
    data = [dict(x) for x in raw]

    return SafetyCalibrationDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        n_samples=n_samples,
    )


def load_safety_probe_dataset(
    tokenizer: PreTrainedTokenizerBase,
    n_samples: int = 200,
    max_length: int = 256,
) -> SafetyProbeDataset:
    """
    Load HarmBench as the safety probe set D_probe.

    Provides 200 harmful prompts for CSM evaluation. Disjoint from D_safe.

    Args:
        tokenizer: HuggingFace tokenizer.
        n_samples: N_probe — probe set size (default: 200).
        max_length: Maximum prompt length.

    Returns:
        SafetyProbeDataset.
    """
    from datasets import load_dataset

    log.info(f"Loading HarmBench safety probe set (N_probe={n_samples})...")
    raw = load_dataset("walledai/HarmBench", split="train", trust_remote_code=True)
    data = [dict(x) for x in raw]

    return SafetyProbeDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        n_samples=n_samples,
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader with reproducible seeding.

    Args:
        dataset: PyTorch Dataset.
        batch_size: Batch size (default: 8, matching paper).
        shuffle: Whether to shuffle (True for training, False for eval).
        num_workers: Parallel data loading workers.
        seed: Random seed for worker initialization.

    Returns:
        Configured DataLoader.
    """
    from safeanchor.utils.reproducibility import get_worker_init_fn, get_generator

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=get_worker_init_fn(seed),
        generator=get_generator(seed),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
