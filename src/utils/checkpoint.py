"""
Checkpoint management for SafeAnchor training.

Saves and loads model checkpoints with full training state including:
- Model weights (LoRA adapters + optionally the full model)
- Optimizer state
- LR scheduler state
- RNG states (for exact training resumption)
- Evaluation metrics
- SafeAnchor subspace state
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from safeanchor.utils.reproducibility import get_rng_state, set_rng_state


log = logging.getLogger(__name__)


class CheckpointDict(TypedDict, total=False):
    """Type definition for checkpoint dictionary."""

    epoch: int
    global_step: int
    domain_index: int
    domain_name: str
    best_metric: float
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None
    config: dict[str, Any]
    rng_states: dict[str, Any]
    metrics_history: dict[str, list[float]]
    pytorch_version: str
    cuda_version: str | None
    timestamp: str
    seed: int


class CheckpointManager:
    """
    Manages model checkpoints with support for best-N retention.

    Args:
        checkpoint_dir: Directory to save/load checkpoints.
        keep_top_k: Number of best checkpoints to retain.
        higher_is_better: If True, larger metric = better.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        keep_top_k: int = 3,
        higher_is_better: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.higher_is_better = higher_is_better
        self.best_metric: float = float("-inf") if higher_is_better else float("inf")
        self._saved_checkpoints: list[tuple[float, Path]] = []

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        global_step: int,
        config: DictConfig,
        metric: float = 0.0,
        scheduler: LRScheduler | None = None,
        domain_index: int = 0,
        domain_name: str = "",
        seed: int = 42,
        metrics_history: dict[str, list[float]] | None = None,
    ) -> Path:
        """
        Save a training checkpoint.

        Args:
            model: Model (LoRA adapters only via PEFT save_pretrained, or full state_dict).
            optimizer: Optimizer to save state for resumption.
            epoch: Current training epoch.
            global_step: Total training steps.
            config: Hydra config (serialized to dict).
            metric: Current evaluation metric (used for best-k selection).
            scheduler: Optional LR scheduler.
            domain_index: Current domain in the sequential pipeline.
            domain_name: Current domain name.
            seed: Random seed for this run.
            metrics_history: History of metric values.

        Returns:
            Path to the saved checkpoint.
        """
        is_best = (
            metric > self.best_metric if self.higher_is_better else metric < self.best_metric
        )
        if is_best:
            self.best_metric = metric

        checkpoint: CheckpointDict = {
            "epoch": epoch,
            "global_step": global_step,
            "domain_index": domain_index,
            "domain_name": domain_name,
            "best_metric": metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": OmegaConf.to_container(config, resolve=True),  # type: ignore
            "rng_states": get_rng_state(),
            "metrics_history": metrics_history or {},
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
        }

        filename = f"step{global_step:07d}_domain{domain_index}_{domain_name}_metric{metric:.4f}.pt"
        path = self.checkpoint_dir / filename

        # Atomic write via temp file
        temp_path = path.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        temp_path.rename(path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            import shutil
            shutil.copy2(path, best_path)
            log.info(f"New best checkpoint: {path} (metric={metric:.4f})")

        # Save latest pointer
        latest_path = self.checkpoint_dir / "latest.pt"
        import shutil
        shutil.copy2(path, latest_path)

        # Prune old checkpoints
        self._saved_checkpoints.append((metric, path))
        self._prune_checkpoints()

        return path

    def load(
        self,
        path: str | Path,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        strict: bool = True,
        restore_rng: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> CheckpointDict:
        """
        Load a checkpoint and restore training state.

        Args:
            path: Path to checkpoint file.
            model: Model to restore weights into.
            optimizer: Optional optimizer for state restoration.
            scheduler: Optional scheduler for state restoration.
            strict: Strict weight loading.
            restore_rng: Whether to restore RNG states.
            map_location: Device to load tensors to.

        Returns:
            Loaded checkpoint dictionary.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        log.info(f"Loading checkpoint: {path}")
        checkpoint: CheckpointDict = torch.load(path, map_location=map_location)

        # Version compatibility warning
        ckpt_version = checkpoint.get("pytorch_version", "unknown")
        if ckpt_version != torch.__version__:
            log.warning(
                f"Checkpoint PyTorch version ({ckpt_version}) differs from "
                f"current ({torch.__version__}). Results may differ."
            )

        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if restore_rng and "rng_states" in checkpoint:
            set_rng_state(checkpoint["rng_states"])
            log.debug("RNG states restored")

        log.info(
            f"Loaded: epoch={checkpoint.get('epoch', 'N/A')}, "
            f"step={checkpoint.get('global_step', 'N/A')}, "
            f"domain={checkpoint.get('domain_name', 'N/A')}, "
            f"metric={checkpoint.get('best_metric', 'N/A'):.4f}"
        )
        return checkpoint

    def _prune_checkpoints(self) -> None:
        """Remove checkpoints beyond keep_top_k."""
        if len(self._saved_checkpoints) <= self.keep_top_k:
            return

        sorted_ckpts = sorted(
            self._saved_checkpoints,
            key=lambda x: x[0],
            reverse=self.higher_is_better,
        )
        to_remove = sorted_ckpts[self.keep_top_k:]
        for _, path in to_remove:
            if path.exists():
                path.unlink()
                log.debug(f"Pruned checkpoint: {path}")

        self._saved_checkpoints = sorted_ckpts[: self.keep_top_k]
