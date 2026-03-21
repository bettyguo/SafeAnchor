"""
Logging configuration for SafeAnchor training.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    use_rich: bool = True,
) -> None:
    """
    Configure logging for training runs.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write logs to file.
        use_rich: Use Rich for colorized console output.
    """
    handlers: list[logging.Handler] = []

    if use_rich:
        handlers.append(RichHandler(rich_tracebacks=True, show_path=False))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )

    # Reduce verbosity from noisy libraries
    for noisy_lib in ["transformers", "datasets", "peft", "accelerate"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


class MetricLogger:
    """
    Lightweight metric logger supporting W&B and TensorBoard.

    Args:
        experiment_name: Experiment identifier.
        wandb_cfg: W&B configuration dict (enabled, project, entity, etc.).
        tb_dir: TensorBoard log directory.
    """

    def __init__(
        self,
        experiment_name: str,
        wandb_cfg: dict[str, Any] | None = None,
        tb_dir: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self._wandb: Any = None
        self._tb: Any = None
        self._step = 0

        if wandb_cfg and wandb_cfg.get("enabled", False):
            try:
                import wandb
                wandb.init(
                    project=wandb_cfg.get("project", "safeanchor"),
                    entity=wandb_cfg.get("entity"),
                    name=experiment_name,
                    tags=wandb_cfg.get("tags", []),
                    notes=wandb_cfg.get("notes"),
                    resume="allow",
                )
                self._wandb = wandb
            except ImportError:
                logging.getLogger(__name__).warning("wandb not installed. Skipping.")

        if tb_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(tb_dir)
            except ImportError:
                logging.getLogger(__name__).warning("TensorBoard not installed. Skipping.")

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dictionary of metrics."""
        step = step if step is not None else self._step
        self._step = step + 1

        if self._wandb:
            self._wandb.log(metrics, step=step)

        if self._tb:
            for key, value in metrics.items():
                self._tb.add_scalar(key, value, step)

    def log_domain_result(self, domain_name: str, safety: float, domain: float) -> None:
        """Log post-domain evaluation results."""
        self.log({
            f"domain/{domain_name}/safety": safety,
            f"domain/{domain_name}/task": domain,
        })

    def close(self) -> None:
        """Close all logging backends."""
        if self._wandb:
            self._wandb.finish()
        if self._tb:
            self._tb.close()
