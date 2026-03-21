#!/usr/bin/env python
"""
SafeAnchor Training Script

Runs the full continual domain adaptation pipeline with safety preservation.

Usage:
    # Default: Llama-2-7B-Chat, Medical → Legal → Code
    python train.py

    # Override model
    python train.py model=mistral

    # Override hyperparameters
    python train.py model.ssi.rho=0.95 model.csm.tau=0.05

    # Debug mode (fast iteration with tiny data)
    python train.py training=debug

    # Multi-GPU
    torchrun --nproc_per_node=2 train.py training=distributed

    # Ablation variants
    python train.py ablation_mode=ssi_osca_strict

    # Resume from checkpoint
    python train.py training.resume=checkpoints/latest.pt

    # Reproduce paper results
    python train.py --config-name=experiment/reproduce_table1
    python train.py --config-name=experiment/reproduce_table2

    # Run all 5 seeds
    for seed in 42 123 456 789 1234; do
        python train.py seed=$seed
    done
"""

from __future__ import annotations

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler

from safeanchor.training.trainer import SafeAnchorTrainer
from safeanchor.utils.reproducibility import print_reproducibility_info


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Main training entry point.

    Args:
        cfg: Hydra configuration (merged from configs/config.yaml + overrides).

    Returns:
        Final composite safety score (for hyperparameter optimization hooks).
    """
    OmegaConf.resolve(cfg)

    print_reproducibility_info()
    log.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Setup W&B if enabled
    if cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.get("entity"),
                name=cfg.experiment_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=cfg.wandb.get("tags", []),
            )
            log.info(f"W&B run: {wandb.run.url}")
        except ImportError:
            log.warning("wandb not installed — skipping experiment tracking")

    # Run training pipeline
    trainer = SafeAnchorTrainer(cfg)
    result = trainer.run()

    log.info("\n" + "=" * 60)
    log.info("FINAL RESULTS")
    log.info(f"  Model:          {result.model_name}")
    log.info(f"  Domains:        {' → '.join(result.domain_sequence)}")
    log.info(f"  Seed:           {result.seed}")
    log.info(f"  Safety Score:   {result.final_safety_score:.1f}")
    log.info(f"  Domain Score:   {result.final_domain_score:.1f}")
    log.info(f"  MMLU:           {result.final_mmlu:.1f}")
    log.info("=" * 60)

    if cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            wandb.log({
                "final/safety": result.final_safety_score,
                "final/domain": result.final_domain_score,
                "final/mmlu": result.final_mmlu,
            })
            wandb.finish()
        except Exception:
            pass

    return result.final_safety_score


if __name__ == "__main__":
    main()
