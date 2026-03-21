"""
SafeAnchor Trainer

Orchestrates the full continual domain adaptation pipeline:
    1. Load base model and apply LoRA adapters (PEFT, r=16, α=32)
    2. Initialize SafeAnchor (compute SSI subspaces + baseline safety)
    3. Sequentially adapt to each domain with OSCA + anchor loss
    4. Run CSM check after each domain
    5. Update safety subspace incrementally
    6. Evaluate on all 8 benchmarks

Training configuration (from paper):
    - LoRA rank r=16, alpha=32, targets Q/K/V/O projections
    - Learning rate 2e-4 with cosine schedule
    - Batch size 8, AdamW optimizer
    - 5,000 training examples per domain, 3 epochs
    - ρ=0.90, τ=0.05, γ=0.1, λ=0.5, β=1.0
    - 2× A100 40GB GPUs, ~8h total pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from safeanchor.data.dataset import (
    load_domain_dataset,
    load_safety_calibration_dataset,
    load_safety_probe_dataset,
    create_dataloader,
)
from safeanchor.evaluation.evaluator import SafeAnchorEvaluator
from safeanchor.models.safeanchor import SafeAnchor, DomainAdaptationResult
from safeanchor.utils.checkpoint import CheckpointManager
from safeanchor.utils.reproducibility import set_seed


log = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Outcome of the full continual adaptation pipeline."""

    model_name: str
    domain_sequence: list[str]
    seed: int
    domain_results: list[DomainAdaptationResult]
    final_safety_score: float
    final_domain_score: float
    final_mmlu: float
    checkpoint_path: str | None


class SafeAnchorTrainer:
    """
    Trainer for the SafeAnchor continual domain adaptation pipeline.

    Handles model loading, LoRA setup, training loop, and evaluation.

    Args:
        cfg: Hydra configuration dictionary.
        device: Torch device for training.

    Example:
        >>> trainer = SafeAnchorTrainer(cfg)
        >>> result = trainer.run()
        >>> print(f"Final safety: {result.final_safety_score:.1f}")
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def run(self) -> TrainingResult:
        """
        Execute the full SafeAnchor continual domain adaptation pipeline.

        Returns:
            TrainingResult with domain results and final evaluation scores.
        """
        # Set random seed for reproducibility
        set_seed(self.cfg.seed, deterministic=self.cfg.get("deterministic", True))
        log.info(f"Seed: {self.cfg.seed}")

        # Load model and tokenizer
        model, tokenizer = self._load_model()

        # Load datasets
        calibration_loader, probe_loader, domain_loaders = self._load_datasets(tokenizer)

        # Initialize SafeAnchor framework
        framework = SafeAnchor(
            model=model,
            rho=self.cfg.model.ssi.rho,
            lambda_adaptive=self.cfg.model.ssi.lambda_adaptive,
            tau=self.cfg.model.csm.tau,
            gamma=self.cfg.model.gamma,
            e_repair=self.cfg.model.csm.e_repair,
            beta=self.cfg.model.csm.beta,
            llamaguard_model_id=self.cfg.safety.llamaguard_model,
            ablation_mode=self.cfg.get("ablation_mode", "full"),
            device=self.device,
        )

        # Initialize: compute subspaces + baseline safety
        baseline_safety = framework.initialize(calibration_loader, probe_loader)
        log.info(f"Baseline safety (s_0): {baseline_safety:.4f}")

        # Sequential domain adaptation
        domain_results = []
        for domain_idx, domain_name in enumerate(self.cfg.domain_order, start=1):
            log.info(f"\n{'='*60}")
            log.info(f"Domain {domain_idx}/{len(self.cfg.domain_order)}: {domain_name.upper()}")
            log.info(f"{'='*60}")

            domain_loader = domain_loaders[domain_name]
            optimizer = self._make_optimizer(model)
            scheduler = self._make_scheduler(
                optimizer, n_steps=self._compute_n_steps(domain_loader)
            )

            result = framework.adapt_domain(
                domain_name=domain_name,
                domain_loader=domain_loader,
                calibration_loader=calibration_loader,
                probe_loader=probe_loader,
                optimizer=optimizer,
                domain_index=domain_idx,
            )
            domain_results.append(result)

            if result.safety_check:
                log.info(
                    f"Post-domain safety: {result.safety_check.refusal_rate:.4f} "
                    f"(baseline: {result.safety_check.baseline_rate:.4f})"
                )

        # Final evaluation
        evaluator = SafeAnchorEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            output_dir=self.cfg.get("output_dir", "outputs"),
        )
        eval_result = evaluator.evaluate_all(
            probe_loader=probe_loader,
            domain_loaders={d: domain_loaders[d] for d in self.cfg.domain_order},
            seed=self.cfg.seed,
            model_name=f"SafeAnchor_{self.cfg.model.name}",
        )

        # Save checkpoint
        checkpoint_path = self._save_checkpoint(model, eval_result.safety.composite_safety)

        log.info("\n" + "=" * 60)
        log.info("TRAINING COMPLETE")
        log.info(f"  Composite Safety: {eval_result.safety.composite_safety:.1f}")
        log.info(f"  Domain Score:     {eval_result.domain.composite_domain:.1f}")
        log.info(f"  MMLU:             {eval_result.mmlu:.1f}")
        log.info(
            f"  Safety Retention: "
            f"{(eval_result.safety.composite_safety / (baseline_safety * 100)):.1%}"
        )
        log.info("=" * 60)

        return TrainingResult(
            model_name=self.cfg.model.name,
            domain_sequence=list(self.cfg.domain_order),
            seed=self.cfg.seed,
            domain_results=domain_results,
            final_safety_score=eval_result.safety.composite_safety,
            final_domain_score=eval_result.domain.composite_domain,
            final_mmlu=eval_result.mmlu,
            checkpoint_path=checkpoint_path,
        )

    def _load_model(self) -> tuple[nn.Module, Any]:
        """Load base model and apply LoRA adapters via PEFT."""
        from peft import LoraConfig, get_peft_model, TaskType

        log.info(f"Loading base model: {self.cfg.model.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.base_model,
            padding_side="right",
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=False,
        )

        # Apply LoRA adapters (PEFT)
        lora_cfg = self.cfg.model.lora
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=list(lora_cfg.target_modules),
            bias=lora_cfg.bias,
        )
        model = get_peft_model(model, peft_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        if self.cfg.training.get("gradient_checkpointing", False):
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()

        return model, tokenizer

    def _load_datasets(
        self, tokenizer: Any
    ) -> tuple[DataLoader, DataLoader, dict[str, DataLoader]]:
        """Load all domain, calibration, and probe datasets."""
        train_cfg = self.cfg.training
        safety_cfg = self.cfg.safety

        # Safety datasets
        calibration_ds = load_safety_calibration_dataset(
            tokenizer,
            n_samples=safety_cfg.n_calibration,
            max_length=train_cfg.max_seq_length,
        )
        probe_ds = load_safety_probe_dataset(
            tokenizer,
            n_samples=safety_cfg.n_probe,
        )
        calibration_loader = create_dataloader(
            calibration_ds, batch_size=train_cfg.batch_size, shuffle=True
        )
        probe_loader = create_dataloader(
            probe_ds, batch_size=train_cfg.batch_size, shuffle=False
        )

        # Domain datasets
        domain_loaders: dict[str, DataLoader] = {}
        for domain in self.cfg.domain_order:
            ds = load_domain_dataset(
                domain=domain,
                tokenizer=tokenizer,
                max_samples=train_cfg.train_samples_per_domain,
                max_length=train_cfg.max_seq_length,
            )
            domain_loaders[domain] = create_dataloader(
                ds, batch_size=train_cfg.batch_size, shuffle=True
            )

        return calibration_loader, probe_loader, domain_loaders

    def _make_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create AdamW optimizer for LoRA parameters."""
        train_cfg = self.cfg.training
        opt_cfg = train_cfg.optimizer
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
            eps=opt_cfg.eps,
        )

    def _make_scheduler(
        self, optimizer: torch.optim.Optimizer, n_steps: int
    ) -> Any:
        """Create cosine learning rate scheduler with warmup."""
        from transformers import get_cosine_schedule_with_warmup

        warmup = int(n_steps * self.cfg.training.scheduler.warmup_ratio)
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=n_steps,
        )

    def _compute_n_steps(self, loader: DataLoader) -> int:
        """Compute total training steps for a domain."""
        return len(loader) * self.cfg.training.epochs_per_domain

    def _save_checkpoint(self, model: nn.Module, metric: float) -> str | None:
        """Save model checkpoint if output directory is configured."""
        output_dir = self.cfg.get("checkpoint_dir", "checkpoints")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / f"safeanchor_{self.cfg.model.name}_seed{self.cfg.seed}.pt"
        torch.save({"model_state_dict": model.state_dict(), "metric": metric}, path)
        log.info(f"Checkpoint saved to {path}")
        return str(path)


def main() -> None:
    """CLI entry point. Used by safeanchor-train console script."""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
    def _main(cfg: DictConfig) -> None:
        trainer = SafeAnchorTrainer(cfg)
        trainer.run()

    _main()
