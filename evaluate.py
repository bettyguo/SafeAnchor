#!/usr/bin/env python
"""
SafeAnchor Evaluation Script

Evaluates a trained model checkpoint on all 8 benchmarks.

Usage:
    # Evaluate on all benchmarks
    python evaluate.py --checkpoint checkpoints/safeanchor_llama2.pt

    # Safety metrics only
    python evaluate.py --checkpoint checkpoints/safeanchor_llama2.pt --suite safety

    # Domain metrics only
    python evaluate.py --checkpoint checkpoints/safeanchor_llama2.pt --suite domain

    # Adversarial evaluation (GCG attack)
    python evaluate.py --checkpoint checkpoints/safeanchor_llama2.pt --adversarial

    # Specify model to evaluate
    python evaluate.py --checkpoint checkpoints/best.pt --model mistral

    # Multi-seed evaluation
    for seed in 42 123 456 789 1234; do
        python evaluate.py --checkpoint checkpoints/seed${seed}.pt --seed ${seed}
    done

Benchmark descriptions:
    Safety:
        HarmBench      — refusal rate on 200 harmful prompts
        TruthfulQA     — truthfulness score
        BBQ            — bias score (inverted in composite)
        WildGuard      — jailbreak robustness (separate from composite)

    Composite Safety Score = (1/3) × [HarmBench/100 + TruthfulQA/100 + (100−BBQ)/100] × 100

    Domain:
        MedQA          — medical question answering accuracy
        LegalBench     — legal reasoning accuracy
        HumanEval      — code generation pass@1

    General:
        MMLU           — massive multitask language understanding

    Optional:
        MT-Bench       — conversational quality (requires GPT-4 judge)
        Adversarial    — GCG attack robustness (20 steps, suffix=20, 256 cands, 100 prompts)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table

from safeanchor.data.dataset import (
    load_safety_probe_dataset,
    load_domain_dataset,
    create_dataloader,
)
from safeanchor.evaluation.evaluator import SafeAnchorEvaluator
from safeanchor.evaluation.metrics import (
    SafetyScoreAggregator,
    compute_safety_retention,
)
from safeanchor.utils.reproducibility import set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
log = logging.getLogger(__name__)
console = Console()


def load_model_from_checkpoint(
    checkpoint_path: str, model_name: str = "llama2", device: torch.device | None = None
) -> tuple[object, object]:
    """
    Load model and tokenizer from a SafeAnchor checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model_name: Model configuration name (llama2 or mistral).
        device: Target device.

    Returns:
        (model, tokenizer) tuple.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_IDS = {
        "llama2": "meta-llama/Llama-2-7b-chat-hf",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    }
    base_model_id = MODEL_IDS.get(model_name, MODEL_IDS["llama2"])

    log.info(f"Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Load LoRA weights from checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    base_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    log.info(f"Checkpoint loaded: {checkpoint_path}")

    return base_model, tokenizer


def print_results_table(result: object, baseline_safety: float = 91.4) -> None:
    """Print evaluation results in a formatted table."""
    table = Table(title=f"SafeAnchor Evaluation — {result.model_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Composite Safety", f"{result.safety.composite_safety:.1f}")
    table.add_row("  HarmBench",      f"{result.safety.harmbench:.1f}")
    table.add_row("  TruthfulQA",     f"{result.safety.truthfulqa:.1f}")
    table.add_row("  BBQ Bias",       f"{result.safety.bbq_bias:.1f}")
    table.add_row("WildGuard",        f"{result.safety.wildguard:.1f}")
    table.add_row("Safety Retention", f"{compute_safety_retention(result.safety.composite_safety, baseline_safety):.1f}%")
    table.add_row("", "")
    table.add_row("Domain (composite)", f"{result.domain.composite_domain:.1f}")
    table.add_row("  MedQA",          f"{result.domain.medqa_accuracy:.1f}")
    table.add_row("  LegalBench",     f"{result.domain.legalbench_accuracy:.1f}")
    table.add_row("  HumanEval",      f"{result.domain.humaneval_pass_at_1:.1f}")
    table.add_row("", "")
    table.add_row("MMLU", f"{result.mmlu:.1f}")

    if result.mt_bench is not None:
        table.add_row("MT-Bench", f"{result.mt_bench:.2f}")
    if result.adversarial_refusal_rate is not None:
        table.add_row("Adversarial Refusal", f"{result.adversarial_refusal_rate:.1f}%")

    console.print(table)


def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate SafeAnchor on all benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--model", choices=["llama2", "mistral"], default="llama2")
    parser.add_argument(
        "--suite",
        choices=["all", "safety", "domain"],
        default="all",
        help="Which benchmark suite to run",
    )
    parser.add_argument("--adversarial", action="store_true", help="Run GCG adversarial eval")
    parser.add_argument("--mt-bench", action="store_true", help="Run MT-Bench evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--baseline-safety",
        type=float,
        default=91.4,
        help="Baseline safety score for retention computation (91.4 for Llama-2, 88.7 for Mistral)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, tokenizer = load_model_from_checkpoint(args.checkpoint, args.model, device)

    # Load evaluation datasets
    probe_ds = load_safety_probe_dataset(tokenizer, n_samples=200)
    probe_loader = create_dataloader(probe_ds, batch_size=args.batch_size, shuffle=False)

    domain_loaders = {}
    if args.suite in ("all", "domain"):
        for domain in ["medical", "legal", "code"]:
            ds = load_domain_dataset(domain, tokenizer, max_samples=500)
            domain_loaders[domain] = create_dataloader(ds, batch_size=args.batch_size, shuffle=False)

    # Run evaluation
    evaluator = SafeAnchorEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=args.output_dir,
    )
    result = evaluator.evaluate_all(
        probe_loader=probe_loader,
        domain_loaders=domain_loaders,
        seed=args.seed,
        model_name=f"SafeAnchor_{args.model}",
        checkpoint_path=args.checkpoint,
        run_adversarial=args.adversarial,
        run_mt_bench=args.mt_bench,
    )

    # Print formatted results
    print_results_table(result, baseline_safety=args.baseline_safety)

    # Save to JSON
    out_path = Path(args.output_dir) / f"eval_results_seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
