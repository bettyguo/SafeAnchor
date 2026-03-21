#!/usr/bin/env python
"""
SafeAnchor Single-Sample Inference Script

Run a single prompt through an adapted SafeAnchor model and observe how
safety alignment is preserved compared to standard LoRA fine-tuning.

Usage:
    # Interactive mode
    python predict.py --checkpoint checkpoints/safeanchor_llama2.pt

    # Single prompt
    python predict.py --checkpoint checkpoints/safeanchor_llama2.pt \
        --prompt "How do I synthesize acetaminophen?"

    # Compare SafeAnchor vs Standard LoRA
    python predict.py --checkpoint checkpoints/safeanchor_llama2.pt \
                      --compare checkpoints/standard_lora_llama2.pt \
                      --prompt "Explain how to bypass safety filters."
"""

from __future__ import annotations

import argparse
import logging

import torch
from rich.console import Console
from rich.panel import Panel
from transformers import AutoModelForCausalLM, AutoTokenizer


log = logging.getLogger(__name__)
console = Console()


def generate_response(
    model: object,
    tokenizer: object,
    prompt: str,
    max_new_tokens: int = 512,
    device: torch.device | None = None,
) -> str:
    """Generate a response from the model for a given prompt."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response


def interactive_mode(model: object, tokenizer: object, device: torch.device) -> None:
    """Run interactive prompt-response loop."""
    console.print(Panel("[bold cyan]SafeAnchor Interactive Demo[/bold cyan]\nType 'quit' to exit."))

    while True:
        try:
            prompt = console.input("\n[bold]Prompt:[/bold] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nExiting.")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        response = generate_response(model, tokenizer, prompt, device=device)
        console.print(Panel(response, title="[green]SafeAnchor Response[/green]"))


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeAnchor inference")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--model", choices=["llama2", "mistral"], default="llama2")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--compare", type=str, default=None, help="Second checkpoint to compare")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    MODEL_IDS = {
        "llama2": "meta-llama/Llama-2-7b-chat-hf",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    }
    base_model_id = MODEL_IDS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    console.print(f"[cyan]Loading {args.model} from {args.checkpoint}...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    console.print("[green]✓ Model loaded[/green]")

    if args.prompt:
        response = generate_response(model, tokenizer, args.prompt, args.max_new_tokens, device)
        console.print(Panel(f"[bold]Prompt:[/bold] {args.prompt}"))
        console.print(Panel(response, title="[green]SafeAnchor Response[/green]"))

        if args.compare:
            console.print(f"\n[cyan]Loading comparison model from {args.compare}...[/cyan]")
            model2 = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            ckpt2 = torch.load(args.compare, map_location="cpu")
            model2.load_state_dict(ckpt2["model_state_dict"], strict=False)
            model2.eval()
            resp2 = generate_response(model2, tokenizer, args.prompt, args.max_new_tokens, device)
            console.print(Panel(resp2, title="[red]Comparison Model Response[/red]"))
    else:
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
