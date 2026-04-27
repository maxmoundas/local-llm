#!/usr/bin/env python
"""
Download a Gemma (or any HuggingFace) model with progress display.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --repo-id google/gemma-4-4b-it
    python scripts/download_model.py --token hf_xxxx
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import typer
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config

console = Console()
app = typer.Typer(add_completion=False)


VRAM_REQUIREMENTS = {
    "1b": {"4bit": 1, "8bit": 2, "none": 3},
    "4b": {"4bit": 3, "8bit": 5, "none": 9},
    "12b": {"4bit": 7, "8bit": 13, "none": 25},
    "27b": {"4bit": 15, "8bit": 28, "none": 55},
    "31b": {"4bit": 17, "8bit": 32, "none": 63},
}


@app.command()
def download(
    repo_id: str = typer.Option(None, "--repo-id", "-r", help="HuggingFace model repo ID"),
    token: str = typer.Option(None, "--token", "-t", help="HuggingFace API token"),
    cache_dir: str = typer.Option(None, "--cache-dir", help="Directory to save model"),
    config_path: str = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    skip_hardware_check: bool = typer.Option(False, "--skip-check", help="Skip hardware warning"),
) -> None:
    """Download a model from HuggingFace Hub."""
    cfg = load_config(config_path)
    repo_id = repo_id or cfg["model"]["repo_id"]
    cache_dir = cache_dir or cfg["model"].get("cache_dir", "./models")
    quantization = cfg["model"].get("quantization", "4bit")
    token = token or os.environ.get("HF_TOKEN")

    console.print(
        Panel(
            f"[bold]Model:[/] [green]{repo_id}[/]\n"
            f"[bold]Destination:[/] [yellow]{cache_dir}[/]\n"
            f"[bold]Quantization:[/] [cyan]{quantization}[/]",
            title="Local LLM — Model Download",
            border_style="blue",
        )
    )

    _check_token(token)
    if not skip_hardware_check:
        _hardware_check(repo_id, quantization)
    _check_disk_space(cache_dir, repo_id)

    console.print(f"\n[bold green]Downloading {repo_id}...[/]")
    console.print("[dim]This may take a while for large models. Progress is shown below.[/]\n")

    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )
        console.print(f"\n[bold green]✓ Downloaded to:[/] {local_path}")
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "403" in error_msg:
            console.print(
                "\n[bold red]Authentication error.[/]\n"
                "1. Accept the model terms at: https://huggingface.co/google/gemma-4-31B-it\n"
                "2. Get your token at: https://huggingface.co/settings/tokens\n"
                "3. Run with: --token hf_your_token  or set HF_TOKEN env variable"
            )
        else:
            console.print(f"\n[bold red]Download failed:[/] {e}")
        raise typer.Exit(1)


def _check_token(token: str | None) -> None:
    if token:
        console.print("[green]✓ HuggingFace token found[/]")
    else:
        console.print(
            "[yellow]⚠  No HF_TOKEN found. Gated models (like Gemma) require authentication.[/]\n"
            "  Set HF_TOKEN env variable or pass --token hf_your_token\n"
            "  Get a token: https://huggingface.co/settings/tokens\n"
        )


def _hardware_check(repo_id: str, quantization: str) -> None:
    model_size_key = _guess_model_size(repo_id)
    if model_size_key is None:
        return

    reqs = VRAM_REQUIREMENTS.get(model_size_key, {})
    required_vram = reqs.get(quantization, reqs.get("4bit", "?"))

    table = Table(title="Hardware Requirements", show_header=True)
    table.add_column("Config", style="cyan")
    table.add_column("Est. VRAM needed", style="yellow")
    table.add_column("RTX 4070 (12GB)", style="green")

    for q, vram in reqs.items():
        fits = "✓ fits" if vram <= 12 else ("⚠ CPU offload" if vram <= 32 else "✗ too large")
        table.add_row(q, f"~{vram}GB", fits)

    console.print(table)

    if required_vram != "?" and required_vram > 12:
        console.print(
            f"\n[yellow]⚠  {repo_id} with {quantization} quantization needs ~{required_vram}GB VRAM.[/]\n"
            f"   Your RTX 4070 has 12GB — accelerate will offload the rest to CPU RAM.\n"
            f"   Generation will work but be slower (~2-5 tok/s).\n"
            f"   For faster inference, consider: google/gemma-4-4b-it or google/gemma-4-12b-it\n"
        )
    else:
        console.print(f"\n[green]✓ {repo_id} fits comfortably in your 12GB VRAM with {quantization}[/]\n")


def _guess_model_size(repo_id: str) -> str | None:
    lower = repo_id.lower()
    for size in ["31b", "27b", "12b", "4b", "1b"]:
        if size in lower:
            return size
    return None


def _check_disk_space(cache_dir: str, repo_id: str) -> None:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024**3)

    model_size_key = _guess_model_size(repo_id)
    # Rough download size estimate (safetensors at bfloat16, user applies quant at load time)
    estimated_gb = {"31b": 63, "27b": 55, "12b": 25, "4b": 9, "1b": 3}.get(
        model_size_key or "", 20
    )

    console.print(f"[dim]Disk free: {free_gb:.1f}GB | Estimated download: ~{estimated_gb}GB[/]")

    if free_gb < estimated_gb * 1.1:
        console.print(
            f"\n[bold red]⚠  Low disk space![/] "
            f"Need ~{estimated_gb}GB, have {free_gb:.1f}GB free on {path.anchor}"
        )
        if not typer.confirm("Continue anyway?"):
            raise typer.Exit(0)


if __name__ == "__main__":
    app()
