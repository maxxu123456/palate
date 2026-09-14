"""Fits a LoRA regression head on the rating dataset. Needs the experiments extra and a GPU."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent
MISSING = "the finetune needs the experiments extra: uv sync --extra experiments"


@dataclass(frozen=True, slots=True)
class Config:
    """One run. Everything that changes between runs lives in the yaml, not in here."""

    base_model: str
    max_length: int
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]
    learning_rate: float
    batch_size: int
    epochs: int
    seed: int
    out_dir: Path


def read_config(path: Path) -> Config:
    """Load the run config, resolving out_dir beside this file."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    lora = raw["lora"]
    fit = raw["train"]
    return Config(
        base_model=raw["base_model"],
        max_length=int(raw["max_length"]),
        r=int(lora["r"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        learning_rate=float(fit["learning_rate"]),
        batch_size=int(fit["batch_size"]),
        epochs=int(fit["epochs"]),
        seed=int(fit["seed"]),
        out_dir=HERE / raw["out_dir"],
    )


def read_split(path: Path, split: str) -> list[tuple[str, float]]:
    """The jsonl dataset.py wrote, one side of the split."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [(r["text"], float(r["rating"])) for r in rows if r["split"] == split]


def batches(
    rows: list[tuple[str, float]], size: int, *, seed: int | None = None
) -> Iterator[tuple[list[str], list[float]]]:
    """Fixed size batches, shuffled when a seed is given."""
    order = list(rows)
    if seed is not None:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), size):
        chunk = order[start : start + size]
        yield [text for text, _ in chunk], [rating for _, rating in chunk]


def imports() -> tuple[Any, Any, Any, Any, Any]:
    """Pull the heavy libraries, or say which extra installs them."""
    try:
        import torch  # noqa: TID251
        from peft import LoraConfig, get_peft_model
        from transformers import (  # noqa: TID251
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise SystemExit(MISSING) from exc
    return torch, LoraConfig, get_peft_model, AutoModelForSequenceClassification, AutoTokenizer


def device_name(torch: Any) -> str:
    """The rented card first, then a laptop, then nothing."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def mae(
    torch: Any, model: Any, tokenizer: Any, config: Config, rows: list[tuple[str, float]]
) -> float:
    """Mean absolute error in stars, which is the number the mean baseline also reports."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for texts, targets in batches(rows, config.batch_size):
            batch = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length,
            ).to(model.device)
            said = model(**batch).logits.squeeze(-1)
            wanted = torch.tensor(targets, device=model.device, dtype=said.dtype)
            total += float(torch.abs(said - wanted).sum())
    return total / len(rows)


def run(config: Config, data: Path) -> float:
    """Fit the adapter, printing the validation error each epoch, and save it."""
    torch, LoraConfig, get_peft_model, AutoSeqCls, AutoTokenizer = imports()
    torch.manual_seed(config.seed)
    train_rows = read_split(data, "train")
    val_rows = read_split(data, "val")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoSeqCls.from_pretrained(config.base_model, num_labels=1, problem_type="regression")
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="SEQ_CLS",
            r=config.r,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=config.target_modules,
        ),
    )
    model.to(device_name(torch))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=config.learning_rate)
    score = float("inf")
    for epoch in range(config.epochs):
        model.train()
        for texts, targets in batches(train_rows, config.batch_size, seed=config.seed + epoch):
            batch = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length,
            ).to(model.device)
            said = model(**batch).logits.squeeze(-1)
            wanted = torch.tensor(targets, device=model.device, dtype=said.dtype)
            loss = torch.nn.functional.l1_loss(said, wanted)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
        score = mae(torch, model, tokenizer, config, val_rows)
        print(f"epoch {epoch + 1} val mae {score:.3f}")
    config.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.out_dir)
    return score


def main() -> None:
    """Train one adapter from a config and a built dataset."""
    parser = argparse.ArgumentParser(description="finetune the rating head")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--data", type=Path, default=HERE / "data" / "ratings.jsonl")
    args = parser.parse_args()
    config = read_config(args.config)
    print(f"val mae {run(config, args.data):.3f}, adapter in {config.out_dir}")


if __name__ == "__main__":
    main()
