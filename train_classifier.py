"""Train the Vision Transformer Detexify command classifier."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from classifier import (
    CommandVocabulary,
    DetexifyCommandDataset,
    LatexCommandClassifier,
    ModelConfig,
    save_classifier_checkpoint,
)
from detexify_data import load_detexify


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    top1_accuracy: float
    top5_accuracy: float


def stratified_split_indices(
    targets: np.ndarray,
    *,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices while retaining every class in the training partition."""

    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional array")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    generator = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for class_index in np.unique(targets):
        indices = np.flatnonzero(targets == class_index)
        generator.shuffle(indices)
        if len(indices) == 1:
            validation_count = 0
        else:
            validation_count = min(
                len(indices) - 1,
                max(1, round(len(indices) * validation_fraction)),
            )
        validation_parts.append(indices[:validation_count])
        train_parts.append(indices[validation_count:])
    train_indices = np.concatenate(train_parts).astype(np.int64, copy=False)
    validation_indices = np.concatenate(validation_parts).astype(np.int64, copy=False)
    generator.shuffle(train_indices)
    generator.shuffle(validation_indices)
    return train_indices, validation_indices


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _accumulate_metrics(
    logits: Tensor,
    targets: Tensor,
) -> tuple[int, int]:
    top_k = min(5, logits.shape[1])
    predictions = logits.topk(top_k, dim=1).indices
    top1_correct = int((predictions[:, 0] == targets).sum().item())
    top5_correct = int((predictions == targets.unsqueeze(1)).any(dim=1).sum().item())
    return top1_correct, top5_correct


def train_one_epoch(
    model: LatexCommandClassifier,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> EpochMetrics:
    model.train()
    loss_sum = 0.0
    top1_correct = 0
    top5_correct = 0
    sample_count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = targets.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        batch_top1, batch_top5 = _accumulate_metrics(logits.detach(), targets)
        top1_correct += batch_top1
        top5_correct += batch_top5
        sample_count += batch_size
    return EpochMetrics(
        loss=loss_sum / sample_count,
        top1_accuracy=top1_correct / sample_count,
        top5_accuracy=top5_correct / sample_count,
    )


@torch.inference_mode()
def evaluate(
    model: LatexCommandClassifier,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    loss_sum = 0.0
    top1_correct = 0
    top5_correct = 0
    sample_count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_function(logits, targets)
        batch_size = targets.shape[0]
        loss_sum += float(loss) * batch_size
        batch_top1, batch_top5 = _accumulate_metrics(logits, targets)
        top1_correct += batch_top1
        top5_correct += batch_top5
        sample_count += batch_size
    return EpochMetrics(
        loss=loss_sum / sample_count,
        top1_accuracy=top1_correct / sample_count,
        top5_accuracy=top5_correct / sample_count,
    )


def _class_weights(targets: np.ndarray, num_classes: int) -> Tensor:
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("Every output class must occur in the training split")
    weights = counts.sum() / (num_classes * counts)
    return torch.from_numpy(weights)


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    frame = load_detexify(args.data_dir)
    if args.max_samples is not None and args.max_samples < frame.height:
        frame = frame.sample(n=args.max_samples, seed=args.seed, shuffle=True)
    vocabulary = CommandVocabulary.from_frame(frame)
    dataset = DetexifyCommandDataset(frame, vocabulary, image_size=32)
    train_indices, validation_indices = stratified_split_indices(
        dataset.targets,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if validation_indices.size == 0:
        raise ValueError("The selected data has no validation examples")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    data_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        shuffle=True,
        generator=data_generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices.tolist()),
        shuffle=False,
        **loader_options,
    )

    config = ModelConfig(
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        attention_heads=args.attention_heads,
        transformer_layers=args.transformer_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    )
    model = LatexCommandClassifier(len(vocabulary), config).to(device)
    weights = _class_weights(dataset.targets[train_indices], len(vocabulary)).to(device)
    loss_function = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(
        json.dumps(
            {
                "device": str(device),
                "classes": len(vocabulary),
                "train_samples": len(train_indices),
                "validation_samples": len(validation_indices),
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "config": asdict(config),
            }
        ),
        flush=True,
    )
    best_accuracy = -1.0
    output_path = Path(args.output)
    for epoch in range(1, args.epochs + 1):
        training_metrics = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )
        scheduler.step()
        report = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train": asdict(training_metrics),
            "validation": asdict(validation_metrics),
        }
        print(json.dumps(report), flush=True)
        if validation_metrics.top1_accuracy > best_accuracy:
            best_accuracy = validation_metrics.top1_accuracy
            save_classifier_checkpoint(
                output_path,
                model,
                vocabulary,
                epoch=epoch,
                metrics={
                    "validation_loss": validation_metrics.loss,
                    "validation_top1_accuracy": validation_metrics.top1_accuracy,
                    "validation_top5_accuracy": validation_metrics.top5_accuracy,
                },
            )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/detexify"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/detexify-vit.pt"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error(
            "epochs and batch-size must be positive; workers cannot be negative"
        )
    return args


if __name__ == "__main__":
    train(parse_args())
