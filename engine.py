"""Training / evaluation helpers shared by every experiment."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# cost accounting
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def count_macs(model: nn.Module, size: int = 28) -> int:
    """Multiply-accumulate operations of one forward pass (conv + linear)."""
    total = 0
    handles = []

    def conv_hook(module: nn.Conv2d, inputs, output):
        nonlocal total
        kernel = module.kernel_size[0] * module.kernel_size[1]
        total += output.numel() * (module.in_channels // module.groups) * kernel

    def linear_hook(module: nn.Linear, inputs, output):
        nonlocal total
        total += module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    model.eval()
    model(torch.zeros(1, 1, size, size))
    for handle in handles:
        handle.remove()
    return total


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
@dataclass
class Metrics:
    loss: float = 0.0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    per_class: dict = field(default_factory=dict)
    confusion: list = field(default_factory=list)
    n: int = 0


def metrics_from_confusion(confusion: torch.Tensor) -> dict:
    """Derive accuracy / macro-F1 / per-class scores from a 10x10 count matrix."""
    confusion = confusion.double()
    total = confusion.sum().item()
    tp = confusion.diag()
    fp = confusion.sum(0) - tp
    fn = confusion.sum(1) - tp
    precision = tp / (tp + fp).clamp_min(1e-9)
    recall = tp / (tp + fn).clamp_min(1e-9)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-9)
    return {
        "accuracy": tp.sum().item() / max(total, 1.0),
        "macro_f1": f1.mean().item(),
        "per_class": {"precision": precision.tolist(), "recall": recall.tolist(),
                      "f1": f1.tolist(), "support": confusion.sum(1).tolist()},
        "confusion": confusion.tolist(),
        "n": int(total),
    }


@torch.inference_mode()
def evaluate(model: nn.Module, loader, criterion: nn.Module | None = None) -> dict:
    model.eval()
    confusion = torch.zeros(10, 10, dtype=torch.long)
    loss_sum = 0.0
    count = 0
    for images, targets in loader:
        logits = model(images)
        if criterion is not None:
            loss_sum += criterion(logits, targets).item() * len(targets)
        predicted = logits.argmax(1)
        confusion += torch.bincount(10 * targets + predicted, minlength=100).reshape(10, 10)
        count += len(targets)
    out = metrics_from_confusion(confusion)
    out["loss"] = loss_sum / max(count, 1)
    return out


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(model: nn.Module, train_loader, val_loader, *, epochs: int, lr: float,
          weight_decay: float, seed: int, checkpoint: Path,
          eval_train: bool = False):
    """Cosine-annealed AdamW training; keeps the best validation checkpoint."""
    torch.manual_seed(seed)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    best = -1.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        seen = 0
        correct = 0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(targets)
            seen += len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
        lr_used = optimizer.param_groups[0]["lr"]
        scheduler.step()
        val = evaluate(model, val_loader, criterion)
        row = {
            "epoch": epoch,
            "lr": lr_used,
            "train_loss": running / max(seen, 1),
            "train_accuracy": correct / max(seen, 1),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "seconds": round(time.time() - started, 2),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val["accuracy"] > best:
            best = val["accuracy"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    return history, best_epoch, best
