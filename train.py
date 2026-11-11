import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor


def make_model(separable=False):
    layers = []
    channels = 1
    for stage, (width, repeats) in enumerate([(16, 1), (32, 1), (64, 2), (96, 2), (96, 2)]):
        for _ in range(repeats):
            if separable and stage >= 2:
                layers += [nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                           nn.Conv2d(channels, width, 1, bias=False)]
            else:
                layers.append(nn.Conv2d(channels, width, 3, padding=1, bias=False))
            layers += [nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
            channels = width
        if stage < 4:
            layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(channels, 128),
                         nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    confusion = torch.zeros(10, 10, dtype=torch.long)
    for images, targets in loader:
        predicted = model(images).argmax(dim=1)
        confusion += torch.bincount(10 * targets + predicted, minlength=100).reshape(10, 10)
    correct = confusion.diag()
    totals = confusion.sum(0) + confusion.sum(1)
    return {"accuracy": correct.sum().item() / confusion.sum().item(),
            "macro_f1": (2 * correct / totals.clamp_min(1)).mean().item(),
            "confusion": confusion.tolist()}


@torch.inference_mode()
def macs(model):
    image = torch.zeros(1, 1, 28, 28)
    total = 0
    model.eval()
    for layer in model:
        image = layer(image)
        if isinstance(layer, nn.Conv2d):
            total += image.numel() * (layer.in_channels // layer.groups) * 9 if layer.kernel_size == (3, 3) else image.numel() * layer.in_channels
        elif isinstance(layer, nn.Linear):
            total += layer.in_features * layer.out_features
    return total


def main(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    transform = Compose([ToTensor(), Normalize((0.286,), (0.353,))])
    train = FashionMNIST(args.data, train=True, transform=transform)
    test = FashionMNIST(args.data, train=False, transform=transform)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(42))
    validation = Subset(train, order[:2000].tolist())
    subset = Subset(train, order[2000:2000 + args.samples].tolist())
    val_loader = DataLoader(validation, batch_size=256)
    test_loader = DataLoader(test, batch_size=256)
    result = {"seed": 42, "epochs": args.epochs, "batch": 128, "lr": 0.001,
              "train": len(subset), "validation": len(validation), "test": len(test),
              "classes": test.classes, "models": {}}
    for name in ("standard", "separable"):
        torch.manual_seed(42)
        loader = DataLoader(subset, batch_size=128, shuffle=True,
                            generator=torch.Generator().manual_seed(42))
        model = make_model(name == "separable")
        optimizer = torch.optim.AdamW(model.parameters(), lr=result["lr"])
        loss_fn = nn.CrossEntropyLoss()
        best = -1
        history = []
        checkpoint = args.output / (name + ".pt")
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0
            for images, targets in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(images), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(images)
            accuracy = evaluate(model, val_loader)["accuracy"]
            row = {"epoch": epoch, "loss": total_loss / len(subset), "validation_accuracy": accuracy}
            history.append(row)
            print(name, row, flush=True)
            if accuracy > best:
                best = accuracy
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        result["models"][name] = {"parameters": sum(x.numel() for x in model.parameters()),
                                  "macs": macs(model), "best_epoch": best_epoch,
                                  "history": history, "test": evaluate(model, test_loader)}
        (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    main(parser.parse_args())
