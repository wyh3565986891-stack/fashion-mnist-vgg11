# 训练一个配置，把报告要用的数字全部写进 runs/<name>/metrics.json。
# 用法：python train.py --name cbam --epochs 16

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from data import build_loaders
from engine import count_macs, count_parameters, evaluate, train
from models import CLASSES, CONFIGS, DESCRIPTIONS, build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--tag", default="", help="输出目录的后缀")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    folder = args.out / (args.name + (f"_{args.tag}" if args.tag else ""))
    folder.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, _ = build_loaders(
        args.data, batch_size=args.batch, workers=args.workers,
        seed=args.seed, validation_size=args.validation_size)
    model = build_model(**CONFIGS[args.name])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    started = time.time()
    history, best_epoch, best_val = train(
        model, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed,
        checkpoint=folder / "model.pt")
    elapsed = time.time() - started

    test = evaluate(model, test_loader, criterion)
    result = {
        "name": args.name,
        "tag": args.tag,
        "description": DESCRIPTIONS[args.name],
        "config": CONFIGS[args.name],
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "train_size": len(train_loader.dataset),
        "validation_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "classes": list(CLASSES),
        "parameters": count_parameters(model),
        "macs": count_macs(model),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "history": history,
        "test": test,
        "train_seconds": round(elapsed, 1),
        "threads": args.threads,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (folder / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.name}] test_acc={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f} "
          f"params={result['parameters']:,} macs={result['macs']:,} "
          f"best_epoch={best_epoch} time={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
