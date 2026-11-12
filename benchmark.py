"""Time a few training steps to pick a sensible thread count and epoch budget."""

from __future__ import annotations

import argparse
import time

import torch

from data import build_loaders
from models import CONFIGS, build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--name", default="vgg11")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    train_loader, _, _, _ = build_loaders("data", batch_size=args.batch, workers=2)
    model = build_model(**CONFIGS[args.name])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batches = []
    for i, (images, targets) in enumerate(train_loader):
        if i == 2:  # warm-up
            started = time.time()
        if i >= 2 + args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        batches.append(time.time())
        if len(batches) > 1:
            # per-step time is recorded only after warm-up to skip JIT/allocation noise
            pass
    elapsed = batches[-1] - batches[0]
    steps = len(batches) - 1
    per_step = elapsed / max(steps, 1)
    epoch = per_step * len(train_loader)
    print(f"threads={args.threads} batch={args.batch} step={per_step * 1000:.1f}ms "
          f"epoch={epoch:.1f}s steps={steps}")


if __name__ == "__main__":
    main()
