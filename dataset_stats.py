# 数据集统计：类别分布、像素均值和标准差

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torchvision.datasets import FashionMNIST

import data as data_module
from models import CLASSES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/dataset.json"))
    args = parser.parse_args()

    train = FashionMNIST(args.data, train=True, download=False)
    test = FashionMNIST(args.data, train=False, download=False)
    train_counts = Counter(int(label) for label in train.targets)
    test_counts = Counter(int(label) for label in test.targets)

    import numpy as np
    pixels = torch.from_numpy(np.stack([np.asarray(img.convert("L")) for img, _ in train]))
    pixels = pixels.float() / 255.0
    mean, std = pixels.mean().item(), pixels.std().item()
    report_mean, report_std = data_module.MEAN, data_module.STD

    payload = {
        "train_size": len(train),
        "test_size": len(test),
        "image_shape": list(pixels.shape[1:]),
        "classes": list(CLASSES),
        "train_per_class": [train_counts[i] for i in range(10)],
        "test_per_class": [test_counts[i] for i in range(10)],
        "pixel_mean": mean,
        "pixel_std": std,
        "normalization_mean": report_mean,
        "normalization_std": report_std,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "train_per_class": payload["train_per_class"],
                      "test_per_class": payload["test_per_class"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
