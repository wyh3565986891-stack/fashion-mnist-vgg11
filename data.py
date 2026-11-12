# Fashion-MNIST 数据：固定划分、增广、DataLoader。

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, ToTensor)

MEAN, STD = 0.286, 0.353


def make_transform(train):
    steps = []
    if train:
        steps += [RandomCrop(28, padding=2), RandomHorizontalFlip(0.5)]
    steps += [ToTensor(), Normalize((MEAN,), (STD,))]
    return Compose(steps)


def preprocess_image(image):
    # 任意尺寸的照片 -> 1x1x28x28 张量
    from PIL import Image
    from torchvision.transforms import functional as TF

    image = image.convert("L")
    scale = 28 / min(image.size)
    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                         Image.BILINEAR)
    left = (image.width - 28) // 2
    top = (image.height - 28) // 2
    image = image.crop((left, top, left + 28, top + 28))
    tensor = TF.to_tensor(image)
    # 数据集是黑底浅色衣服，实拍图往往是白底深色衣服，平均亮度偏高就反色
    if tensor.mean() > 0.5:
        tensor = 1 - tensor
    return TF.normalize(tensor, (MEAN,), (STD,)).unsqueeze(0)


def build_datasets(root, validation_size=5000, seed=42):
    # 官方 60k 训练集切出 5k 做验证，测试集原样保留
    train_full = FashionMNIST(root, train=True, transform=make_transform(True), download=False)
    val_full = FashionMNIST(root, train=True, transform=make_transform(False), download=False)
    test = FashionMNIST(root, train=False, transform=make_transform(False), download=False)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_full), generator=generator).tolist()
    train = Subset(train_full, order[validation_size:])
    validation = Subset(val_full, order[:validation_size])
    return train, validation, test


def build_loaders(root, batch_size=128, workers=4, seed=42, validation_size=5000):
    train, validation, test = build_datasets(root, validation_size, seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=workers, persistent_workers=workers > 0,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(validation, batch_size=512, num_workers=workers,
                            persistent_workers=workers > 0)
    test_loader = DataLoader(test, batch_size=512, num_workers=workers,
                             persistent_workers=workers > 0)
    return train_loader, val_loader, test_loader, test
