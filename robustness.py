# 合成退化下的鲁棒性对比

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from data import MEAN, STD, build_loaders
from engine import evaluate
from models import CONFIGS, build_model

MODELS = ("vgg11", "sep", "cbam", "final")


def to_pixel(images: torch.Tensor) -> torch.Tensor:
    return (images * STD + MEAN).clamp(0, 1)


def to_normalized(pixel: torch.Tensor) -> torch.Tensor:
    return (pixel - MEAN) / STD


def corrupt(name: str, images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    pixel = to_pixel(images)
    if name == "clean":
        pass
    elif name == "rotate+15":
        pixel = torch.stack([TF.rotate(image, 15) for image in pixel])
    elif name == "gaussian noise":
        pixel = (pixel + 0.15 * torch.randn(pixel.shape, generator=generator)).clamp(0, 1)
    elif name == "gaussian blur":
        pixel = torch.stack([TF.gaussian_blur(image, 3, 1.0) for image in pixel])
    elif name == "brightness x0.4":
        pixel = TF.adjust_brightness(pixel, 0.4)
    elif name == "occlusion 10x10":
        occluded = pixel.clone()
        rows = torch.randint(0, 18, (pixel.shape[0],), generator=generator)
        cols = torch.randint(0, 18, (pixel.shape[0],), generator=generator)
        for i in range(pixel.shape[0]):
            occluded[i, :, rows[i]:rows[i] + 10, cols[i]:cols[i] + 10] = 0.5
        pixel = occluded
    elif name == "downsample x2":
        small = F.interpolate(pixel, scale_factor=0.5, mode="bilinear", align_corners=False)
        pixel = F.interpolate(small, size=(28, 28), mode="bilinear", align_corners=False)
    else:
        raise ValueError(name)
    return to_normalized(pixel).clamp(-MEAN / STD, (1 - MEAN) / STD)


CONDITIONS = ("clean", "rotate+15", "gaussian noise", "gaussian blur",
              "brightness x0.4", "occlusion 10x10", "downsample x2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()

    _, _, test_loader, _ = build_loaders(args.data, batch_size=512, workers=2)
    images = torch.cat([batch for batch, _ in test_loader])
    targets = torch.cat([target for _, target in test_loader])
    print(f"test images: {tuple(images.shape)}")

    models = []
    for name in MODELS:
        model = build_model(**CONFIGS[name])
        model.load_state_dict(torch.load(args.runs / name / "model.pt", weights_only=True))
        models.append((name, model))
        print(f"{name}: {sum(p.numel() for p in model.parameters()):,} params")

    rows = []
    for condition in CONDITIONS:
        generator = torch.Generator().manual_seed(2026)
        corrupted = corrupt(condition, images, generator)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(corrupted, targets), batch_size=512)
        row = {"corruption": condition, "accuracy": {}, "loss": {}}
        for name, model in models:
            result = evaluate(model, loader, torch.nn.CrossEntropyLoss())
            row["accuracy"][name] = result["accuracy"]
            row["loss"][name] = result["loss"]
            print(f"  {condition:16s} {name:6s} acc={result['accuracy']:.4f}")
        rows.append(row)

    payload = {"models": list(MODELS), "rows": rows,
               "test_size": int(len(targets))}
    (args.runs / "robustness.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {args.runs / 'robustness.json'}")


if __name__ == "__main__":
    main()
