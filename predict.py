# 单张图的 top-k 预测

import argparse
from pathlib import Path

import torch
from PIL import Image

from data import preprocess_image
from models import CLASSES, CONFIGS, build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--name", default="final", choices=sorted(CONFIGS))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    model = build_model(**CONFIGS[args.name])
    model.load_state_dict(torch.load(args.runs / args.name / "model.pt", weights_only=True))
    model.eval()
    tensor = preprocess_image(Image.open(args.image))
    with torch.inference_mode():
        probabilities = model(tensor).softmax(1)[0]
        values, indices = probabilities.topk(args.topk)
    print(f"model: {args.name}  image: {args.image}")
    for value, index in zip(values, indices):
        print(f"  {CLASSES[int(index)]:<14s} {float(value):.4f}")


if __name__ == "__main__":
    main()
