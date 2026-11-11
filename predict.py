import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from train import make_model
from torchvision.transforms import Compose, Normalize, ToTensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("runs/standard.pt"))
    parser.add_argument("--separable", action="store_true")
    args = parser.parse_args()
    model = make_model(args.separable)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()
    with Image.open(args.image) as source:
        image = ImageOps.pad(ImageOps.exif_transpose(source).convert("L"), (28, 28), color=0)
    tensor = Compose([ToTensor(), Normalize((0.286,), (0.353,))])(image).unsqueeze(0)
    classes = json.loads(args.weights.with_name("results.json").read_text(encoding="utf-8"))["classes"]
    with torch.inference_mode():
        scores, indices = model(tensor).softmax(1)[0].topk(3)
    for score, index in zip(scores, indices):
        print(f"{classes[index]}: {score:.4f}")
