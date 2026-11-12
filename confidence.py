# 置信度阈值和人工复核的覆盖率-精度权衡，输出 runs/confidence.json

import argparse
import json
from pathlib import Path

import torch

from data import build_loaders
from models import CONFIGS, build_model

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


@torch.inference_mode()
def collect(model, loader):
    model.eval()
    probabilities, labels = [], []
    for images, targets in loader:
        probabilities.append(model(images).softmax(1))
        labels.append(targets)
    return torch.cat(probabilities), torch.cat(labels)


def analyse(probabilities: torch.Tensor, labels: torch.Tensor) -> dict:
    confidence, prediction = probabilities.max(1)
    correct = prediction.eq(labels)
    rows = []
    for threshold in THRESHOLDS:
        covered = confidence >= threshold
        rejected = ~covered
        rows.append({
            "threshold": threshold,
            "coverage": round(float(covered.float().mean()), 4),
            "accuracy_covered": round(float(correct[covered].float().mean()), 4)
            if covered.any() else None,
            "accuracy_rejected": round(float(correct[rejected].float().mean()), 4)
            if rejected.any() else None,
            "review_share": round(float(rejected.float().mean()), 4),
        })
    return {"accuracy": round(float(correct.float().mean()), 4), "rows": rows,
            "confidence_mean": round(float(confidence.mean()), 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["final", "vgg11"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("runs/confidence.json"))
    args = parser.parse_args()

    _, _, test_loader, _ = build_loaders(args.data, batch_size=512, workers=0)
    records = []
    for name in args.models:
        model = build_model(**CONFIGS[name])
        model.load_state_dict(torch.load(args.runs / name / "model.pt", weights_only=True))
        probabilities, labels = collect(model, test_loader)
        record = {"name": name, **analyse(probabilities, labels)}
        records.append(record)
        print(name, record["accuracy"], [(r["threshold"], r["coverage"], r["accuracy_covered"])
                                         for r in record["rows"]])
    args.out.write_text(json.dumps({"thresholds": list(THRESHOLDS), "models": records},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
