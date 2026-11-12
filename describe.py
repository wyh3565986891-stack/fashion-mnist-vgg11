"""Per-stage cost table (parameters / MACs) for the report, straight from the code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from models import CONFIGS, build_model

STAGE_NAMES = ["stage1", "stage2", "stage3", "stage4", "stage5"]


def describe(name: str) -> dict:
    model = build_model(**CONFIGS[name])
    rows: dict[str, dict] = {}
    hooks = []

    shapes: dict[str, tuple] = {}

    def make_shape_hook(module, key):
        def hook(mod, inputs, output):
            channels, height, width = output.shape[1:]
            if key in shapes:
                old_channels, old_h, old_w = shapes[key]
                if mod.kernel_size == (3, 3):
                    height, width = old_h, old_w
                channels = max(channels, old_channels)
            shapes[key] = (channels, height, width)
        return hook

    for index, stage in enumerate(model.features):
        key = STAGE_NAMES[index]
        rows[key] = {"params": sum(p.numel() for p in stage.parameters()), "macs": 0,
                     "modules": [type(m).__name__ for m in stage.modules()
                                 if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.GroupNorm,
                                                   nn.MaxPool2d, nn.ReLU, nn.GELU))]}
        for module in stage.modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(make_shape_hook(module, key)))

    rows["classifier"] = {"params": sum(p.numel() for p in model.classifier.parameters()),
                          "macs": 0, "modules": ["Linear", "Dropout"]}

    def make_cost_hook(module, key):
        def hook(mod, inputs, output):
            if isinstance(mod, nn.Conv2d):
                rows[key]["macs"] += output.numel() * (mod.in_channels // mod.groups) \
                    * mod.kernel_size[0] * mod.kernel_size[1]
            else:
                rows[key]["macs"] += mod.in_features * mod.out_features
        return hook

    for module_name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if module_name.startswith("features."):
            key = STAGE_NAMES[int(module_name.split(".")[1])]
        else:
            key = "classifier"
        hooks.append(module.register_forward_hook(make_cost_hook(module, key)))

    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, 1, 28, 28))
    for handle in hooks:
        handle.remove()

    table = []
    for key, row in rows.items():
        table.append({"stage": key,
                      "shape": list(shapes.get(key, ())) or None,
                      "modules": sorted(set(row["modules"])),
                      "params": row["params"], "macs": row["macs"]})
    total_params = sum(r["params"] for r in table)
    total_macs = sum(r["macs"] for r in table)
    for row in table:
        row["params_share"] = row["params"] / total_params
        row["macs_share"] = row["macs"] / total_macs
    return {"name": name, "rows": table, "params": total_params, "macs": total_macs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", default=["vgg11", "final"])
    parser.add_argument("--out", type=Path, default=Path("runs/architecture.json"))
    args = parser.parse_args()
    payload = {name: describe(name) for name in args.names}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, data in payload.items():
        print(f"\n## {name}  params={data['params']:,}  macs={data['macs']:,}")
        for row in data["rows"]:
            shape = "x".join(str(v) for v in row["shape"]) if row["shape"] else "-"
            print(f"| {row['stage']:10s} | {shape:12s} | {row['params']:>9,d} "
                  f"({row['params_share']:5.1%}) | {row['macs']:>10,d} ({row['macs_share']:5.1%}) |")


if __name__ == "__main__":
    main()
