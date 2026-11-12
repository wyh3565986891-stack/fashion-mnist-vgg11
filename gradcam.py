# Grad-CAM：baseline 和组合模型分别看图片的哪里

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from data import build_loaders
from models import CONFIGS, CLASSES, build_model

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class GradCAM:
    # 取某个 stage 的特征图和梯度，加权求和得到热力图

    def __init__(self, model: torch.nn.Module, layer: int = 3):
        self.model = model
        self.activations = None
        self.gradients = None
        self.layer = model.features[layer - 1]
        self.handle = self.layer.register_forward_hook(self._forward)
        self.handle_back = self.layer.register_full_backward_hook(self._backward)

    def _forward(self, module, inputs, output):
        self.activations = output.detach()

    def _backward(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, target: int | None = None):
        self.model.eval()
        logits = self.model(image)
        if target is None:
            target = int(logits.argmax(1))
        self.model.zero_grad(set_to_none=True)
        logits[0, target].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam, target, int(logits.argmax(1))

    def close(self):
        self.handle.remove()
        self.handle_back.remove()


MEAN, STD = 0.286, 0.353


def denormalize(image: torch.Tensor) -> np.ndarray:
    array = image * STD + MEAN
    return array[0, 0].numpy().clip(0, 1)


def overlay(image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    colored = plt.get_cmap("jet")(cam)[..., :3]
    return (0.45 * np.stack([image] * 3, -1) + 0.55 * colored).clip(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 3, 8, 12, 20])
    parser.add_argument("--layer", type=int, default=3, help="hooked stage, 1-based")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    _, _, test_loader, test_set = build_loaders("data", batch_size=512, workers=2)
    picks = [test_set[i] for i in args.indices]
    images = torch.stack([p[0] for p in picks])
    labels = [int(p[1]) for p in picks]

    models = []
    for name in ("vgg11", "final"):
        model = build_model(**CONFIGS[name])
        model.load_state_dict(torch.load(args.runs / name / "model.pt", weights_only=True))
        models.append((name, model))

    columns = len(args.indices)
    fig, axes = plt.subplots(2, columns, figsize=(2.0 * columns, 4.4))
    summary = []
    for row, (name, model) in enumerate(models):
        cam_tool = GradCAM(model, args.layer)
        for column, index in enumerate(args.indices):
            cam, target, prediction = cam_tool(images[column:column + 1])
            image = denormalize(images[column:column + 1])
            axis = axes[row, column]
            axis.imshow(overlay(image, cam))
            axis.axis("off")
            truth = CLASSES[labels[column]].split("/")[0]
            predicted = CLASSES[prediction].split("/")[0]
            mark = "对" if prediction == labels[column] else "错"
            axis.set_title(f"{mark} {predicted}\n(true {truth})", fontsize=6.5)
            summary.append({"model": name, "index": index, "target": target,
                            "prediction": prediction, "label": labels[column]})
        cam_tool.close()
        axes[row, 0].text(-0.12, 0.5, name, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="center", fontsize=9)
    fig.suptitle(f"Grad-CAM（Stage {args.layer}）：模型关注的区域（上：baseline，下：separable+CBAM+GELU）",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "gradcam.png", bbox_inches="tight", dpi=170)
    plt.close(fig)
    (args.runs / "gradcam.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out / "gradcam.png")


if __name__ == "__main__":
    main()
