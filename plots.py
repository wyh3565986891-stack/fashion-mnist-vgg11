# 报告里的图全部由 runs/*.json 生成，不手改

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160

COLORS = ["#2b6cb0", "#2f855a", "#b7791f", "#c53030", "#805ad5", "#dd6b20", "#319795", "#d53f8c"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fig_training_curves(summary, out: Path):
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.6), sharex=True, sharey=True)
    for axis in axes.flat:
        axis.set_visible(False)
    records = [record for record in summary if record.get("tag") != "bs16"]
    for axis, record in zip(axes.flat, records):
        axis.set_visible(True)
        history = record["history"]
        epochs = [row["epoch"] for row in history]
        axis.plot(epochs, [row["train_accuracy"] for row in history], color="#c53030",
                  marker="o", ms=3, lw=1.3, label="train")
        axis.plot(epochs, [row["val_accuracy"] for row in history], color="#2b6cb0",
                  marker="s", ms=3, lw=1.3, label="validation")
        axis.set_title(f'{record["name"]}  (test {record["test"]["accuracy"]:.3f})', fontsize=9)
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=7)
        best = record["best_epoch"]
        axis.axvline(best, color="#718096", ls="--", lw=0.8)
    axes.flat[0].legend(fontsize=7, loc="lower right")
    for axis in axes[-1]:
        axis.set_xlabel("epoch", fontsize=8)
    for axis in axes[:, 0]:
        axis.set_ylabel("accuracy", fontsize=8)
    fig.suptitle("训练/验证准确率曲线（虚线为选中轮次）", fontsize=11)
    fig.tight_layout()
    path = out / "training_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_accuracy(summary, out: Path):
    records = sorted(summary, key=lambda r: r["test"]["accuracy"])
    names = [r["name"] for r in records]
    values = [r["test"]["accuracy"] for r in records]
    colors = ["#2b6cb0" if r["name"] == "vgg11" else "#68a3d3" for r in records]
    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    bars = axis.barh(names, values, color=colors, height=0.62)
    for bar, record in zip(bars, records):
        axis.text(bar.get_width() + 0.0015, bar.get_y() + bar.get_height() / 2,
                  f'{record["test"]["accuracy"]:.4f}  ({record["delta_accuracy"]:+.4f})',
                  va="center", fontsize=7.5)
    axis.set_xlim(min(values) - 0.02, max(values) + 0.015)
    axis.set_xlabel("test accuracy")
    axis.set_title("测试集准确率（括号内为相对基线变化）", fontsize=10)
    axis.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = out / "accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_cost_accuracy(summary, out: Path):
    fig, axis = plt.subplots(figsize=(7.0, 4.3))
    for record in summary:
        color = "#2b6cb0" if record["name"] == "vgg11" else "#c53030"
        axis.scatter(record["macs"], record["test"]["accuracy"], s=48, color=color, zorder=3)
        axis.annotate(record["name"], (record["macs"], record["test"]["accuracy"]),
                      textcoords="offset points", xytext=(6, 4), fontsize=8)
    axis.set_xscale("log")
    axis.set_xlabel("MACs per image (log scale)")
    axis.set_ylabel("test accuracy")
    axis.set_title("精度—计算量权衡：左下角为更优", fontsize=10)
    axis.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out / "cost_accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_confusion(axis, record, title):
    matrix = np.array(record["test"]["confusion"], dtype=float)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    classes = record["classes"]
    axis.set_xticks(range(10))
    axis.set_xticklabels([c.split("/")[0] for c in classes], rotation=60, fontsize=6)
    axis.set_yticks(range(10))
    axis.set_yticklabels([c.split("/")[0] for c in classes], fontsize=6)
    for i in range(10):
        for j in range(10):
            if matrix[i, j] > 0.02:
                axis.text(j, i, f"{matrix[i, j] * 100:.0f}", ha="center", va="center",
                          fontsize=5.5, color="white" if matrix[i, j] > 0.55 else "#1a202c")
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("predicted", fontsize=8)
    axis.set_ylabel("true", fontsize=8)
    return image


def fig_confusion(summary, out: Path):
    records = {r["name"]: r for r in summary}
    picks = [name for name in ("vgg11", "sep", "cbam", "final") if name in records]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.0 * len(picks), 3.9))
    for axis, name in zip(np.atleast_1d(axes), picks):
        draw_confusion(axis, records[name], f'{name}  acc {records[name]["test"]["accuracy"]:.4f}')
    fig.tight_layout()
    path = out / "confusion.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_per_class(summary, out: Path):
    records = {r["name"]: r for r in summary}
    names = [n for n in ("vgg11", "sep", "cbam", "final") if n in records]
    classes = records[names[0]]["classes"]
    x = np.arange(len(classes))
    width = 0.8 / len(names)
    fig, axis = plt.subplots(figsize=(9.2, 4.2))
    for i, name in enumerate(names):
        f1 = records[name]["test"]["per_class"]["f1"]
        axis.bar(x + i * width, f1, width=width, label=name, color=COLORS[i])
    axis.set_xticks(x + width * (len(names) - 1) / 2)
    axis.set_xticklabels([c.replace("/", "/\n") for c in classes], rotation=30, ha="right", fontsize=7)
    axis.set_ylim(0.6, 1.0)
    axis.set_ylabel("per-class F1")
    axis.set_title("各类别 F1：细分类服饰仍是瓶颈", fontsize=10)
    axis.legend(fontsize=8, ncol=len(names))
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "per_class_f1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_robustness(path: Path, out: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    names = [row["corruption"] for row in data["rows"]]
    x = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(8.4, 3.8))
    width = 0.8 / len(data["models"])
    for i, model in enumerate(data["models"]):
        values = [row["accuracy"][model] for row in data["rows"]]
        axis.bar(x + i * width, values, width=width, label=model, color=COLORS[i])
    axis.set_xticks(x + width * (len(data["models"]) - 1) / 2)
    axis.set_xticklabels(names, rotation=20, ha="right", fontsize=7.5)
    axis.set_ylim(0, 1)
    axis.set_ylabel("accuracy")
    axis.set_title("合成退化下的准确率（鲁棒性）", fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    result = out / "robustness.png"
    fig.savefig(result, bbox_inches="tight")
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = load(args.runs / "summary.json")
    for record in summary:
        if record.get("tag") == "bs16":
            record["name"] = f'{record["name"]}_bs{record["batch"]}'
    made = [fig_training_curves(summary, args.out), fig_accuracy(summary, args.out),
            fig_cost_accuracy(summary, args.out), fig_confusion(summary, args.out),
            fig_per_class(summary, args.out)]
    if (args.runs / "robustness.json").exists():
        made.append(fig_robustness(args.runs / "robustness.json", args.out))
    for path in made:
        print(path)


if __name__ == "__main__":
    main()
