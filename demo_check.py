# 验收前自检：数据、九组权重、汇总 JSON、图、报告 PDF、网页演示。
# 用法：python demo_check.py

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIGS = ["vgg11", "sep", "sep_all", "se", "cbam", "gelu", "mish", "gn", "final"]
FIGURES = ["accuracy", "confusion", "cost_accuracy", "gradcam", "per_class_f1",
           "robustness", "training_curves", "webapp"]
JSONS = ["summary", "architecture", "dataset", "robustness", "confidence", "export", "gradcam"]
REPORT = ROOT.parent / "output" / "CNN大作业" / "2427080125王炎煌.pdf"

ok = []
bad = []


def check(name, condition, detail=""):
    (ok if condition else bad).append(f"{name}{('：' + detail) if detail else ''}")


check("Python", sys.version_info >= (3, 10), sys.version.split()[0])
try:
    import torch
    check("PyTorch", True, torch.__version__)
except Exception as exc:
    check("PyTorch", False, str(exc))

raw = ROOT / "data" / "FashionMNIST" / "raw"
files = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
         "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
missing = [f for f in files if not any(raw.glob(f + "*"))] if raw.is_dir() else files
check("Fashion-MNIST 数据", not missing, f"缺 {missing}" if missing else "4 个 IDX 文件")

for name in CONFIGS:
    folder = ROOT / "runs" / name
    check(f"权重 {name}", (folder / "model.pt").exists())
    metrics = folder / "metrics.json"
    acc = ""
    if metrics.exists():
        acc = f"acc {json.loads(metrics.read_text(encoding='utf-8'))['test']['accuracy']:.4f}"
    check(f"指标 {name}", metrics.exists(), acc)

for name in JSONS:
    check(f"汇总 {name}.json", (ROOT / "runs" / f"{name}.json").exists())
for name in FIGURES:
    check(f"图 {name}.png", (ROOT / "figures" / f"{name}.png").exists())
check("报告 PDF", REPORT.exists(), f"{REPORT.stat().st_size // 1024} KB" if REPORT.exists() else str(REPORT))

# 网页演示：起在 8011，请求一次首页再关掉
port = 8011
with socket.socket() as sock:
    busy = sock.connect_ex(("127.0.0.1", port)) == 0
if busy:
    check("网页演示", False, f"端口 {port} 已被占用")
else:
    process = subprocess.Popen(
        [sys.executable, "webapp.py", "--name", "final", "--port", str(port)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    served = False
    for _ in range(40):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                body = response.read().decode("utf-8", "ignore")
                served = response.status == 200 and "服饰图像分类" in body
            if served:
                break
        except Exception:
            continue
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    check("网页演示", served, f"端口 {port}")

print(f"通过 {len(ok)} 项，失败 {len(bad)} 项")
for line in bad:
    print("  ×", line)
if not bad:
    print("全部就绪，可以开始验收。")
