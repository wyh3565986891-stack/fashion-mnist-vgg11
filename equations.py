# 用 matplotlib 的 mathtext 把 LaTeX 公式渲染成图片，供 reportlab 拼版。
# 不依赖本机 LaTeX，字体用 Computer Modern，出来的效果接近论文。

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image

matplotlib.rcParams["mathtext.fontset"] = "cm"

CACHE = Path("figures/eq")
DPI = 600


def render(latex, fontsize=11.5):
    # 公式和字号做键，同一公式只渲染一次；返回 (png 路径, 宽 pt, 高 pt)
    CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(f"{latex}|{fontsize}".encode("utf-8")).hexdigest()[:12]
    path = CACHE / f"eq_{digest}.png"
    if not path.exists():
        math_to_image(f"${latex}$", str(path), prop=FontProperties(size=fontsize),
                      dpi=DPI, format="png")
    from PIL import Image
    with Image.open(path) as image:
        width, height = image.size
    return str(path), width / DPI * 72, height / DPI * 72


if __name__ == "__main__":
    samples = [
        r"\mathrm{MACs}=\sum_{l} H'_{l} W'_{l}\cdot \frac{C^{in}_{l}}{g_{l}}\cdot k_{l}^{2}\cdot C^{out}_{l}",
        r"F_{1}=\frac{2PR}{P+R}",
        r"r=\frac{k^{2}C_{in}C_{out}}{k^{2}C_{in}+C_{in}C_{out}}=\frac{1}{C_{out}}+\frac{1}{k^{2}}",
        r"\hat{x}=\frac{x-\mu_{g}}{\sqrt{\sigma_{g}^{2}+\epsilon}}",
        r"\mathrm{Mish}(x)=x\tanh\left(\ln(1+e^{x})\right)",
    ]
    for item in samples:
        print(render(item))
