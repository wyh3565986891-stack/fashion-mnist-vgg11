# Fashion-MNIST 上的窄通道 VGG11：一次只改一处，九个配置对照

这是课程大作业的代码。基线是一个按 28×28 输入缩小过的 VGG11（五层 stage，通道 16-32-64-96-96，约 39 万参数），
在 Fashion-MNIST 上训练到 0.9226。然后在这条基线上逐一替换卷积、注意力、激活函数、归一化层，
看每一项各自换来多少精度、多少计算量，最后把模型导成 ONNX、做个网页演示，看它在退化输入上稳不稳。

所有实验共用一套数据划分、随机种子和训练超参数，只有网络结构不同。报告里的数字都是脚本从 `runs/*.json` 读出来的，
不用手抄。

## 实验矩阵

| 编号 | 配置名 | 结构修改 |
| --- | --- | --- |
| C0 | `vgg11` | 基线：标准 3×3 卷积 + BatchNorm + ReLU |
| C1 | `sep` | Stage 3–5 的标准卷积换成深度可分离卷积 |
| C2 | `sep_all` | 五个 stage 全部换成深度可分离卷积 |
| C3 | `se` | Stage 3–5 末端插入 SE 通道注意力 |
| C4 | `cbam` | Stage 3–5 末端插入 CBAM（通道 + 空间） |
| C5 | `gelu` | 激活换成 GELU |
| C6 | `mish` | 激活换成 Mish |
| C7 | `gn` | BatchNorm 换成 GroupNorm（8 组） |
| C8 | `final` | 可分离卷积 + CBAM + GELU 的组合模型 |
| — | `vgg11 --tag bs16` | 小批量对照：batch=16、8 轮 |
| — | `gn --tag bs16` | 小批量对照：batch=16、8 轮 |

## 主要结果

官方 60000 张训练集按固定种子 42 切成 55000 / 5000（验证），官方 10000 张测试集只在最后评一次。
训练用 AdamW + 余弦退火，label smoothing 0.05，RandomCrop(28, padding=2) 加水平翻转，16 轮，batch 128。

| 配置 | 结构修改 | 参数量 | MACs | 测试准确率 | macro-F1 | 相对基线 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vgg11` | 标准卷积 + BN + ReLU | 386,618 | 5,156,736 | 0.9226 | 0.9224 | +0.00 pp |
| `sep` | Stage 3–5 可分离卷积 | 71,162 | 1,551,936 | 0.9048 | 0.9050 | −1.78 pp |
| `sep_all` | 五个 stage 全可分离 | 67,091 | 684,048 | 0.8836 | 0.8834 | −3.90 pp |
| `se` | SE 注意力 | 392,538 | 5,162,368 | 0.9190 | 0.9190 | −0.36 pp |
| `cbam` | CBAM 注意力 | 392,544 | 5,173,782 | 0.9193 | 0.9193 | −0.33 pp |
| `gelu` | GELU 激活 | 386,618 | 5,156,736 | 0.9246 | 0.9242 | +0.20 pp |
| `mish` | Mish 激活 | 386,618 | 5,156,736 | 0.9257 | 0.9254 | +0.31 pp |
| `gn` | GroupNorm(8) | 386,618 | 5,156,736 | 0.9151 | 0.9147 | −0.75 pp |
| `final` | 可分离 + CBAM + GELU | 77,088 | 1,568,982 | 0.9067 | 0.9063 | −1.59 pp |

小批量对照（batch=16，8 轮）：`vgg11` 0.9113、`gn` 0.9076。GN 和 BN 的差距从 0.75 pp 缩到 0.37 pp，但没有反超。

几点观察：

- 压缩放在哪里比用什么方法更重要。Stage 3 只占 14.4% 的参数，却占了 52.5% 的乘加；可分离卷积只铺在 Stage 3–5，参数压到 18.4%、乘加压到 30.1%，代价是 1.78 pp 精度。五个 stage 全换，只再多省 4,071 个参数和 867,888 次乘加，精度又掉 2.12 pp。
- 注意力在这个任务上没有帮上忙。SE 和 CBAM 各多 1.5% 的参数，准确率分别低 0.36 pp 和 0.33 pp；Grad-CAM 也没看出明显的定位改善，衬衫和 T 恤的互相误判从 173 例变成 193 例和 218 例。
- 换激活函数不用付参数代价。GELU 和 Mish 参数量、乘加都不变，分别高 0.20 pp 和 0.31 pp。不过这个差距比换随机种子的波动还小，只能当方向看。

## 部署侧结果

| 模型 | 参数量 | ONNX 体积 | int8 体积 | PyTorch 延迟 | ONNX 延迟 | int8 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `vgg11` | 386,618 | 1515 KB | 407 KB | 0.77 ms | 0.09 ms | 1.47 ms |
| `final` | 77,088 | 334 KB | 176 KB | 3.52 ms | 0.25 ms | 2.20 ms |

CPU 单线程、batch=1 实测（`export_onnx.py`）：

- ONNX Runtime 的 fp32 推理比 PyTorch eager 快 3–9 倍，主要是图优化把 conv 合并掉了。
- 动态 int8 量化在这个小通道模型上是负优化：体积缩到 27%–53%，但单线程延迟比 fp32 ONNX 慢 8.8–17.3 倍。onnxruntime 的 ConvInteger 对小通道卷积缺少优化 kernel，反量化和调度的开销盖过了整数运算的收益。
- 可分离卷积在 PyTorch eager 下反而更慢，depthwise 的 CPU 实现效率低。同一个模型换个运行时，结论就变了。

鲁棒性用 6 种合成退化在 10000 张测试集上测：平均掉点 `vgg11` 21.8 pp、`sep` 17.7 pp、`cbam` 18.6 pp、`final` 16.0 pp。
最脆弱的是高斯噪声（平均 −32.4 pp），最不敏感的是亮度缩放（−10.1 pp）。`final` 在 10×10 遮挡下只有 0.7186，四个模型里最低，说明结构改动之间不是简单叠加。

置信度分流（`final`，阈值 0.60）：8.3% 的样本低于阈值，这部分准确率只有 50.9%；其余 91.7% 自动通过，准确率 94.3%。
人工只看 8.3% 的样本，流水线的等效准确率就从 0.9067 抬到 0.9474。

## 目录

```
cnn_coursework/
├── models.py         # VGG11 主体、SE / CBAM / Mish / 可分离卷积、9 组配置
├── data.py           # 数据划分、增广、DataLoader、单图预处理
├── engine.py         # 训练循环、评估、混淆矩阵与 F1、MACs 统计
├── train.py          # 单配置训练入口 → runs/<name>/metrics.json
├── aggregate.py      # 汇总为 runs/summary.json / summary.csv
├── describe.py       # 逐 stage 参数量与 MACs → runs/architecture.json
├── dataset_stats.py  # 数据集统计 → runs/dataset.json
├── plots.py          # 训练曲线、准确率、精度-计算量、混淆矩阵、per-class F1、鲁棒性
├── gradcam.py        # Grad-CAM 可视化 → figures/gradcam.png
├── robustness.py     # 合成退化鲁棒性 → runs/robustness.json
├── confidence.py     # 置信度—覆盖率权衡 → runs/confidence.json
├── export_onnx.py    # ONNX 导出、int8 量化、单线程延迟 → runs/export.json
├── webapp.py         # 零依赖 Web 演示（上传 / 随机测试图 / Top-3 / 转人工）
├── predict.py        # 命令行单图预测
├── report.py         # 由 JSON + PNG 生成中文 PDF 报告
├── download.py       # 下载并校验 Fashion-MNIST
├── demo_check.py     # 验收前自检：数据、权重、图、报告、网页演示
└── requirements.txt
```

## 快速开始

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python download.py                          # 数据下到 data/，已有文件会校验 MD5
python train.py --name vgg11 --epochs 16
python predict.py sample.png --name vgg11 --topk 3
python demo_check.py                        # 检查数据、九组权重、图、报告和网页演示
```

`train.py` 会把权重、逐轮曲线、per-class 指标、混淆矩阵和参数量/乘加写进 `runs/<name>/metrics.json`。
同一个配置想多跑一组，用 `--tag` 区分目录。

### 复现全部实验

```bash
python dataset_stats.py
python describe.py
for /f %n in (vgg11 sep sep_all se cbam gelu mish gn final) do python train.py --name %n --epochs 16
python train.py --name vgg11 --epochs 8 --batch 16 --tag bs16
python train.py --name gn --epochs 8 --batch 16 --tag bs16
python aggregate.py
python plots.py
python gradcam.py
python robustness.py
python confidence.py
python export_onnx.py
python report.py --out ../output/CNN大作业/2427080125王炎煌.pdf
```

Linux/macOS 把 `for /f` 那行换成 `for n in vgg11 sep sep_all se cbam gelu mish gn final; do python train.py --name $n --epochs 16; done`。

6 线程 CPU 上单配置大约 6–8 分钟，九个串行一个多小时。三个进程并行跑的话单轮 22–31 秒。

### Web 演示

```bash
python webapp.py --name final --port 8000
# 打开 http://127.0.0.1:8000，拖图进去或点「随机测试图」
```

只用标准库的 `http.server`、PIL 和 PyTorch，没有 Flask/FastAPI。返回 Top-3 类目和概率，最高概率低于 0.60 时标「转人工复核」。
预处理会把「深色衣服 + 白底」的实拍图自动反色，换成训练时的「浅色衣服 + 黑底」。

## 生成的文件

| 文件 | 内容 |
| --- | --- |
| `runs/<name>/metrics.json` | 单配置的全部指标：history / test / per_class / confusion / parameters / macs |
| `runs/summary.json`、`summary.csv` | 11 组实验的横向对比 |
| `runs/architecture.json` | 逐 stage 的参数量、乘加、输出尺寸 |
| `runs/dataset.json` | 类别计数、像素均值/标准差 |
| `runs/robustness.json` | 4 个模型 × 7 种条件的准确率与损失 |
| `runs/confidence.json` | 5 个阈值下的覆盖率 / 准确率 / 转人工比例 |
| `runs/export.json` | ONNX / int8 体积与单线程延迟 |
| `runs/gradcam.json` | Grad-CAM 的预测与真实标签对照 |
| `figures/*.png` | 报告里用到的全部图 |

## 环境

- Windows 11（`Windows-11-10.0.26200-SP0`），Python 3.12.10
- PyTorch 2.15.0+cpu / torchvision 0.29.2+cpu
- NumPy 2.5.4 / Matplotlib 3.11.2 / Pillow 13.0.0
- onnx 1.23.2 / onnxruntime 1.31.1
- reportlab 5.1.0

全部在 CPU 上跑，不需要显卡。

## 报告

中文报告 16 页，13 张表、8 张图，由 `report.py` 从 JSON 和图生成：

```
../output/CNN大作业/2427080125王炎煌.pdf
```

改完实验按 `aggregate.py → plots.py → report.py` 重跑一遍，报告里的数会跟着变。

## 参考

1. Simonyan K, Zisserman A. *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR 2015. arXiv:1409.1556
2. Xiao H, Rasul K, Vollgraf R. *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747, 2017
3. Howard A G, et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv:1704.04861, 2017
4. Hu J, Shen L, Sun G. *Squeeze-and-Excitation Networks*. CVPR 2018
5. Woo S, et al. *CBAM: Convolutional Block Attention Module*. ECCV 2018
