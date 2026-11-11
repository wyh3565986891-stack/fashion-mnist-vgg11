# VGG11 风格的服饰分类

普通卷积和深度可分离卷积使用同一组 Fashion-MNIST 数据。训练脚本保存每轮验证准确率、选中的权重、测试集混淆矩阵和参数统计。

在项目目录下使用 Python 3.12 运行。

```powershell
python -m pip install -r requirements.txt
python download.py
python train.py
python predict.py demo.png
```

`train.py` 默认使用 10000 张训练图片和 2000 张验证图片。训练与验证从官方训练集固定随机划分，测试用官方测试集。新实验可加 `--output runs_recheck`。逐轮记录和最终指标在 `runs/results.json`。

`predict.py` 接收单件服饰图片，输出前三类和分数。演示图采用与训练图相同的亮前景、黑背景格式。用轻量版时传入 `--weights runs/separable.pt --separable`。

数据来自 [Zalando Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)，模型参考 [VGG](https://arxiv.org/abs/1409.1556)。
