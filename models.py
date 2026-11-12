# 窄通道 VGG11：层数照 VGG11 的 1-1-2-2-2 排，通道宽度按 28x28 缩小。
# 卷积、注意力、激活、归一化都可以换，消融实验每次只动其中一项。

import torch
from torch import nn
from torch.nn import functional as F

WIDTHS = (16, 32, 64, 96, 96)
DEPTH = (1, 1, 2, 2, 2)
CLASSES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")


class Mish(nn.Module):
    # x * tanh(softplus(x))
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def make_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return Mish()
    raise ValueError(f"unknown activation: {name}")


class SqueezeExcite(nn.Module):
    # 全局平均池化 -> 瓶颈 MLP -> 逐通道权重
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class CBAM(nn.Module):
    # 通道注意力（平均 + 最大描述子）之后接空间注意力
    def __init__(self, channels, reduction=8, kernel=7):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel = self.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1))
                               + self.mlp(F.adaptive_max_pool2d(x, 1)))
        x = x * channel
        pooled = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * self.sigmoid(self.spatial(pooled))


def make_norm(name, channels, groups):
    if name == "bn":
        return nn.BatchNorm2d(channels)
    if name == "gn":
        if channels % groups:
            raise ValueError(f"channels={channels} not divisible by groups={groups}")
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unknown norm: {name}")


def conv_block(in_ch, out_ch, conv, norm, groups, act):
    # 一个 Conv-Norm-Act 单元；sep 展开成 depthwise + pointwise
    if conv == "standard":
        body = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
    elif conv == "sep":
        body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
        )
    else:
        raise ValueError(f"unknown conv: {conv}")
    return nn.Sequential(body, make_norm(norm, out_ch, groups), make_activation(act))


class VGG11(nn.Module):
    def __init__(self, conv="standard", attention="none", act="relu", norm="bn",
                 groups=8, widths=WIDTHS, depth=DEPTH, sep_from=0, attn_from=2,
                 num_classes=10, dropout=0.2, head=128):
        super().__init__()
        stages = []
        in_ch = 1
        for s, (width, repeats) in enumerate(zip(widths, depth)):
            layers = []
            for _ in range(repeats):
                stage_conv = "sep" if (conv == "sep" and s >= sep_from) else "standard"
                layers.append(conv_block(in_ch, width, stage_conv, norm, groups, act))
                in_ch = width
            if attention != "none" and s >= attn_from:
                layers.append(SqueezeExcite(in_ch) if attention == "se" else CBAM(in_ch))
            if s < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            stages.append(nn.Sequential(*layers))
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, head),
            make_activation(act),
            nn.Dropout(dropout),
            nn.Linear(head, head // 2),
            make_activation(act),
            nn.Linear(head // 2, num_classes),
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x)

    @property
    def last_stage(self):
        return self.features[-1]


def build_model(conv="standard", attention="none", act="relu", norm="bn",
                groups=8, sep_from=0, **kwargs):
    model = VGG11(conv=conv, attention=attention, act=act, norm=norm,
                  groups=groups, sep_from=sep_from, **kwargs)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.zeros_(module.bias)
    return model


# 九个配置的差别只在这几个关键字参数上
CONFIGS = {
    "vgg11":   dict(),
    "sep":     dict(conv="sep", sep_from=2),
    "sep_all": dict(conv="sep", sep_from=0),
    "se":      dict(attention="se"),
    "cbam":    dict(attention="cbam"),
    "gelu":    dict(act="gelu"),
    "mish":    dict(act="mish"),
    "gn":      dict(norm="gn"),
    "final":   dict(conv="sep", sep_from=2, attention="cbam", act="gelu"),
}

DESCRIPTIONS = {
    "vgg11": "基线：标准卷积 + BN + ReLU",
    "sep": "Stage 3-5 换成深度可分离卷积",
    "sep_all": "五个 stage 全部换成深度可分离卷积",
    "se": "Stage 3-5 加 SE 通道注意力",
    "cbam": "Stage 3-5 加 CBAM 通道 + 空间注意力",
    "gelu": "ReLU 换 GELU",
    "mish": "ReLU 换 Mish",
    "gn": "BatchNorm 换 GroupNorm(8)",
    "final": "可分离卷积 + CBAM + GELU",
}
