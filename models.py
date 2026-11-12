"""VGG11-style backbone for 28x28 grayscale images.

Every core CNN component is switchable so that the ablation study only changes
one thing at a time:

* ``conv``: ``standard`` (3x3 conv) or ``sep`` (3x3 depthwise + 1x1 pointwise)
* ``attention``: ``none``, ``se`` (Squeeze-Excitation) or ``cbam`` (channel + spatial)
* ``act``: ``relu``, ``gelu`` or ``mish``
* ``norm``: ``bn`` (BatchNorm) or ``gn`` (GroupNorm)

The layer count follows VGG11 (1-1-2-2-2 convolutions in five stages), but the
channel widths are reduced for 28x28 inputs, hence "narrow VGG11".
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

WIDTHS = (16, 32, 64, 96, 96)
DEPTH = (1, 1, 2, 2, 2)
CLASSES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")


class Mish(nn.Module):
    """x * tanh(softplus(x)); a smooth, non-monotonic-below-zero activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return Mish()
    raise ValueError(f"unknown activation: {name}")


class SqueezeExcite(nn.Module):
    """SE: global average pool -> bottleneck MLP -> per-channel gate."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class CBAM(nn.Module):
    """CBAM: channel attention (avg + max descriptors) followed by spatial attention."""

    def __init__(self, channels: int, reduction: int = 8, kernel: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = self.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1))
                               + self.mlp(F.adaptive_max_pool2d(x, 1)))
        x = x * channel
        pooled = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * self.sigmoid(self.spatial(pooled))


def make_norm(name: str, channels: int, groups: int) -> nn.Module:
    if name == "bn":
        return nn.BatchNorm2d(channels)
    if name == "gn":
        if channels % groups:
            raise ValueError(f"channels={channels} not divisible by groups={groups}")
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unknown norm: {name}")


def conv_block(in_ch: int, out_ch: int, conv: str, norm: str, groups: int,
               act: str) -> nn.Module:
    """One Conv-Norm-Act cell; ``sep`` expands into depthwise + pointwise."""
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
    """Narrow VGG11 with switchable convolution, attention, activation and norm."""

    def __init__(self, conv: str = "standard", attention: str = "none",
                 act: str = "relu", norm: str = "bn", groups: int = 8,
                 widths: tuple[int, ...] = WIDTHS, depth: tuple[int, ...] = DEPTH,
                 sep_from: int = 0, attn_from: int = 2, num_classes: int = 10,
                 dropout: float = 0.2, head: int = 128):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x))
        return self.classifier(x)

    @property
    def last_stage(self) -> nn.Module:
        """Feature map used for Grad-CAM."""
        return self.features[-1]


def build_model(conv="standard", attention="none", act="relu", norm="bn",
                groups=8, sep_from=0, **kwargs) -> VGG11:
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


CONFIGS = {
    # name: keyword arguments passed to build_model
    "vgg11":        dict(),
    "sep":          dict(conv="sep", sep_from=2),
    "sep_all":      dict(conv="sep", sep_from=0),
    "se":           dict(attention="se"),
    "cbam":         dict(attention="cbam"),
    "gelu":         dict(act="gelu"),
    "mish":         dict(act="mish"),
    "gn":           dict(norm="gn"),
    "final":        dict(conv="sep", sep_from=2, attention="cbam", act="gelu"),
}

DESCRIPTIONS = {
    "vgg11": "Baseline: standard conv, BN, ReLU",
    "sep": "Depthwise separable conv in stages 3-5",
    "sep_all": "Depthwise separable conv in all five stages",
    "se": "SE channel attention in stages 3-5",
    "cbam": "CBAM channel + spatial attention in stages 3-5",
    "gelu": "GELU instead of ReLU",
    "mish": "Mish instead of ReLU",
    "gn": "GroupNorm(8) instead of BatchNorm",
    "final": "Separable conv + CBAM + GELU",
}
