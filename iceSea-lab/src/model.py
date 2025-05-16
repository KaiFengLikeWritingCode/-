"""
Model factory for sea-ice semantic segmentation.

新增:
    • SMPWrapper — 把 segmentation_models_pytorch 的输出包装成
      {'out': logits}，对齐 TorchVision segmentation API。
"""
from __future__ import annotations

import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision.models.segmentation import deeplabv3_resnet50


class SMPWrapper(nn.Module):
    """统一 segmentation_models_pytorch 的输出接口为 dict."""
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return {"out": self.net(x)}


def get_model(
    name: str,
    num_classes: int,
    backbone: str = "resnet34",
    pretrained_backbone: bool = False,
):
    """
    Args
    ----
    name:
        "deeplabv3+" | "unet"
    num_classes:
        输出类别 (logits 通道数)
    backbone:
        仅对 U-Net 生效；指定 encoder_name
    pretrained_backbone:
        是否加载 ImageNet 预训练权重
    """
    name = name.lower()

    if name == "deeplabv3+":
        model = deeplabv3_resnet50(
            weights=("IMAGENET1K_V2" if pretrained_backbone else None),
            aux_loss=None,
            num_classes=num_classes,
        )
        return model

    elif name == "unet":
        net = smp.Unet(
            encoder_name=backbone,
            encoder_weights=("imagenet" if pretrained_backbone else None),
            in_channels=3,
            classes=num_classes,
        )
        return SMPWrapper(net)        # ★ 包装后返回

    else:
        raise ValueError(f"Unsupported model name: {name}")
