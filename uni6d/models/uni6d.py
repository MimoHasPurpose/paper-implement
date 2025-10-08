"""
Uni6D model architecture (PyTorch)

This file contains a clean, modular implementation of a Uni6D-like
6D pose estimation architecture.

Components:
- ResNet backbone (configurable)
- Simple FPN neck
- Multi-head pose predictor:
    - objectness/segmentation head
    - rotation head (quaternion)
    - translation head (3D offset)
    - confidence head (optional)

Notes:
- This implementation is intended as a starting point for research.
- Losses, dataset loaders, and training loop are not included.

Usage:
    from uni6d_model import Uni6D
    model = Uni6D(num_classes=21, backbone='resnet50', pretrained=False)
    x = torch.randn(2, 3, 480, 640)
    outs = model(x)

"""
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def conv_bn_relu(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FPN(nn.Module):
    """Small Feature Pyramid Network.

    Accepts a dict of features from a backbone and returns a single fused
    feature map at a higher resolution via lateral connections.
    """

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.smooth_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # features should be ordered from c3, c4, c5 (low -> high)
        xs = [l(f) for l, f in zip(self.lateral_convs, features)]
        # top-down fusion
        for i in range(len(xs) - 1, 0, -1):
            up = F.interpolate(xs[i], size=xs[i - 1].shape[-2:], mode='nearest')
            xs[i - 1] = xs[i - 1] + up
        out = self.smooth_convs[0](xs[0])
        # optionally upsample to a higher resolution
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return out


class PoseHead(nn.Module):
    """Heads for rotation, translation, and mask/objectness.

    The heads are simple conv stacks followed by final prediction convs.
    Rotation is parameterized with a 4-d quaternion (not normalized here).
    """

    def __init__(self, in_channels, num_classes=1, head_channels=128, pred_quat=True):
        super().__init__()
        self.shared = nn.Sequential(
            conv_bn_relu(in_channels, head_channels, 3, 1, 1),
            conv_bn_relu(head_channels, head_channels, 3, 1, 1),
        )
        # segmentation/objectness head (per-pixel or per-instance heatmap)
        self.mask_head = nn.Sequential(
            conv_bn_relu(head_channels, head_channels // 2, 3, 1, 1),
            nn.Conv2d(head_channels // 2, num_classes, 1),
        )
        # rotation head (quaternion)
        self.rot_head = nn.Sequential(
            conv_bn_relu(head_channels, head_channels // 2, 3, 1, 1),
            nn.Conv2d(head_channels // 2, 4 if pred_quat else 6, 1),
        )
        # translation head (3 channels x,y,z or offsets)
        self.trans_head = nn.Sequential(
            conv_bn_relu(head_channels, head_channels // 2, 3, 1, 1),
            nn.Conv2d(head_channels // 2, 3, 1),
        )
        # optional confidence scalar map
        self.conf_head = nn.Sequential(
            conv_bn_relu(head_channels, head_channels // 2, 3, 1, 1),
            nn.Conv2d(head_channels // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s = self.shared(x)
        mask = self.mask_head(s)
        rot = self.rot_head(s)
        trans = self.trans_head(s)
        conf = self.conf_head(s)
        return dict(mask=mask, rot=rot, trans=trans, conf=conf)


class Uni6D(nn.Module):
    """Uni6D-like architecture.

    Args:
        num_classes: number of object classes for mask/heatmap output. Use 1
            for binary per-instance mask when instances are separated by postproc.
        backbone: one of 'resnet18','resnet34','resnet50'.
        pretrained: whether to load ImageNet weights for backbone.
        fpn_out_channels: number of channels from FPN.
    """

    def __init__(
        self,
        num_classes: int = 1,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        fpn_out_channels: int = 256,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = self._make_backbone(backbone, pretrained)
        # We will use resnet layers: layer2, layer3, layer4 (c3,c4,c5)
        in_channels_list = self._backbone_out_channels(backbone)
        self.fpn = FPN(in_channels_list, out_channels=fpn_out_channels)
        self.pose_head = PoseHead(fpn_out_channels, num_classes=num_classes)

    def _make_backbone(self, name: str, pretrained: bool):
        if name == 'resnet18':
            m = models.resnet18(pretrained=pretrained)
        elif name == 'resnet34':
            m = models.resnet34(pretrained=pretrained)
        elif name == 'resnet50':
            m = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError('Unsupported backbone')
        # remove classification head
        m = nn.Sequential(
            m.conv1,
            m.bn1,
            m.relu,
            m.maxpool,
            m.layer1,
            m.layer2,
            m.layer3,
            m.layer4,
        )
        return m

    def _backbone_out_channels(self, name: str):
        # channels for resnet c3, c4, c5
        if name in ('resnet18', 'resnet34'):
            return (128, 256, 512)
        elif name == 'resnet50':
            return (512, 1024, 2048)
        else:
            raise ValueError('Unsupported backbone')

    def forward_backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Manually run through layers to capture intermediate outputs
        # x -> conv1..maxpool -> layer1 -> layer2 (c3) -> layer3 (c4) -> layer4 (c5)
        m = self.backbone
        x = m[0](x)  # conv1
        x = m[1](x)  # bn1
        x = m[2](x)  # relu
        x = m[3](x)  # maxpool
        x = m[4](x)  # layer1
        c3 = m[5](x)  # layer2
        c4 = m[6](c3)  # layer3
        c5 = m[7](c4)  # layer4
        return c3, c4, c5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        c3, c4, c5 = self.forward_backbone(x)
        f = self.fpn((c3, c4, c5))
        preds = self.pose_head(f)
        # postprocessing hint: rotations are per-pixel. Usually you will sample
        # at object centers or apply NMS on mask/heatmap to get instance-level poses.
        return preds


if __name__ == '__main__':
    # quick smoke test
    model = Uni6D(num_classes=1, backbone='resnet50', pretrained=False)
    # print(model.summary())
    x = torch.randn(2, 3, 480, 640)
    outs = model(x)
    for k, v in outs.items():
        print(k, v.shape)
