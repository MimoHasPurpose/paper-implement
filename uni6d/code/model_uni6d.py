# model_uni6d.py
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.resnet import resnet50
from torchvision.ops import MultiScaleRoIAlign

# RT head: predict quaternion (4) and translation (3) per RoI
class RTHead(nn.Module):
    def __init__(self, in_channels, fc_dim=1024):
        super().__init__()
        # simple MLP head similar to paper (two FCs)
        self.fc1 = nn.Linear(in_channels, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.act = nn.ReLU(inplace=True)
        # outputs per RoI: quaternion (4) + translation (3)
        self.q_out = nn.Linear(fc_dim, 4)
        self.t_out = nn.Linear(fc_dim, 3)

    def forward(self, x):
        # x: N x C (flattened RoI pooled feature)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        q = self.q_out(x)
        t = self.t_out(x)
        # normalize quaternion
        q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
        return q, t

# abc head: FCN-like to output 3-channel map per RoI (14x14 -> 14x14x3)
class ABCHead(nn.Module):
    def __init__(self, in_channels, spatial_size=(14,14)):
        super().__init__()
        # a small conv net to produce 3 channels
        C = in_channels
        self.conv1 = nn.Conv2d(C, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, 3, padding=1)
        self.out = nn.Conv2d(128, 3, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: N x C x H x W (RoI features)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        out = self.out(x)  # N x 3 x H x W
        # pool / reshape if you want per-RoI vector; the paper uses full map
        return out

def adapt_pretrained_conv1(conv1, new_in_channels):
    """
    conv1: nn.Conv2d pretrained (in_channels=3)
    new_in_channels: desired total channels (e.g. 27)
    Approach: copy the existing weights for RGB into first 3 channels, init others with kaiming.
    """
    import torch.nn.init as init
    old_w = conv1.weight.data  # [out,3,k,k]
    out_ch, _, k, _ = old_w.shape
    new_w = torch.zeros((out_ch, new_in_channels, k, k))
    # copy for RGB
    new_w[:, :3, :, :] = old_w
    # for other channels, init with mean of RGB kernels to give stable start
    mean_rgb = old_w.mean(dim=1, keepdim=True)
    for c in range(3, new_in_channels):
        new_w[:, c:c+1, :, :] = mean_rgb
    new_conv = nn.Conv2d(new_in_channels, out_ch, kernel_size=k, stride=conv1.stride, padding=conv1.padding, bias=(conv1.bias is not None))
    new_conv.weight.data = new_w
    if conv1.bias is not None:
        new_conv.bias.data = conv1.bias.data
    return new_conv

def build_uni6d(num_classes, in_channels=27, pe_D=16, pretrained_backbone=True, device='cuda'):
    """
    num_classes: number of object classes + 1 for background (torchvision expects num_classes incl background)
    in_channels: total input channels (RGB + D + UV + XY + NRM + PE)
    """
    # load a resnet50 backbone from torchvision and make it a feature extractor with FPN
    # torchvision provides a helper to get a backbone; but we'll construct ResNet and modify the first conv
    backbone = resnet50(pretrained=pretrained_backbone)
    # modify conv1
    backbone.conv1 = adapt_pretrained_conv1(backbone.conv1, new_in_channels=in_channels)
    # keep layers up to layer4
    # create an FPN on top using torchvision's feature extractor utilities
    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    # select the returned layers
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    backbone_with_fpn = BackboneWithFPN(backbone, return_layers=return_layers, in_channels_list=[256,512,1024,2048], out_channels=256)
    # create MaskRCNN
    model = torchvision.models.detection.MaskRCNN(backbone_with_fpn, num_classes=num_classes, rpn_anchor_generator=None)
    # Now add our RT and abc heads
    # We need to know the channel dimension of the RoI features: MaskRCNN roi_heads.box_head output dim is 1024 typically.
    # We will attach RT head to the flattened box_head representation; the mask head uses conv features.
    # The roi_heads.box_head in torchvision (for ResNet50-FPN) is a TwoMLPHead that outputs 1024-d vector per RoI
    # So get that out_features:
    box_head_out_features = model.roi_heads.box_head.fc6.out_features if hasattr(model.roi_heads.box_head.fc6, 'out_features') else 1024
    # create RT head
    model.roi_heads.rt_head = RTHead(in_channels=box_head_out_features, fc_dim=1024)
    # attach ABC head: we will run it on the features before box_head (roi features spatial), that is model.roi_heads.mask_head's input
    # Find the channel count of the pooled features: mask_head.conv5 has in_channels 256 for standard MaskRCNN
    pooled_feature_channels = 256
    model.roi_heads.abc_head = ABCHead(in_channels=pooled_feature_channels, spatial_size=(14,14))
    # For training we will compute RT/abc losses in train loop using these heads and the RoI features.
    return model
