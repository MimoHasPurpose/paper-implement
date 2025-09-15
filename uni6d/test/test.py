# resnet50_fpn_test.py
"""
Run a ResNet-50 + FPN backbone on a folder of images and inspect feature maps.

Usage:
    python resnet50_fpn_test.py --img_dir /path/to/images --batch_size 4 --visualize

Requirements:
    pip install torch torchvision pillow matplotlib
"""

import os
import argparse
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF

# Simple image dataset (reads common image formats)
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = sorted([
            os.path.join(root, f) for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = TF.to_tensor(img)
        return img_t, os.path.basename(p)

def make_backbone(device, pretrained=True, trainable_layers=3):
    """
    Construct ResNet50 + FPN backbone using torchvision helper.
    Returns a nn.Module that maps images -> dict[str -> Tensor] feature maps.
    The keys are usually: ['0','1','2','3'] or ['0','1','2','3','pool'] depending on versions.
    """
    # name 'resnet50' and pretrained True will load torchvision weights for the backbone part
    backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained, trainable_layers=trainable_layers)
    backbone.to(device)
    backbone.eval()
    return backbone

def visualize_feature_map(feature, title=None, n_channels=4):
    """
    Visualize the first n_channels of a feature map (C x H x W) as small images.
    """
    C, H, W = feature.shape
    n = min(n_channels, C)
    fig, axs = plt.subplots(1, n, figsize=(3*n, 3))
    for i in range(n):
        ax = axs[i] if n>1 else axs
        fm = feature[i].cpu().numpy()
        # normalize for display
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        ax.imshow(fm, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"ch {i}")
    if title:
        fig.suptitle(title)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', required=True, help='Folder with test images')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--visualize', action='store_true', help='Show feature maps with matplotlib')
    args = parser.parse_args()

    device = torch.device(args.device)
    transform = transforms.Compose([
        transforms.Resize((480, 640)),   # standardized size (you can change)
        transforms.ToTensor(),
        # normalize by ImageNet stats since backbone is pretrained
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    dataset = ImageFolderDataset(args.img_dir, transform=transform)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in {args.img_dir}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=lambda x: list(zip(*x)))
    backbone = make_backbone(device, pretrained=True, trainable_layers=3)

    with torch.no_grad():
        for batch_idx, (imgs, names) in enumerate(dataloader):
            # imgs: tuple of tensors -> convert to list of tensors (C,H,W) for backbone API
            imgs = list(img.to(device) for img in imgs)
            # The backbone expects a single batched tensor or ImagesList in detection APIs.
            # resnet_fpn_backbone returns a module that expects a Tensor (B,C,H,W) or a list depending on version.
            # We'll stack to B,C,H,W here.
            batch = torch.stack(imgs, dim=0)
            # If backbone is written to accept list[Tensor], call backbone(images_list)
            try:
                features = backbone(batch)   # dict of feature_name -> Tensor (B,C,H,W) or (C,H,W) depending
            except Exception:
                # fallback: call with list of tensors (older torchvision)
                features = backbone(list(imgs))

            # Print feature map names and shapes
            print(f"\nBatch {batch_idx} - images: {names}")
            for k,v in features.items():
                if isinstance(v, (list, tuple)):
                    # sometimes returns list, handle gracefully
                    sizes = [x.shape for x in v]
                    print(f"  {k}: list of tensors with shapes {sizes}")
                else:
                    print(f"  {k}: {tuple(v.shape)}")  # e.g. (B, 256, H', W')

            # Visualize per-image first feature level (if requested)
            if args.visualize:
                # choose the highest-resolution FPN feature (usually '0' or 'fpn_lateral2' depending on torchvision)
                # We'll pick the smallest stride (largest H) by looking at shapes.
                # features: dict name->Tensor(B,C,H,W)
                # find entry with largest H
                best_k = None
                best_H = -1
                for k,v in features.items():
                    t = v
                    if isinstance(t, (list,tuple)):
                        t = t[0]
                    _, C, H, W = t.shape
                    if H > best_H:
                        best_H = H; best_k = k
                print("Visualizing feature level:", best_k)
                feat = features[best_k]  # B,C,H,W
                # visualize for first image in batch
                feat_img0 = feat[0].cpu()
                visualize_feature_map(feat_img0, title=f"Feature {best_k} (img {names[0]})", n_channels=6)

            # only run one batch for quick test; remove the break to run all
            break

if __name__ == "__main__":
    main()
