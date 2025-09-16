# dataset_linemod.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import json
import math

# Utility: compute depth normals from depth map
def compute_depth_normals(depth, intrinsics):
    """
    depth: HxW float32 in meters (0 for missing)
    intrinsics: dict with fx, fy, cx, cy
    returns NxHxWx3 normals (here HxWx3)
    """
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    h, w = depth.shape
    # compute 3D points
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    Z = depth.copy()
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    pts = np.stack([X, Y, Z], axis=-1)  # HxWx3
    # compute gradients
    dzdx = np.gradient(pts, axis=1)
    dzdy = np.gradient(pts, axis=0)
    # cross product to get normals
    nx = dzdy[...,1]*dzdx[...,2] - dzdy[...,2]*dzdx[...,1]
    ny = dzdy[...,2]*dzdx[...,0] - dzdy[...,0]*dzdx[...,2]
    nz = dzdy[...,0]*dzdx[...,1] - dzdy[...,1]*dzdx[...,0]
    n = np.stack([nx, ny, nz], axis=-1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    n = n / norm
    # replace invalid where depth==0
    n[Z==0] = 0
    return n.astype(np.float32)

# Positional encoding (paper uses trigonometric PE)
def positional_encoding(h, w, D=16):
    # produce HxWxD channels
    pe = np.zeros((h, w, D), dtype=np.float32)
    # follow paper formula roughly: use sine/cos on normalized coords
    xs = (np.arange(w).astype(np.float32) / max(1, w-1)) * 2 * np.pi
    ys = (np.arange(h).astype(np.float32) / max(1, h-1)) * 2 * np.pi
    xs = xs[None, :, None]  # 1 x W x 1
    ys = ys[:, None, None]  # H x 1 x 1
    freqs = np.exp(np.linspace(0, np.log(10000.0), D//4))
    idx = 0
    for f in freqs:
        pe[..., idx] = np.sin(xs / f).squeeze()
        pe[..., idx+1] = np.cos(xs / f).squeeze()
        pe[..., idx+2] = np.sin(ys / f).squeeze()
        pe[..., idx+3] = np.cos(ys / f).squeeze()
        idx += 4
    return pe

class LineMODDataset(Dataset):
    """
    Minimal dataset for LineMOD. Expects per-frame:
      - color image (H x W x 3)
      - depth image (H x W) in meters (float32)
      - annotations: list of objects with bbox, mask, class_id, pose (R,T)
      - camera intrinsics (fx,fy,cx,cy)
      - object models (3D vertices) loaded elsewhere for evaluation
    Adapt paths and parsing according to your LineMOD copy.
    """
    def __init__(self, frames_list, objects_info, transforms=None, pe_D=16):
        """
        frames_list: list of dicts pointing to color/depth/anno files
        objects_info: dict mapping class_id -> model vertices (Nx3)
        """
        self.frames = frames_list
        self.objects_info = objects_info
        self.transforms = transforms
        self.pe_D = pe_D

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        entry = self.frames[idx]
        color = cv2.imread(entry['color_path'], cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
        depth = cv2.imread(entry['depth_path'], cv2.IMREAD_UNCHANGED).astype(np.float32)
        # If depth is in mm
        if depth.max() > 20:
            depth = depth / 1000.0
        h, w = depth.shape[:2]
        intr = entry['intrinsics']  # dict fx,fy,cx,cy

        # UV plain channels
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        U = xs.astype(np.float32)
        V = ys.astype(np.float32)
        # inverse projected XY
        X = (U - intr['cx']) * depth / intr['fx']
        Y = (V - intr['cy']) * depth / intr['fy']
        XY = np.stack([X, Y], axis=-1).astype(np.float32)

        # normals
        NRM = compute_depth_normals(depth, intr)  # HxWx3

        # positional encoding
        PE = positional_encoding(h, w, D=self.pe_D)  # HxWxD

        # build input tensor: RGB(3), D(1), UV(2), XY(2), NRM(3), PE(D)
        rgb = color.astype(np.float32)
        depth_ch = depth[..., None].astype(np.float32)
        uv = np.stack([U, V], axis=-1).astype(np.float32)
        in_tensor = np.concatenate([rgb, depth_ch, uv, XY, NRM, PE], axis=-1)  # H x W x C_in
        # transpose to C x H x W
        in_tensor = torch.from_numpy(in_tensor).permute(2,0,1)

        # Prepare target in the format expected by torchvision Mask R-CNN
        # target = dict with boxes, labels, masks, image_id, area, iscrowd
        targets = []
        # We assume annotation provides per-object mask and bbox and class_id and pose
        annos = entry['annos']
        boxes = []
        labels = []
        masks = []
        poses = []  # store R,T for computing abc/RT losses
        for a in annos:
            boxes.append(a['bbox'])  # [xmin,ymin,xmax,ymax]
            labels.append(a['class_id'])
            masks.append(a['mask'].astype(np.uint8))
            poses.append({'R': np.array(a['R'], dtype=np.float32), 'T': np.array(a['T'], dtype=np.float32)})
        if len(boxes) == 0:
            # provide dummy
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0,h,w), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(np.array(boxes, dtype=np.float32))
            labels = torch.as_tensor(np.array(labels, dtype=np.int64))
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        # attach poses info for the RT head / abc head computation during training
        target['poses'] = poses
        target['intrinsics'] = intr

        return in_tensor, target
