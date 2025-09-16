# train.py
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset_linemod import LineMODDataset
from model_uni6d import build_uni6d
import torchvision
import time
import numpy as np

def rt_loss(pred_q, pred_t, gt_R, gt_T, model_vertices, sampled_idxs=None):
    """
    Lrt = mean over sampled model vertices of || (R x + T) - (R* x + T*) ||.
    pred_q: Nx4 quaternion
    pred_t: Nx3 translation
    gt_R: Nx3x3 or same expressed from gt quaternion
    gt_T: Nx3
    model_vertices: Vx3 array of object's model vertices
    sampled_idxs: indices of vertices to sample. If None, sample 100 vertices uniformly.
    Returns scalar loss
    """
    bs = pred_q.shape[0]
    device = pred_q.device
    if sampled_idxs is None:
        # sample 100 vertices
        nverts = model_vertices.shape[0]
        sampled_idxs = np.random.choice(nverts, min(100, nverts), replace=False)
    sampled = torch.from_numpy(model_vertices[sampled_idxs].astype(np.float32)).to(device)  # Kx3
    K = sampled.shape[0]
    pred_R = quat_to_rotmat(pred_q)  # Nx3x3
    # transform
    pred_pts = torch.matmul(pred_R, sampled.t().unsqueeze(0).expand(bs,-1,-1)) + pred_t.unsqueeze(-1)  # Nx3xK
    gt_pts = torch.matmul(gt_R, sampled.t().unsqueeze(0).expand(bs,-1,-1)) + gt_T.unsqueeze(-1)
    diff = pred_pts - gt_pts
    loss = diff.norm(dim=1).mean()
    return loss

def abc_loss(pred_map, gt_3d_points_map, mask):
    """
    pred_map: N x 3 x H x W (predicted abc)
    gt_3d_points_map: N x 3 x H x W (groundtruth mapping of each pixel in RoI to model coordinates)
    mask: N x 1 x H x W boolean: only compute loss where mask is present (visible)
    L1 loss as in paper
    """
    l1 = torch.abs(pred_map - gt_3d_points_map)
    if mask is not None:
        l1 = l1 * mask
        denom = mask.sum() * 3.0 + 1e-8
    else:
        denom = torch.tensor(pred_map.numel(), device=pred_map.device)
    return l1.sum() / denom

def quat_to_rotmat(q):
    """
    q: N x 4 (w,x,y,z) or (x,y,z,w) depending. We'll assume q is (w,x,y,z).
    returns N x 3 x 3
    """
    # Normalize
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    w = q[:,0]; x=q[:,1]; y=q[:,2]; z=q[:,3]
    B = q.shape[0]
    R = torch.zeros((B,3,3), device=q.device)
    R[:,0,0] = 1 - 2*(y*y + z*z)
    R[:,0,1] = 2*(x*y - z*w)
    R[:,0,2] = 2*(x*z + y*w)
    R[:,1,0] = 2*(x*y + z*w)
    R[:,1,1] = 1 - 2*(x*x + z*z)
    R[:,1,2] = 2*(y*z - x*w)
    R[:,2,0] = 2*(x*z - y*w)
    R[:,2,1] = 2*(y*z + x*w)
    R[:,2,2] = 1 - 2*(x*x + y*y)
    return R

def train_one_epoch(model, dataloader, optimizer, device, epoch, objects_info, lambda0=1.0, lambda1=1.0):
    model.train()
    running_loss = 0.0
    for i, (img_tensor, targets) in enumerate(dataloader):
        # img_tensor: list? Because MaskRCNN expects list of images; adapt:
        # here img_tensor is batch_size x C x H x W, but MaskRCNN expects list[Tensor]
        imgs = [it.to(device) for it in img_tensor] if isinstance(img_tensor, list) or isinstance(img_tensor, tuple) else [img_tensor.to(device)]
        # torchvision MaskRCNN expects list of targets with keys boxes, labels, masks; our target includes poses in 'poses'
        targs = []
        for t in targets:
            newt = {}
            newt['boxes'] = t['boxes'].to(device)
            newt['labels'] = t['labels'].to(device)
            newt['masks'] = t['masks'].to(device)
            # keep pose and intrinsics for later
            newt['poses'] = t['poses']
            newt['intrinsics'] = t['intrinsics']
            targs.append(newt)
        # run MaskRCNN forward to get detection losses and roi features
        # Trick: torchvision's MaskRCNN forward accepts images and targets and returns losses dictionary.
        losses = model(imgs, targs)  # during training returns dict of losses
        # Now we need to compute RT and abc losses.
        # We must extract RoI features and the box_head pooled features for the proposals corresponding to gt objects.
        # Torchvision does not directly return the internal pooled features, so one way is to modify torchvision code or
        # re-run the roi_heads.forward with saved proposals. For simplicity, here's a practical approach:
        #  - after model(imgs, targs) call, model.roi_heads has process to compute features; but not returned
        # We'll call model.transform and model.backbone and roi_heads directly to get features for GT boxes.
        # This is more code but gives access to features:
        images, _ = model.transform(imgs)  # ImagesList
        features = model.backbone(images.tensors)
        proposals, _ = model.rpn(images, features, targs)  # proposals is list per image
        # now get roi_features for each proposal using roi_heads
        # But we want features for GT boxes. Let's build rois = targs boxes as proposals
        # Prepare box tensors in expected format
        from torchvision.models.detection.roi_heads import paste_masks_in_image
        # Build per-image rois as list of tensors [num_gt, 4]
        rois_for_heads = [t['boxes'].to(device) for t in targs]
        # Use roi_heads.box_roi_pool to pool features for these rois
        box_features = model.roi_heads.box_roi_pool(features, rois_for_heads, images.image_sizes)
        # box_features -> pass through box_head to get flattened representation
        box_features_flat = model.roi_heads.box_head(box_features)
        # compute RT predictions
        q_pred, t_pred = model.roi_heads.rt_head(box_features_flat)
        # For abc head, we need the spatial RoI features as for mask head
        mask_features = model.roi_heads.mask_roi_pool(features, rois_for_heads, images.image_sizes)
        abc_pred_maps = model.roi_heads.abc_head(mask_features)  # N x 3 x H x W

        # Now compute GT RT and ABC target maps from targs poses and the object's model vertices and render mapping into RoI map.
        # This mapping is dataset-dependent: we must produce gt_3d_points_map aligned with RoI spatial grid.
        # For now, we compute RT loss by sampling model vertices (available in objects_info).
        # Build gt_R and gt_T for each instance
        gt_Rs = []
        gt_Ts = []
        model_vertices_batch = []
        for img_idx, t in enumerate(targs):
            poses = t['poses']  # list of dicts (R, T)
            labels = t['labels'].cpu().numpy().tolist()
            for j,p in enumerate(poses):
                gt_Rs.append(torch.from_numpy(p['R'].astype(np.float32)).to(device))
                gt_Ts.append(torch.from_numpy(p['T'].astype(np.float32)).to(device))
                cls = labels[j]
                model_vertices_batch.append(objects_info[cls])  # Nx3 numpy
        if len(gt_Rs) > 0:
            gt_Rs = torch.stack(gt_Rs, dim=0)
            gt_Ts = torch.stack(gt_Ts, dim=0)
            # compute rt loss per sample: note matching order must align; here we assumed sequential extraction
            loss_rt = rt_loss(q_pred, t_pred, gt_Rs, gt_Ts, model_vertices_batch)
        else:
            loss_rt = torch.tensor(0.0, device=device)

        # abc loss would need groundtruth mapping of each pixel in RoI to object model coords; that's a dataset-specific mapping.
        # For now set abc loss to zero placeholder (you should implement generation of gt_3d_points_map from per-pixel model coordinate mapping).
        loss_abc = torch.tensor(0.0, device=device)

        total_loss = sum(loss for loss in losses.values()) + lambda0 * loss_rt + lambda1 * loss_abc

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        if i % 20 == 0:
            print(f"Epoch {epoch} iter {i} loss {total_loss.item():.4f}")

    return running_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # you need to prepare frames_list and objects_info (models) beforehand
    frames_train = ...  # list of entries for training
    objects_info = ...  # dict class_id -> vertices Nx3
    dataset = LineMODDataset(frames_train, objects_info, transforms=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    num_classes = max(objects_info.keys()) + 1  # +1 if classes start at 1
    in_channels = 3 + 1 + 2 + 2 + 3 + 16  # RGB + D + UV + XY + NRM + PE_D
    model = build_uni6d(num_classes=num_classes, in_channels=in_channels, pe_D=16, pretrained_backbone=True, device=device).to(device)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0075, momentum=0.9, weight_decay=1e-4)
    for epoch in range(40):
        loss = train_one_epoch(model, dataloader, optimizer, device, epoch, objects_info)
        print(f"Epoch {epoch} avg loss {loss:.4f}")

if __name__ == '__main__':
    main()
