# training.py - robust PVNet training loop (drop-in)
import os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --------- USER CONFIG ----------
DATA_ROOT = "..\datasets\LINEMOD\cat"
BATCH_SIZE = 1
EPOCHS = 1
LR = 1e-3
NUM_WORKERS = 0
NUM_KEYPOINTS = 8
NUM_CLASSES = 1
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# -------------------------------

# local imports (make sure these files exist)
from datasets.linemod_dataset import LineMODDataset
from models.pvnet import PVNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# dataset + dataloader
dataset = LineMODDataset(DATA_ROOT, input_size=480, training=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=NUM_WORKERS, pin_memory=(device.type=='cuda'))

# model + optimizer
model = PVNet(num_keypoints=NUM_KEYPOINTS, num_classes=NUM_CLASSES).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# helper: cosine vector loss (vec_pred & vec_gt must be same spatial size)
def compute_vector_loss(vec_pred, vec_gt, mask_s):
    eps = 1e-6
    K = vec_pred.shape[1] // 2
    vx = vec_pred[:, :K, :, :]
    vy = vec_pred[:, K:, :, :]
    tx = vec_gt[:, :K, :, :]
    ty = vec_gt[:, K:, :, :]

    vnorm = torch.clamp(torch.sqrt(vx * vx + vy * vy), min=eps)
    tnorm = torch.clamp(torch.sqrt(tx * tx + ty * ty), min=eps)

    vxn = vx / vnorm
    vyn = vy / vnorm
    txn = tx / tnorm
    tyn = ty / tnorm

    cos = vxn * txn + vyn * tyn      # [B, K, Hs, Ws]
    mask = mask_s.unsqueeze(1)       # [B,1,Hs,Ws]
    loss_map = (1.0 - cos) * mask
    loss = loss_map.sum() / (mask.sum() * K + eps)
    return loss

# training loop
for epoch in range(1, EPOCHS + 1):
    print("starting epoch 1")
    model.train()
    epoch_loss = 0.0
    # print("starting epoch 2")
    n_samples = 0
    t0 = time.time()
    # print("starting epoch 3")

    for batch in loader:
        print("loading photos")
        image = batch['image'].to(device, non_blocking=True)    # [B,3,H,W]
        vec_gt = batch['vec_gt'].to(device, non_blocking=True)  # [B,2K,Hs,Ws]
        mask_s = batch['mask_s'].to(device, non_blocking=True)  # [B,Hs,Ws]

        # forward
        out = model(image)
        if isinstance(out, (tuple, list)):
            vec_pred_full = out[0]
            seg_pred_full = out[1] if len(out) > 1 else None
        else:
            vec_pred_full = out
            seg_pred_full = None

        if not torch.is_tensor(vec_pred_full):
            raise RuntimeError("Model vector output is not a tensor")

        # resize predicted vector field to supervision resolution (vec_gt spatial size)
        target_h, target_w = vec_gt.shape[2], vec_gt.shape[3]
        vec_pred = F.interpolate(vec_pred_full, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)

        # vector loss
        loss_vec = compute_vector_loss(vec_pred, vec_gt, mask_s)

        # segmentation loss handling (ensure shapes match)
        seg_loss = 0.0
        if seg_pred_full is not None:
            # downsample seg logits to supervision resolution
            seg_pred = F.interpolate(seg_pred_full, size=(target_h, target_w),
                                      mode='bilinear', align_corners=False)  # [B, C, Hs, Ws]

            # If seg_pred has multiple channels (e.g. bg+obj), pick the object logit channel
            if seg_pred.shape[1] == 1:
                seg_obj_logit = seg_pred  # already single-channel logit
            elif seg_pred.shape[1] >= 2:
                # assume channel ordering [bg, obj, ...]; use channel 1
                seg_obj_logit = seg_pred[:, 1:2, :, :]
            else:
                # unexpected channel count — fallback to mean across channels
                seg_obj_logit = seg_pred.mean(dim=1, keepdim=True)

            seg_target = mask_s.unsqueeze(1)  # [B,1,Hs,Ws] with 0/1 values

            # BCEWithLogits requires input and target same shape
            if seg_obj_logit.shape != seg_target.shape:
                # final safety: broadcast/reshape if necessary
                seg_obj_logit = seg_obj_logit[:, :1, :, :]
                if seg_obj_logit.shape[2:] != seg_target.shape[2:]:
                    seg_obj_logit = F.interpolate(seg_obj_logit, size=seg_target.shape[2:], 
                                                  mode='bilinear', align_corners=False)

            bce = torch.nn.BCEWithLogitsLoss(reduction='none')
            seg_map = bce(seg_obj_logit, seg_target)  # [B,1,Hs,Ws]
            seg_loss = (seg_map * seg_target).sum() / (seg_target.sum() + 1e-6)

        loss = loss_vec + 0.1 * seg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item() * image.size(0)
        n_samples += image.size(0)

    t1 = time.time()
    avg_loss = epoch_loss / (n_samples + 1e-9)
    print(f"[Epoch {epoch}/{EPOCHS}] loss: {avg_loss:.6f} time: {t1-t0:.1f}s")

    # save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"pvnet_epoch{epoch}.pth")
    torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': opt.state_dict()}, ckpt_path)
    print("Saved checkpoint:", ckpt_path)

print("Training finished.")
