import pdb
# training.py - robust PVNet training loop (drop-in)
import os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from test.test1 import test_img_print


# configuration:
DATA_ROOT = "../datasets/LINEMOD/cat"
BATCH_SIZE = 1
EPOCHS = 1
# LR = 1e-3
# NUM_WORKERS = 0
NUM_KEYPOINTS = 8
NUM_CLASSES = 1
CHECKPOINT_DIR = "checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)



NUM_WORKERS=0
BATCH_SIZE=1

from datasets.linemod_dataset import LineMODDataset

# from models.uni6d import Uni6d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# dataset + dataloader
dataset = LineMODDataset(DATA_ROOT, input_size=480, training=True)
# DataLoader
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=NUM_WORKERS, pin_memory=(device.type=='cuda'))

# model + optimizer
# model = Uni6d(num_keypoints=NUM_KEYPOINTS, num_classes=NUM_CLASSES).to(device)

# opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001,momentum=0.9)

def loss():
    pass



## training loop

for epochs in range(1, EPOCHS+1):
    # model.train()
    epoch_loss = 0.0
    n_samples = 0
    t0 = time.time()
    print("length of loader:",len(loader))
    for batch in loader:
        print("loading pics:")
        image = batch['image'].to(device, non_blocking=True)    # [B,3,H,W]
        # test_img_print(image)

    
        # print("line 74",image)
        vec_gt = batch['vec_gt'].to(device, non_blocking=True)  # [B,2K,Hs,Ws]
        mask_s = batch['mask_s'].to(device, non_blocking=True)  # [B,Hs,Ws]

        # out=model(image)

        # prediction


        # loss 
        # rt, abc, mask bbox, cls, rpn

        # ckpt_path = os.path.join(CHECKPOINT_DIR, f"pvnet_epoch{epoch}.pth")
        # torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': opt.state_dict()}, ckpt_path)
        # print("Saved checkpoint:", ckpt_path)






print("training finished:")