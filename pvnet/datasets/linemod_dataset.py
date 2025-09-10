# datasets/linemod_dataset.py
# Flexible LINEMOD loader tuned to varied folder layouts.
# Outputs a dict with keys: image (Tensor C,H,W float [0..1]), vec_gt (2K,Hs,Ws),
# mask_s (Hs,Ws), kpts2d (K,2), K (3x3), R (3x3), t (3,)
import pdb
import os
import csv
import json
import pickle
from glob import glob

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def _read_pose_txt(path):
    """Expect a single line with 12 numbers: r11..r33 tx ty tz OR 9+3 with spaces."""
    with open(path, 'r') as f:
        s = f.read().strip().replace(',', ' ')
    parts = [float(x) for x in s.split() if x != '']
    if len(parts) >= 12:
        R = np.array(parts[:9], dtype=np.float32).reshape(3,3)
        t = np.array(parts[9:12], dtype=np.float32)
        return R, t
    elif len(parts) >= 6:
        # fallback: maybe R as Rodrigues (3) + t (3)
        if 6 <= len(parts) < 9:
            return None
    return None

class LineMODDataset(Dataset):
    def __init__(self, obj_path, input_size=480, training=True):
        """
        obj_path: Either the object folder (e.g. ".../LINEMOD/cat") or parent folder.
        """
        # Resolve object folder
        if os.path.isdir(os.path.join(obj_path, 'JPEGImages')) or os.path.isdir(os.path.join(obj_path,'mask')): # checking if image folder and masks folder exists
            self.root = obj_path ## root set to object path
            print("break:",os.path.isdir(os.path.join(obj_path,'JPEGImages')))
        else:
            candidates = [os.path.join(obj_path, d) for d in os.listdir(obj_path) if os.path.isdir(os.path.join(obj_path,d))]
            if len(candidates) == 1:
                self.root = candidates[0]
            else:
                chosen = None
                for d in candidates:
                    name = os.path.basename(d).lower()
                    if name in ['cat','duck','ape','iron','bench','bicycle','camera']:
                        chosen = d; break
                if chosen is None and candidates:
                    chosen = candidates[0]
                if chosen is None:
                    raise RuntimeError(f"No object subfolder found inside {obj_path}")
                self.root = chosen

        self.input_size = int(input_size)
        if self.input_size % 4 != 0:
            # make sure divisible by 4 for the 1/4 stride
            self.input_size = (self.input_size // 4) * 4
            if self.input_size == 0:
                self.input_size = 480
        self.training = training

        # find image dir
        for name in ('JPEGImages', 'images', 'test', 'img', 'rgb'):
            p = os.path.join(self.root, name)
            if os.path.isdir(p):
                print("break: ",os.path.isdir(p))
                self.img_dir = p; break
        else:
            # fallback: any directory that contains jpg/png files
            all_dirs = [os.path.join(self.root,d) for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root,d))]
            found = None
            for p in all_dirs:
                if any(fn.lower().endswith(x) for fn in os.listdir(p) for x in ['.jpg','.png']):
                    found = p; break
            if found:
                self.img_dir = found
            else:
                # fallback to root (maybe images are directly here)
                self.img_dir = self.root

        # find mask dir
        for name in ('mask','masks','amodal_mask'):
            p = os.path.join(self.root, name)
            if os.path.isdir(p):
                print("break:",name, os.path.isdir(p))
                self.mask_dir = p; break
        else:
            self.mask_dir = None

        # camera intrinsics (try common files)
        K_candidates = ['camera.json','cam.json','meta.json','info.json']
        self.K = None
        self.dist = np.zeros((5,), dtype=np.float32)
        for c in K_candidates:
            p = os.path.join(self.root, c)
            # print("break: camera intrinsics:", p)
            if os.path.exists(p):
                print("break: camera intrinsics:", p)
                try:
                    with open(p,'r') as f:
                        j = json.load(f)
                    if 'K' in j:
                        self.K = np.array(j['K'], dtype=np.float32)
                    if 'dist' in j:
                        self.dist = np.array(j.get('dist',[0,0,0,0,0]), dtype=np.float32)
                    if 'keypoints_3d' in j:
                        self.kpts3d = np.array(j['keypoints_3d'], dtype=np.float32)
                except Exception:
                    pass
                break

        if self.K is None:
            # default focal, center guess - user should replace with real intrinsics or put a meta.json
            # print("Warning: no camera intrinsics found (camera.json/meta.json). Using a default K.")
            self.K = np.array([[600.,0.,self.input_size/2.],
                               [0.,600.,self.input_size/2.],
                               [0.,0.,1.]], dtype=np.float32)

        # load poses: prefer CSV splits, else per-image pose files, else test.json/test.pkl
        self.items = []
        csv_candidates = [os.path.join(self.root,'train_split.csv'), os.path.join(self.root,'test_split.csv')]
        print("break:csv",csv_candidates[0],csv_candidates[1])
        a="E:\\1Github\\Research\\paper-implement\\datasets\\LINEMOD\\cat\\train_split.csv"
        b="E:\\1Github\\Research\\paper-implement\\datasets\\LINEMOD\\cat\\test_split.csv"

        csv_candidates[0]=a
        csv_candidates[1]=b
        
        
        csv_found = None
        for c in csv_candidates:
            if os.path.exists(c):
                # print(c)
                csv_found = c
                break
        # print("break",csv_found)
        if False: #real value was csv_found but i falsed it
            print("oky",csv_found)
            with open(csv_found, newline='') as f:
                reader = csv.DictReader(f)
                # print(reader)
                for r in reader:
                    # train_split.csv, and test_split.csv are not as per this format!
                    # R = np.fromstring(r['R'], sep=' ').reshape(3,3).astype(np.float32)
                    R = np.array([float(x) for x in r['cam_R_m2c'].split()]).reshape(3,3).astype(np.float32)

                    # t = np.fromstring(r['t'], sep=' ').astype(np.float32)
                    t = np.array([float(x) for x in r['cam_t_m2c'].split()]).astype(np.float32)

                    self.items.append({'image': r['image'], 'mask': r.get('mask', r['image']), 'R': R, 't': t})
        
        else:
            print("breaks",end="\n")
            # look for a pose folder with one file per image, or a test.json/test.pkl
            pose_dir = os.path.join(self.root, 'pose')
            # pdb.set_trace()
            if os.path.isdir(pose_dir):
                print("pose folder:",pose_dir)
                img_list = sorted([os.path.basename(x) for x in glob(os.path.join(self.img_dir,'*')) if os.path.isfile(x)])
                # print("img: ",img_list)
                for img_name in img_list:
                    # print("image name: ",img_name)
                    base = os.path.splitext(img_name)[0]
                    # print("base",base)
                    for ext in ('.txt','.pose','.csv'):
                        p = os.path.join(pose_dir, base + ext)
                        # print("p is a thing:",p)
                        # pdb.set_trace()
                        if os.path.exists(p):
                            # print("p exists: ",p)
                            parsed = _read_pose_txt(p)
                            # print("parsed",parsed)
                            if parsed:
                                R,t = parsed
                                self.items.append({'image': img_name, 'mask': img_name, 'R': R, 't': t})
                                break
                    else:
                        continue
            else:
                json_p = os.path.join(self.root, 'test.json')
                pkl_p = os.path.join(self.root, 'test.pkl')
                if os.path.exists(json_p):
                    try:
                        with open(json_p,'r') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            for row in data:
                                if 'filename' in row and ('R' in row or 'r' in row):
                                    R = np.array(row.get('R') or row.get('r'), dtype=np.float32).reshape(3,3)
                                    t = np.array(row.get('t'), dtype=np.float32)
                                    self.items.append({'image': row['filename'], 'mask': row.get('mask', row['filename']), 'R': R, 't': t})
                        elif isinstance(data, dict):
                            for k,v in data.items():
                                R = np.array(v.get('R') or v.get('r'), dtype=np.float32).reshape(3,3)
                                t = np.array(v.get('t'), dtype=np.float32)
                                self.items.append({'image': k, 'mask': k, 'R': R, 't': t})
                    except Exception:
                        pass
                elif os.path.exists(pkl_p):
                    try:
                        with open(pkl_p,'rb') as f:
                            data = pickle.load(f)
                        if isinstance(data, list):
                            for row in data:
                                if 'filename' in row and 'R' in row:
                                    R = np.array(row['R'], dtype=np.float32).reshape(3,3)
                                    t = np.array(row['t'], dtype=np.float32)
                                    self.items.append({'image': row['filename'], 'mask': row.get('mask', row['filename']), 'R': R, 't': t})
                    except Exception:
                        pass

        if len(self.items) == 0:
            raise RuntimeError(f"No annotations found for dataset at {self.root}. Provide train_split.csv or per-image pose files in 'pose/'.")

        # read keypoints_3d: try common files 'corners', 'farthest', 'dense_pts' or meta
        if not hasattr(self, 'kpts3d') or self.kpts3d is None:
            for cand in ('corners','dense_pts','dense_pts.txt','corners.txt','farthest'):
                p = os.path.join(self.root, cand)
                if os.path.exists(p):
                    try:
                        arr = np.loadtxt(p)
                        if arr.ndim == 1:
                            arr = arr.reshape(-1,3)
                        self.kpts3d = arr.astype(np.float32)
                        break
                    except Exception:
                        continue
            if not hasattr(self, 'kpts3d') or self.kpts3d is None:
                # fallback: create 8 arbitrary points in object frame (placeholder)
                self.kpts3d = np.array([[0.05,0.05,0.05],[-0.05,0.05,0.05],[0.05,-0.05,0.05],[-0.05,-0.05,0.05],
                                        [0.05,0.05,-0.05],[-0.05,0.05,-0.05],[0.05,-0.05,-0.05],[-0.05,-0.05,-0.05]], dtype=np.float32)

        # ensure kpts3d is Nx3
        k = np.array(self.kpts3d)
        if k.ndim == 1 and k.size % 3 == 0:
            k = k.reshape(-1,3)
        if k.ndim == 2 and k.shape[1] == 9:
            # some bad files had 9 columns per row; try to flatten then reshape
            k = k.reshape(-1,3)
        if k.ndim != 2 or k.shape[1] != 3:
            # fallback to 8 cube corners
            k = np.array([[0.05,0.05,0.05],[-0.05,0.05,0.05],[0.05,-0.05,0.05],[-0.05,-0.05,0.05],
                          [0.05,0.05,-0.05],[-0.05,0.05,-0.05],[0.05,-0.05,-0.05],[-0.05,-0.05,-0.05]], dtype=np.float32)
        self.kpts3d = k.astype(np.float32)

        # build full item paths
        for it in self.items:
            it['image_path'] = os.path.join(self.img_dir, it['image'])
            if self.mask_dir:
                it['mask_path'] = os.path.join(self.mask_dir, it.get('mask', it['image']))
            else:
                it['mask_path'] = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = cv2.imread(it['image_path'], cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image {it['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if it['mask_path'] and os.path.exists(it['mask_path']):
            mask = cv2.imread(it['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = np.ones((h,w), dtype=np.uint8)

        # Project 3D keypoints to 2D using R,t and intrinsics
        R = it['R']
        t = it['t'].reshape(3,1)
        rvec, _ = cv2.Rodrigues(R)
        kpts2d, _ = cv2.projectPoints(self.kpts3d, rvec, t, self.K, self.dist)
        kpts2d = kpts2d.reshape(-1,2)

        # ------------------ FIXED SQUARE RESIZE (avoid feature-map mismatch) ------------------
        # Force non-aspect-preserving resize to input_size x input_size so model branches align.
        orig_h, orig_w = h, w
        new_h = new_w = int(self.input_size)

        # compute per-axis scales
        scale_x = new_w / float(orig_w)
        scale_y = new_h / float(orig_h)

        # resize image and mask to exact square
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # scale 2D keypoints accordingly (x scaled by scale_x, y by scale_y)
        kpts2d[:, 0] = kpts2d[:, 0] * scale_x
        kpts2d[:, 1] = kpts2d[:, 1] * scale_y

        # update intrinsics K for new scale
        K_new = self.K.copy()
        K_new[0, 0] *= scale_x
        K_new[1, 1] *= scale_y
        K_new[0, 2] *= scale_x
        K_new[1, 2] *= scale_y

        # crop to multiples of 4 (should be safe since input_size is multiple of 4)
        H4, W4 = (new_h//4)*4, (new_w//4)*4
        img = img[:H4, :W4]
        mask = mask[:H4, :W4]
        kpts2d = np.clip(kpts2d, [0,0], [W4-1,H4-1])

        Hs, Ws = H4//4, W4//4
        mask_s = cv2.resize(mask, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
        # ---------------------------------------------------------------------------------------

        # pixel centers at stride-4
        ys, xs = np.mgrid[0:Hs, 0:Ws]
        xs_full = xs*4 + 2
        ys_full = ys*4 + 2

        k = self.kpts3d.shape[0]
        vec = np.zeros((2*k, Hs, Ws), dtype=np.float32)
        for i in range(k):
            dx = kpts2d[i,0] - xs_full
            dy = kpts2d[i,1] - ys_full
            mag = np.sqrt(dx*dx + dy*dy) + 1e-6
            vec[i, ...] = dx / mag
            vec[i+k, ...] = dy / mag

        vec *= mask_s[None,:,:]

        img_t = torch.from_numpy(img.transpose(2,0,1)).float()/255.0
        vec_t = torch.from_numpy(vec)
        mask_t = torch.from_numpy(mask_s.astype(np.float32))
        kpts2d_t = torch.from_numpy(kpts2d.astype(np.float32))

        return {
            'image': img_t,
            'vec_gt': vec_t,
            'mask_s': mask_t,
            'kpts2d': kpts2d_t,
            'K': torch.from_numpy(K_new),
            'R': torch.from_numpy(R),
            't': torch.from_numpy(t.flatten()),
        }
