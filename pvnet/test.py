# test.py - robust end-to-end inference (drop-in)
# Set DATA_ROOT below to your object folder and run the script.
import os
import sys
import glob
import numpy as np
import torch
import cv2

from models.pvnet import PVNet
from datasets.linemod_dataset import LineMODDataset
from utils.voting import ransac_voting

# ----------------- CONFIG -----------------
DATA_ROOT = r"..\datasets\LINEMOD\cat"   # <<< your provided path
DEFAULT_CKPT = os.path.join("checkpoints", "pvnet_epoch1.pth")
NUM_TEST_IMAGES = 2
NUM_CLASSES = 1
# ------------------------------------------

def safe_torch_load(path):
    """Try to load with weights_only if available, else fallback to classic load."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cuda", weights_only=True)
    except TypeError:
        print("torch.load: weights_only not supported on this PyTorch, falling back (be careful with untrusted files).")
        return torch.load(path, map_location="cuda")
    except Exception as e:
        print("torch.load fallback exception:", e)
        return torch.load(path)

def normalize_keypoints2d(kpts):
    """
    Robustly extract a single (x,y) per keypoint from ransac_voting outputs.
    Handles outputs like:
      - list of (x,y) pairs
      - list of (mean, cov) tuples
      - list of (candidates_array, cov)
      - nested ragged lists
    Returns: np.array shape (N,2) dtype float32
    """
    if kpts is None:
        raise ValueError("ransac returned None")

    # quick attempt: if numeric ndarray with shape (N,2) accept
    try:
        arr = np.asarray(kpts)
    except Exception:
        arr = None

    if isinstance(arr, np.ndarray) and arr.dtype != object:
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2].astype("float32")
        if arr.ndim == 1 and arr.size % 2 == 0:
            return arr.reshape(-1, 2).astype("float32")

    extracted = []
    for i, el in enumerate(kpts):
        # tuple/list case: common is (mean, cov) or (candidates, cov)
        if isinstance(el, (tuple, list)):
            first = el[0]
            a = np.asarray(first)
            if isinstance(a, np.ndarray) and a.ndim == 1 and a.size >= 2:
                extracted.append(a[:2].astype("float32"))
                continue
            if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[-1] >= 2:
                cand = a.reshape(-1, a.shape[-1])[:, :2]
                mean_xy = cand.mean(axis=0)
                extracted.append(mean_xy.astype("float32"))
                continue
            # try flatten and take first two
            try:
                flat = np.asarray(first).reshape(-1)
                if flat.size >= 2:
                    extracted.append(flat[:2].astype("float32"))
                    continue
            except Exception:
                pass
            # search within tuple/list parts
            found = False
            for part in el:
                try:
                    p = np.asarray(part)
                    if p.ndim == 1 and p.size >= 2:
                        extracted.append(p[:2].astype("float32"))
                        found = True
                        break
                    if p.ndim >= 2 and p.shape[-1] >= 2:
                        cand = p.reshape(-1, p.shape[-1])[:, :2]
                        extracted.append(cand.mean(axis=0).astype("float32"))
                        found = True
                        break
                except Exception:
                    continue
            if found:
                continue
            raise ValueError(f"Cannot interpret ransac element #{i}: {repr(el)}")

        else:
            # element not a tuple/list
            try:
                a = np.asarray(el)
                if a.ndim == 1 and a.size >= 2:
                    extracted.append(a[:2].astype("float32"))
                    continue
                if a.ndim >= 2 and a.shape[-1] >= 2:
                    extracted.append(a.reshape(-1, a.shape[-1])[0, :2].astype("float32"))
                    continue
                flat = np.asarray(el).reshape(-1)
                if flat.size >= 2:
                    extracted.append(flat[:2].astype("float32"))
                    continue
            except Exception:
                pass
            raise ValueError(f"Cannot interpret ransac element #{i}: {repr(el)}")

    if len(extracted) == 0:
        raise ValueError("No keypoints extracted from ransac output.")
    out = np.asarray(extracted, dtype="float32").reshape(-1, 2)
    return out

def solve_pnp_safe(kps2d, kps3d, K, min_points=4, reproj_thresh=8.0):
    """Robust wrapper around solvePnPRansac / solvePnP. Returns (R, t) or None."""
    kps2d = np.asarray(kps2d, dtype=np.float32)
    kps3d = np.asarray(kps3d, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)

    if kps2d.ndim != 2 or kps2d.shape[1] != 2:
        print("solve_pnp_safe: bad kps2d shape", kps2d.shape)
        return None
    if kps3d.ndim != 2 or kps3d.shape[1] != 3:
        print("solve_pnp_safe: bad kps3d shape", kps3d.shape)
        return None
    if kps2d.shape[0] != kps3d.shape[0]:
        print("solve_pnp_safe: mismatch counts", kps2d.shape[0], kps3d.shape[0])
        return None

    N = kps2d.shape[0]
    if N < min_points:
        print(f"solve_pnp_safe: need >= {min_points} points, got {N}")
        return None

    mask_finite = np.isfinite(kps2d).all(axis=1) & np.isfinite(kps3d).all(axis=1)
    if not mask_finite.all():
        kps2d = kps2d[mask_finite]
        kps3d = kps3d[mask_finite]
        if kps2d.shape[0] < min_points:
            print("solve_pnp_safe: after removing invalid points, < min_points")
            return None

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=kps3d,
            imagePoints=kps2d,
            cameraMatrix=K,
            distCoeffs=None,
            reprojectionError=reproj_thresh,
            flags=cv2.SOLVEPNP_ITERATIVE,
            iterationsCount=200,
            confidence=0.99
        )
        if not success or inliers is None or len(inliers) < min_points:
            # fallback to solvePnP if enough points exist
            if kps2d.shape[0] >= min_points:
                ok, rvec, tvec = cv2.solvePnP(kps3d, kps2d, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    R, _ = cv2.Rodrigues(rvec)
                    return R, tvec.flatten()
            print("solve_pnp_safe: RANSAC failed or too few inliers:", None if inliers is None else len(inliers))
            return None

        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.flatten()
    except Exception as e:
        print("solve_pnp_safe: exception:", e)
        return None

def main():
    if not os.path.isdir(DATA_ROOT):
        print("DATA_ROOT not found:", DATA_ROOT)
        sys.exit(1)
    print("Using DATA_ROOT:", DATA_ROOT)

    # build dataset
    dataset = LineMODDataset(DATA_ROOT, input_size=480, training=False)
    print("Dataset size:", len(dataset))

    # canonical 3D keypoints (from dataset)
    if not hasattr(dataset, "kpts3d") or dataset.kpts3d is None:
        print("Dataset has no kpts3d. Place farthestK.txt or meta.json with keypoints_3d in the object folder.")
        sys.exit(1)
    kpts3d = np.asarray(dataset.kpts3d, dtype="float32")
    print("Canonical kpts3d shape:", kpts3d.shape)

    # pick a checkpoint
    ckpt_path = DEFAULT_CKPT if os.path.exists(DEFAULT_CKPT) else None
    if ckpt_path is None:
        ck = glob.glob(os.path.join("checkpoints", "*.pth"))
        if ck:
            ckpt_path = ck[0]
    if ckpt_path is None:
        print("No checkpoint found in 'checkpoints/'. Provide a trained .pth file at DEFAULT_CKPT.")
        sys.exit(1)

    print("Loading checkpoint:", ckpt_path)
    ckpt = safe_torch_load(ckpt_path)

    # build model
    NUM_KEYPOINTS = kpts3d.shape[0]
    model = PVNet(num_keypoints=NUM_KEYPOINTS, num_classes=NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # load weights (handle common checkpoint structures)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        try:
            model.load_state_dict(ckpt)
        except Exception:
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                try:
                    sd = ckpt
                    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
                        new = {k.replace("module.", ""): v for k, v in sd.items()}
                        model.load_state_dict(new)
                    else:
                        raise
                except Exception as e:
                    print("Failed to load checkpoint:", e)
                    sys.exit(1)

    model.eval()
    print("Model loaded. Running inference on", min(NUM_TEST_IMAGES, len(dataset)), "images...")

    for idx in range(min(NUM_TEST_IMAGES, len(dataset))):
        item = dataset[idx]
        with torch.no_grad():
            image_t = item["image"].unsqueeze(0).to(device)  # [1,3,H,W]
            H_img = item["image"].shape[1]
            W_img = item["image"].shape[2]
            vecs_out, seg_out = model(image_t)
            vecs = vecs_out[0].cpu().numpy()  # [2K, H, W]
            mask = item["mask_s"].cpu().numpy() if isinstance(item["mask_s"], torch.Tensor) else item["mask_s"]

            raw_kp2d = ransac_voting(vecs, mask)
            print(f"\nImage {idx}: raw ransac type: {type(raw_kp2d)}")

            try:
                kp2d = normalize_keypoints2d(raw_kp2d)  # (N,2)
            except Exception as e:
                print("Failed to normalize ransac output:", e)
                try:
                    print("Raw ransac output (repr):", repr(raw_kp2d)[:1000])
                except Exception:
                    print("Raw ransac output could not be repr()-ed.")
                continue

            # Clip to image bounds
            kp2d[:, 0] = np.clip(kp2d[:, 0], 0, W_img - 1)
            kp2d[:, 1] = np.clip(kp2d[:, 1], 0, H_img - 1)

            # align counts
            if kp2d.shape[0] != kpts3d.shape[0]:
                print(f"WARNING: got {kp2d.shape[0]} 2D vs {kpts3d.shape[0]} 3D keypoints. Truncating to min.")
                n = min(kp2d.shape[0], kpts3d.shape[0])
                kp2d = kp2d[:n]
                kpts3d_use = kpts3d[:n]
            else:
                kpts3d_use = kpts3d

            print("DEBUG: kp2d shape:", kp2d.shape, "kpts3d shape:", kpts3d_use.shape)
            print("DEBUG: first kp2d (x,y):", kp2d[:min(8, kp2d.shape[0])])
            print("DEBUG: first kpts3d:", kpts3d_use[:min(8, kpts3d_use.shape[0])])

            if kp2d.shape[0] < 4:
                print(f"Skipping image {idx}: only {kp2d.shape[0]} keypoints available (<4).")
                continue

            cam = item["K"].cpu().numpy() if isinstance(item["K"], torch.Tensor) else item["K"]
            cam = np.asarray(cam, dtype="float32")

            res = solve_pnp_safe(kp2d, kpts3d_use, cam)
            if res is None:
                print(f"PnP failed for image {idx}.")
            else:
                R, t = res
                print(f"\n=== Image {idx} Pose ===")
                print("2D pts shape:", kp2d.shape, "3D pts shape:", kpts3d_use.shape)
                print("Pose R:\n", R)
                print("Pose t:\n", t)

    print("\nDone.")

if __name__ == "__main__":
    main()
