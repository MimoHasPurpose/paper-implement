import cv2
import numpy as np

def solve_pnp(keypoints_2d, keypoints_3d, camera_matrix):
    keypoints_2d = np.array([k[0] for k in keypoints_2d], dtype=np.float32)
    keypoints_3d = np.array(keypoints_3d, dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(keypoints_3d, keypoints_2d, camera_matrix, None)
    if not success:
        raise RuntimeError("PnP failed.")

    R, _ = cv2.Rodrigues(rvec)
    t = tvec
    return R, t
