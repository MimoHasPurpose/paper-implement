# utils_pose.py
import numpy as np
import torch

def quaternion_to_rotmat_np(q):
    # q shape (...,4) assume (w,x,y,z)
    w,x,y,z = q[...,0], q[...,1], q[...,2], q[...,3]
    # build rotation matrices
    R = np.zeros(q.shape[:-1] + (3,3))
    R[...,0,0] = 1 - 2*(y*y + z*z)
    R[...,0,1] = 2*(x*y - z*w)
    R[...,0,2] = 2*(x*z + y*w)
    R[...,1,0] = 2*(x*y + z*w)
    R[...,1,1] = 1 - 2*(x*x + z*z)
    R[...,1,2] = 2*(y*z - x*w)
    R[...,2,0] = 2*(x*z - y*w)
    R[...,2,1] = 2*(y*z + x*w)
    R[...,2,2] = 1 - 2*(x*x + y*y)
    return R

def ADD_distance(pred_R, pred_T, gt_R, gt_T, model_vertices):
    """
    model_vertices: Vx3 numpy
    pred_R, pred_T, gt_R, gt_T either numpy arrays (3x3,3)
    computes mean of ||(R_pred x + T_pred) - (R_gt x + T_gt)|| over vertices
    """
    X = model_vertices.T  # 3 x V
    pred_pts = pred_R.dot(X) + pred_T[:,None]
    gt_pts = gt_R.dot(X) + gt_T[:,None]
    diff = pred_pts - gt_pts
    return np.mean(np.linalg.norm(diff, axis=0))
