import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def load_intrinsics(path):
    """
    Example loader for intrinsics. Replace with your JSON/YAML reader.
    Return dict with fx, fy, cx, cy.
    """
    # Example intrinsics for LineMOD (you should load from your file!)
    return {"fx": 572.4114, "fy": 573.57043, "cx": 325.2611, "cy": 242.04899}

def draw_pose(img, R, T, intr, model_pts, color=(0,255,0)):
    """
    Project model points and draw 2D bounding box (wireframe).
    model_pts: Nx3 numpy array (CAD vertices).
    R: 3x3 rotation, T: 3x1 translation.
    """
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    proj = (R @ model_pts.T + T.reshape(3,1)).T  # Nx3
    xs = (proj[:,0] * fx / proj[:,2]) + cx
    ys = (proj[:,1] * fy / proj[:,2]) + cy
    # Draw projected points as small dots
    for (x,y) in zip(xs,ys):
        if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]:
            cv2.circle(img, (int(x),int(y)), 1, color, -1)
    return img

def visualize_dataset(rgb_dir, depth_dir, mask_dir, pose_file, model_file, intr_file):
    # load model
    mesh = o3d.io.read_triangle_mesh(model_file)
    model_pts = np.asarray(mesh.vertices)

    # load intrinsics
    intr = load_intrinsics(intr_file)

    # load ground truth poses (depends on your LineMOD split format)
    # Example: text file with R(3x3) and T(3,) per line
    poses = np.load(pose_file, allow_pickle=True).item()  # adapt to your gt file

    frame_ids = sorted(os.listdir(rgb_dir))
    idx = 0

    while True:
        rgb = cv2.imread(os.path.join(rgb_dir, frame_ids[idx]))
        depth = cv2.imread(os.path.join(depth_dir, frame_ids[idx].replace("color", "depth")),
                           cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(os.path.join(mask_dir, frame_ids[idx].replace("color", "mask")),
                          cv2.IMREAD_GRAYSCALE)

        rgb_vis = rgb.copy()

        if str(idx) in poses:
            pose = poses[str(idx)]
            R = pose["R"]   # 3x3
            T = pose["T"]   # 3x1
            rgb_vis = draw_pose(rgb_vis, R, T, intr, model_pts)

        # show everything
        fig, axs = plt.subplots(1,3, figsize=(15,5))
        axs[0].imshow(cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB))
        axs[0].set_title("RGB + pose projection")
        axs[1].imshow(depth, cmap="plasma")
        axs[1].set_title("Depth")
        axs[2].imshow(mask, cmap="gray")
        axs[2].set_title("Mask")
        for a in axs: a.axis("off")
        plt.show()

        key = input("Press Enter for next, or q to quit: ")
        if key.lower() == "q":
            break
        idx = (idx+1) % len(frame_ids)

if __name__ == "__main__":
    # Example usage: adapt these paths to your LineMOD folder structure
    rgb_dir = "LINEMOD/01/rgb/"
    depth_dir = "LINEMOD/01/depth/"
    mask_dir = "LINEMOD/01/mask/"
    pose_file = "LINEMOD/01/poses.npy"     # adapt
    model_file = "LINEMOD/models/obj_01.ply"
    intr_file = "LINEMOD/camera.json"      # adapt

    visualize_dataset(rgb_dir, depth_dir, mask_dir, pose_file, model_file, intr_file)
