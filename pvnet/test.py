import open3d as o3d

# Replace "path/to/your_file.ply" with the actual path to your PLY file
ply_file_path = "/home/mimo/Desktop/github/paper-implement/datasets/LINEMOD/cat.ply" 

# Read the PLY file into an Open3D point cloud object
point_cloud = o3d.io.read_point_cloud(ply_file_path)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])