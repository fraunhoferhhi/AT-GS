#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import pymeshlab
from tqdm import tqdm
import open3d as o3d
import os
import re
import trimesh


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    if resolution != pil_image.size:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

@torch.compile
def build_rotation(r):
    q = torch.nn.functional.normalize(r)

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    # 取出四元数的各个分量
    rr, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # 计算重复使用的项
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    rx, ry, rz = rr * x, rr * y, rr * z

    # 使用就地操作填充旋转矩阵
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - rz)
    R[:, 0, 2] = 2 * (xz + ry)
    R[:, 1, 0] = 2 * (xy + rz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - rx)
    R[:, 2, 0] = 2 * (xz - ry)
    R[:, 2, 1] = 2 * (yz + rx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    # old_f = sys.stdout
    # class F:
    #     def __init__(self, silent):
    #         self.silent = silent

    #     def write(self, x):
    #         if not self.silent:
    #             if x.endswith("\n"):
    #                 old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
    #             else:
    #                 old_f.write(x)

    #     def flush(self):
    #         old_f.flush()

    # sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def normal2rotation(n):
    # construct a random rotation matrix from normal
    # it would better be positive definite and orthogonal
    n = torch.nn.functional.normalize(n)
    # w0 = torch.rand_like(n)
    w0 = torch.tensor([[1, 0, 0]]).expand(n.shape)
    R0 = w0 - torch.sum(w0 * n, -1, True) * n
    R0 *= torch.sign(R0[:, :1])
    R0 = torch.nn.functional.normalize(R0)
    R1 = torch.cross(n, R0)
    
    # i = 7859
    # print(R1[i])
    R1 *= torch.sign(R1[:, 1:2]) * torch.sign(n[:, 2:])
    # print(R1[i])
    R = torch.stack([R0, R1, n], -1)
    # print(R[i], torch.det(R).sum(), torch.trace(R[i]))
    q = rotmat2quaternion(R)
    # print(q[i], torch.norm(q[i]))
    # R = quaternion2rotmat(q)
    # print(R[i])
    # for i in range(len(q)):
    #     if torch.isnan(q[i].sum()):
    #         print(i)
    # exit()
    return q

@torch.compile()
def quaternion_multiply(a, b):
    """
    Multiply two sets of quaternions.
    
    Parameters:
    a (Tensor): A tensor containing N quaternions, shape = [N, 4]
    b (Tensor): A tensor containing N quaternions, shape = [N, 4]
    
    Returns:
    Tensor: A tensor containing the product of the input quaternions, shape = [N, 4]
    """
    a_norm=torch.nn.functional.normalize(a)
    b_norm=torch.nn.functional.normalize(b)
    w1, x1, y1, z1 = a_norm[:, 0], a_norm[:, 1], a_norm[:, 2], a_norm[:, 3]
    w2, x2, y2, z2 = b_norm[:, 0], b_norm[:, 1], b_norm[:, 2], b_norm[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    # result is normalized
    return torch.stack([w, x, y, z], dim=1) 

@torch.compile()
def quaternion2rotmat(q):
    r, x, y, z = q.split(1, -1)
    # R = torch.eye(4).expand([len(q), 4, 4]).to(q.device)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)
    ], -1).reshape([len(q), 3, 3]);
    return R

@torch.compile()
def rotmat2quaternion(R, normalize=False):
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-6
    r = torch.sqrt(1 + tr) / 2
    # print(torch.sum(torch.isnan(r)))
    q = torch.stack([
        r,
        (R[:, 2, 1] - R[:, 1, 2]) / (4 * r),
        (R[:, 0, 2] - R[:, 2, 0]) / (4 * r),
        (R[:, 1, 0] - R[:, 0, 1]) / (4 * r)
    ], -1)
    if normalize:
        q = torch.nn.functional.normalize(q, dim=-1)
    return q


def knn_pcl(pcl0, pcl1, feat, K):
    nn_dist, nn_idx, nn_vtx = knn_points(pcl0[..., :3][None], pcl1[..., :3][None], K=K+1, return_nn=True)
    nn_dist = nn_dist[0, :, 1:]
    nn_idx = nn_idx[0, :, 1:]
    nn_vtx = nn_vtx[0, :, 1:]
    nn_vtx = torch.mean(nn_vtx, axis=1)
    nn_feat = torch.mean(feat[nn_idx], axis=1)
    return nn_vtx, nn_feat


def compute_obb(vertices):
    # Compute covariance matrix
    cov_matrix = np.cov(vertices, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_idx]
    eigenvalues = eigenvalues[sort_idx]
    
    # Center the vertices
    centroid = np.mean(vertices, axis=0)
    vertices_centered = vertices - centroid
    
    # Rotate vertices to align with eigenvectors
    vertices_transformed = vertices_centered @ eigenvectors
    
    # Compute min and max coordinates in transformed space
    min_coords = np.min(vertices_transformed, axis=0)
    max_coords = np.max(vertices_transformed, axis=0)
    center = centroid
    # right multiply eigenvectors: local bbox to world
    # left multiply eigenvectors: world to local bbox 
    rotation = eigenvectors
    
    return center, rotation, min_coords, max_coords

def remove_vertex_colors(obj_path):
    """
    Read an OBJ file, remove all vertex colors, and overwrite the same file.

    :param obj_path: Path to the OBJ file to be modified.
    """
    with open(obj_path, 'r') as infile:
        lines = infile.readlines()

    with open(obj_path, 'w') as outfile:
        for line in lines:
            if line.startswith('v '):
                # Remove vertex colors from lines starting with 'v '
                parts = line.split()
                if len(parts) > 4:  # Assume extra values are vertex colors
                    line = ' '.join(parts[:4]) + '\n'
            outfile.write(line)

def poisson_mesh(mesh_path, vtx, normal, color, depth, use_pymeshlab, hhi=False, n_faces=None, smooth_iter=5, add_floor_pc=False):
    print('Poisson meshing')
    if hhi: mesh_path = mesh_path.replace('.ply', '.obj')

    vtx_np = vtx.cpu().numpy()
    if hhi: vtx_np = vtx_np * 1000
    normal_np = normal.cpu().numpy()

    # poisson recon
    if use_pymeshlab or hhi:
        if hhi: 
            # # Step 1: Filter out points where y < floor_cut
            # filtered_indices = vtx_np[:, 1] >= floor_cut
            # vtx_np = vtx_np[filtered_indices]
            # normal_np = normal_np[filtered_indices]
            
            if add_floor_pc:
                # Step 2: Project the bottom points onto the y = floor_cut/min_y plane
                min_y = vtx_np[:, 1].min()
                bottom_points = vtx_np[vtx_np[:, 1] < min_y+5] # 5mm above the floor
                projected_points = bottom_points[:, [0, 2]]  # Only x and z coordinates

                # Step 3A: Find the 2D convex hull
                from scipy.spatial import ConvexHull, Delaunay
                hull = ConvexHull(projected_points)
                hull_vertices = projected_points[hull.vertices]

                # Step 4: Generate a dense grid of points inside the hull
                # Define a grid bounding box
                x_min, x_max = projected_points[:, 0].min(), projected_points[:, 0].max()
                z_min, z_max = projected_points[:, 1].min(), projected_points[:, 1].max()
                # Create a grid of points
                grid_density = 300  # Adjust for more/less dense grid
                x_grid, z_grid = np.meshgrid(
                    np.linspace(x_min, x_max, grid_density),
                    np.linspace(z_min, z_max, grid_density)
                )
                grid_points = np.vstack((x_grid.ravel(), z_grid.ravel())).T

                # 4A: Delaunay + find_simplex
                # Create a Delaunay triangulation of the hull vertices
                tri = Delaunay(hull_vertices)
                # Retain points inside the convex hull
                inside_hull = tri.find_simplex(grid_points) >= 0
                floor_points = grid_points[inside_hull]

                # Step 5: Add the y coordinate and ensure proper xyz order
                floor_points_3d = np.hstack((
                    floor_points[:, [0]], 
                    np.ones((floor_points.shape[0], 1)) * (-5),  # y = -5mm
                    floor_points[:, [1]]
                ))

                # Step 6: Assign normals for the floor points
                floor_normals = np.tile([0, -1, 0], (floor_points_3d.shape[0], 1))  # Normals point to -y direction

                # Step 7: Combine the filtered points and normals with the floor points and normals
                vtx_np = np.vstack((vtx_np, floor_points_3d))
                normal_np = np.vstack((normal_np, floor_normals))
            
            # # save the pcd to ply files
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(vtx_np.astype(np.float64))
            # pcd.normals = o3d.utility.Vector3dVector(normal_np.astype(np.float64))
            # pcd_path = mesh_path.replace('.obj', '_filtered.ply')
            # o3d.io.write_point_cloud(pcd_path, pcd)
            # pcd.points = o3d.utility.Vector3dVector(floor_points_3d.astype(np.float64))
            # pcd.normals = o3d.utility.Vector3dVector(floor_normals.astype(np.float64))
            # pcd_path = mesh_path.replace('.obj', '_floor.ply')
            # o3d.io.write_point_cloud(pcd_path, pcd)

        ms = pymeshlab.MeshSet()
        pts = pymeshlab.Mesh(vtx_np, [], normal_np)
        ms.add_mesh(pts)      
        ms.generate_surface_reconstruction_screened_poisson(depth=depth, threads=os.cpu_count() // 2, preclean=True, samplespernode=15)
        if n_faces: ms.meshing_decimation_quadric_edge_collapse(targetfacenum = n_faces)
        
        # remove connected components having diameter less than p% of the diameter of the entire mesh
        p = pymeshlab.PercentageValue(30)
        ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=p)
                
        ms.save_current_mesh(mesh_path)
        remove_vertex_colors(mesh_path)
        

    else: # use open3d
        color_np = np.ones_like(vtx_np) * 0.8
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx_np.astype(np.float64))
        pcd.normals = o3d.utility.Vector3dVector(normal_np.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(color_np.astype(np.float64))
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth,n_threads=os.cpu_count() // 2)
        
        # o3d.io.write_triangle_mesh(mesh_path, mesh)        
        # print('Mesh cleaning')
        # mesh = o3d.io.read_triangle_mesh(mesh_path)

        # Apply the OBB to the mesh
        center, rotation, min_coords, max_coords = compute_obb(vtx_np)
        mesh_vertices = np.asarray(mesh.vertices)
        vertices_centered = mesh_vertices - center
        vertices_transformed = vertices_centered @ rotation
        buffer = 0 #0.01 
        min_coords = min_coords - buffer
        max_coords = max_coords + buffer    
        # Generate mask for vertices inside the OBB
        is_inside = np.all(
            (vertices_transformed >= min_coords) & 
            (vertices_transformed <= max_coords),
            axis=1
        )
        mesh.remove_vertices_by_mask(~is_inside) 

        # only keep largest cluster
        if True:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            mesh.remove_triangles_by_mask(triangles_to_remove)

        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

        o3d.io.write_triangle_mesh(mesh_path, mesh)
        

def find_min_numbered_subfolder(folder_path):
    # Regular expression to match subfolders with the pattern xxxx_(d+)
    pattern = re.compile(r'_\d+$')
    
    min_number = float('inf')
    selected_subfolder = None

    # Iterate through all subfolders
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):  # Ensure it's a directory
            match = pattern.search(subfolder)
            if match:
                # Extract the number from the matched pattern
                number = int(match.group(0)[1:])  # Remove the '_' and convert to int
                if number < min_number:
                    min_number = number
                    selected_subfolder = subfolder_path

    return selected_subfolder


def get_min_max_subfolder_numbers(directory_path):
    """
    Get the smallest and largest numbers from the subfolder names in the given directory.

    :param directory_path: The path to the directory containing the subfolders.
    :return: A tuple (min_number, max_number) with the smallest and largest numbers found,
             or (None, None) if no valid subfolders are found.
    """
    min_number = None
    max_number = None
    
    # Regex pattern to match folder names ending with a number
    pattern = re.compile(r'_(\d+)')
    
    if not os.path.exists(directory_path):
        return None, None

    # Iterate through the subdirectories in the given directory
    for subdir in os.listdir(directory_path):
        match = pattern.search(subdir)
        if match:
            number = int(match.group(1))
            if min_number is None or number < min_number:
                min_number = number
            if max_number is None or number > max_number:
                max_number = number
    
    return min_number, max_number


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        assert False