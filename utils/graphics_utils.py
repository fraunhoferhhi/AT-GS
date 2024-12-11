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
import math
import numpy as np
from typing import NamedTuple
from utils.general_utils import quaternion2rotmat

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(q, t, translate=0.0, scale=1.0):
    Rt = torch.zeros((4, 4)).to(q.device).to(torch.float32)
    R = quaternion2rotmat(torch.nn.functional.normalize(q[None]))[0]
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


######## from eye/view/cam space to clip space
def getProjectionMatrix(znear, zfar, fovX, fovY, W, H, prcp): 
     fx = fov2focal(fovX, W)
     fy = fov2focal(fovY, H)
     cx = prcp[0] * W 
     cy = prcp[1] * H
     top = znear * cy / fy 
     bottom = -znear * (H - cy) / fy 
     right = znear * (W - cx) / fx 
     left = -znear * cx / fx 

     P = torch.zeros(4, 4) 
     z_sign = 1.0 

     P[0, 0] = 2.0 * znear / (right - left) 
     P[1, 1] = 2.0 * znear / (top - bottom) 
     P[0, 2] = -(right + left) / (right - left) 
     P[1, 2] = (top + bottom) / (top - bottom) 
     P[3, 2] = z_sign 
     P[2, 2] = z_sign * zfar / (zfar - znear) 
     P[2, 3] = -(zfar * znear) / (zfar - znear) 

     return P

# only used for rendering mesh video
def getProjectionMatrix2(znear, zfar, fx, fy, W, H, prcp): 
    #  fx = fov2focal(fovX, W)
    #  fy = fov2focal(fovY, H)
     cx = prcp[0] * W 
     cy = prcp[1] * H
    #  breakpoint()
     top = znear * cy / fy 
     bottom = -znear * (H - cy) / fy 
     right = znear * (W - cx) / fx 
     left = -znear * cx / fx 

     P = torch.zeros(4, 4) 
     z_sign = 1.0 

     P[0, 0] = 2.0 * znear / (right - left) 
     P[1, 1] = 2.0 * znear / (top - bottom) 
     P[0, 2] = -(right + left) / (right - left) 
     P[1, 2] = (top + bottom) / (top - bottom) 
     P[3, 2] = z_sign 
     P[2, 2] = z_sign * zfar / (zfar - znear) 
     P[2, 3] = -(zfar * znear) / (zfar - znear) 

     return P

# the input fov may not be real fov. but it is fine if it is from focal2fov()
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

# only called from dataset_readers.py. For non-centered cam, the output fov is not real fov. 
# But it is fine as long as the output fov is only used to get the focal_length back using fov2focal()
# the output fov is also used in getProjectionMatrix()
def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))