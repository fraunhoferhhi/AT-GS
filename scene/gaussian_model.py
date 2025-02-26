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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_symmetric, build_scaling_rotation, build_rotation, quaternion_multiply, normal2rotation
from torch import nn
import os
from torch.utils.cpp_extension import load
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.general_utils import quaternion2rotmat
from utils.graphics_utils import BasicPointCloud
import tinycudann as tcnn
from ntc import NeuralTransformationCache
import json5

class GaussianModel:

    def setup_functions(self):
        
        # @torch.compile
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
                
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.rotation_compose = quaternion_multiply
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        self._xyz_bound_min = None
        self._xyz_bound_max = None
        
        self._d_xyz = None
        self._d_rot = None
        self._d_rot_matrix = None
        self._d_scaling = None
        self._d_opacity = None
        
        # for deformed points
        self._new_xyz = None
        self._new_rot = None
        self._new_scaling = None
        self._new_opacity = None
        self._new_feature = None
            
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.scale_gradient_accum = torch.empty(0)
        self.opac_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # will have 4th param.: self.gaussians.config.append(camera_lr > 0)
        try:
            self.config = [args.surface, args.normalize_depth, args.perpix_depth]
        except AttributeError:
            self.config = [True, True, True]
        self.setup_functions()
        self.utils_mod = load(name="cuda_utils", sources=["utils/ext.cpp", "utils/cuda_utils.cu"])

        self.unified_densification = args.unified_densification

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.scale_gradient_accum,
            self.opac_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.config
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        scale_gradient_accum,
        opac_accum,
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.config) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.scale_gradient_accum = scale_gradient_accum
        self.opac_accum = opac_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self._new_scaling is not None:
            return self._new_scaling
        else:
            return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self._new_rot is not None:
            return self._new_rot # no need for rotation_activation, because it is already normalized
        else:
            return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        if self._new_xyz is not None:
            return self._new_xyz
        else:
            return self._xyz
    
    @property
    def get_features(self):
        if self._new_feature is not None:
            return self._new_feature
        else:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)  
          
    @property
    def get_opacity(self):
        if self._new_opacity is not None:
            return self._new_opacity
        else:
            return self.opacity_activation(self._opacity)
        
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    @property
    def get_normal(self):
        return quaternion2rotmat(self.get_rotation)[..., 2]
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2 / 4))[...,None].repeat(1, 3)
        
        if self.config[0] > 0: # args.surface
            if np.abs(np.sum(pcd.normals)) < 1: # go this way because pcd.normals are all 0
                # duplicate the points to 4 times with random normals when normals are unknown
                dup = 4
                fused_point_cloud = torch.cat([fused_point_cloud for _ in range(dup)], 0)
                fused_color = torch.cat([fused_color for _ in range(dup)], 0)
                scales = torch.cat([scales for _ in range(dup)], 0)
                normals = np.random.rand(len(fused_point_cloud), 3) - 0.5
                normals /= np.linalg.norm(normals, 2, 1, True)
            else:
                normals = pcd.normals

            rots = normal2rotation(torch.from_numpy(normals).to(torch.float32)).to("cuda")
            scales[..., -1] -= 1e10 # squeeze z scaling
        else:
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def training_one_frame_s1_setup(self,training_args):
        ntc_conf_path=training_args.ntc_conf_path
        with open(ntc_conf_path) as ntc_conf_file:
            ntc_conf = json5.load(ntc_conf_file)
        if training_args.only_mlp:
            model=tcnn.Network(n_input_dims=3, n_output_dims=8, network_config=ntc_conf["network"]).to(torch.device("cuda"))
        else:
            model=tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=8, encoding_config=ntc_conf["encoding"], network_config=ntc_conf["network"]).to(torch.device("cuda"))
        self.ntc=NeuralTransformationCache(model,self.get_xyz_bound()[0],self.get_xyz_bound()[1])
        self.ntc.load_state_dict(torch.load(training_args.ntc_path))
        self._xyz_bound_min = self.ntc.xyz_bound_min
        self._xyz_bound_max = self.ntc.xyz_bound_max
        if training_args.ntc_lr is not None:
            ntc_lr=training_args.ntc_lr # used
        else:
            ntc_lr=ntc_conf["optimizer"]["learning_rate"]
        self.ntc_optimizer = torch.optim.Adam(self.ntc.parameters(), lr=ntc_lr)            
       

    def training_one_frame_s2_setup(self, training_args):
    # def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opac_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.config[3] = False # training_args.camera_lr > 0
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, save_type='all'):
        mkdir_p(os.path.dirname(path))
        if save_type=='origin':  
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()
        elif save_type=='all':
            xyz = self.get_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self.get_features[:,0:1,:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self.get_features[:,1:,:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self.inverse_opacity_activation(self.get_opacity).detach().cpu().numpy()
            scale = self.scaling_inverse_activation(self.get_scaling).detach().cpu().numpy()
            rotation = self.get_rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]  

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, ratio):
        # opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * ratio))
        opacities_new = inverse_sigmoid(self.get_opacity * ratio)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, spatial_lr_scale=0):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree
        # self.active_sh_degree = self.max_sh_degree - 1

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.scale_gradient_accum = self.scale_gradient_accum[valid_points_mask]
        self.opac_accum = self.opac_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        torch.cuda.empty_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opac_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, pre_mask=True, keep_original=False):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda") # only useful when there is clone before
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
                
        selected_pts_mask *= pre_mask
        
        # # split if the gaussian size is too big. does not help
        # scale_mask = torch.max(self.get_scaling, dim=1).values > 0.01 # 1cm
        # selected_pts_mask = torch.logical_or(selected_pts_mask, scale_mask)

        if keep_original: N -= 1

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        if self.config[0] > 0:
            new_scaling[:, -1] = -1e10
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        if not keep_original:
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, pre_mask=True):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        selected_pts_mask *= pre_mask
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
    
    def densify_unified(self, grads, grad_threshold, scene_extent, N=2, pre_mask=True, keep_original=True):
        assert N==2      
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
                
        selected_pts_mask *= pre_mask

        if keep_original: N -= 1

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
  
        grads_selected = grads[selected_pts_mask] #n1
        grads_mean = grads_selected.mean()
        grads_selected = torch.clamp(grads_selected, max=grads_mean*2)
        # normalize to [0,1]
        stds_scales = grads_selected / (grads_mean*2)
        stds = stds * stds_scales # n3
        
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds) # sample xyz independently
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        if not keep_original:
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.prune_points(prune_filter)
    


    def adaptive_prune(self, min_opacity, extent):
        n_ori = len(self._xyz)
        prune_opac = (self.get_opacity < min_opacity).squeeze()

        scale_min = self.get_scaling[:, :2].min(1).values
        scale_max = self.get_scaling[:, :2].max(1).values
        prune_scale = scale_max > 0.5 * extent
        prune_scale += (scale_min * scale_max) < (1e-8 * extent**2) # no impact
        
        prune_vis = (self.denom == 0).squeeze()
        prune = prune_opac + prune_vis + prune_scale
        self.prune_points(prune)
        # print(f'prune by: opac:{prune_opac.sum()}, scale:{prune_scale.sum()}, vis:{prune_vis.sum()}; extent:{extent}')
        # print(f'prune: {n_ori}-->{len(self._xyz)}')

    def adaptive_densify(self, max_grad, extent, unified_densification=False):
        grad_pos = self.xyz_gradient_accum / self.denom
        grad_scale = self.scale_gradient_accum /self.denom
        running_avg_opac = self.opac_accum /self.denom
        grad_pos[grad_pos.isnan()] = 0.0
        grad_scale[grad_scale.isnan()] = 0.0
        running_avg_opac[running_avg_opac.isnan()] = 0.0
                
        larger = torch.le(grad_scale, 1e-7)[:, 0]  
        denser = torch.le(running_avg_opac, 2)[:, 0] 
        pre_mask = denser * larger
        
        if unified_densification:
            self.percent_dense = 0
            self.densify_unified(grad_pos, max_grad, extent, pre_mask=pre_mask)
        else:
            self.densify_and_clone(grad_pos, max_grad, extent, pre_mask=pre_mask)
            self.densify_and_split(grad_pos, max_grad, extent)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # gradients of projected 2d gausssian
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # gradients of 3d scene gausssian
        self.scale_gradient_accum[update_filter] += self._scaling.grad[update_filter, :2].sum(1, True)
        self.opac_accum[update_filter] += self._opacity[update_filter] # wrong? forget grad?
        self.denom[update_filter] += 1
    
    def query_ntc(self):
        mask, self._d_xyz, self._d_rot = self.ntc(self._xyz)
        
        self._new_xyz = self._d_xyz + self._xyz
        self._new_rot = self.rotation_compose(self._rotation, self._d_rot)
        
    def update_by_ntc(self):
        self._xyz = self.get_xyz.detach().requires_grad_(True)
        self._features_dc = self.get_features[:,0:1,:].detach().requires_grad_(True)
        self._features_rest = self.get_features[:,1:,:].detach().requires_grad_(True)
        self._opacity = self._opacity.detach().requires_grad_(True)
        self._scaling = self._scaling.detach().requires_grad_(True)
        self._rotation = self.get_rotation.detach().requires_grad_(True)
        
        self._d_xyz = None
        self._d_rot = None
        self._d_rot_matrix = None
        self._d_scaling = None
        self._d_opacity = None
        
        self._new_xyz = None
        self._new_rot = None
        self._new_scaling = None
        self._new_opacity = None
        self._new_feature = None
                    
    def get_xyz_bound(self, percentile=86.6):
        with torch.no_grad():
            if self._xyz_bound_min is None:
                half_percentile = (100 - percentile) / 200
                self._xyz_bound_min = torch.quantile(self._xyz,half_percentile,dim=0)
                self._xyz_bound_max = torch.quantile(self._xyz,1 - half_percentile,dim=0)
            return self._xyz_bound_min, self._xyz_bound_max

    def to_occ_grid(self, cutoff, grid_dim_max=512, bound_overwrite=None):
        if bound_overwrite is None:
            xyz_min = self._xyz.min(0)[0]
            xyz_max = self._xyz.max(0)[0]
            xyz_len = xyz_max - xyz_min
            xyz_min -= xyz_len * 0.1
            xyz_max += xyz_len * 0.1
        else:
            xyz_min, xyz_max = bound_overwrite
        xyz_len = xyz_max - xyz_min

        # print(xyz_min, xyz_max, xyz_len)
        
        # voxel size
        grid_len = xyz_len.max() / grid_dim_max
        # necessary grid size, smaller than [512,512,512]
        grid_dim = (xyz_len / grid_len + 0.5).to(torch.int32)

        grid = self.utils_mod.gaussian2occgrid(xyz_min, xyz_max, grid_len, grid_dim,
                                               self.get_xyz, self.get_rotation, self.get_scaling, self.get_opacity,
                                               torch.tensor([cutoff]).to(torch.float32).cuda())
        
        return grid, -xyz_min, 1 / grid_len, grid_dim