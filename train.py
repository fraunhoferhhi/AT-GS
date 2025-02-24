
import time
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, cos_loss 
from gaussian_renderer import render
import numpy as np
import sys
import json
import json5
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, depth2rgb, normal2rgb, depth2normal, masked_psnr, compute_curvature, erode_mask, normal2curv, resample_points, grid_prune
from torchvision.utils import save_image
import torch.nn.functional as F
from utils.debug_utils import save_tensor_img
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import get_min_max_subfolder_numbers, str2bool, poisson_mesh
import re
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from collections import defaultdict
from PIL import Image
from utils.raft_utils import C_RAFT
sys.path.append('submodules')
from RAFT.utils import flow_viz


def training_one_frame(dataset, opt, pipe, load_iteration, testing_iterations, saving_iterations, checkpoint, debug_from):
    start_time=time.time()
    test_res = []
    first_iter = 0
    use_mask = dataset.use_mask
    tb_writer = prepare_per_frame_logger(dataset)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    if opt.iter_s1 > 0: 
        gaussians.training_one_frame_s1_setup(opt) # setup s1
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if args.normals_rendered:           
        with torch.no_grad():     
            views = scene.getTrainCameras(1)
            background4of = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")
            raft_model = C_RAFT()
                        
            # Group images by their shape
            grouped_views = defaultdict(list)
            for view in views:
                image_shape = tuple(args.rgb_chw[f'{view.image_name}'].shape)
                grouped_views[image_shape].append(view)

            for image_shape, group in grouped_views.items():
                if image_shape[2] < 1100:
                    batch_size = 4 #8
                elif image_shape[2] < 1600:
                    batch_size = 2 #4
                else: batch_size = 1 #2
                num_batches = (len(group) + batch_size - 1) // batch_size  # Ensure correct batch calculation

                for i in tqdm(range(num_batches), desc=f"Computing optical flows for shape {image_shape}", mininterval=5.0):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(group))  # Ensure the last batch is correctly indexed
                    batch_views = group[start_idx:end_idx]
                    rgb_batch = torch.stack([view.get_gtImage(background4of, False).cuda() for view in batch_views])
                    normals_batch = torch.stack([args.normals[f'{view.image_name}'] for view in batch_views])
                    image1_batch = torch.stack([args.rgb_chw[f'{view.image_name}'] for view in batch_views])
                    
                    if args.save_snapshot:
                        warped_normals_batch, flow_bwd_batch = raft_model.raft_warp(image1_batch, rgb_batch, normals_batch, args.save_snapshot)
                        for view, warped_normals, flow_bwd in zip(batch_views, warped_normals_batch, flow_bwd_batch):
                            args.warped_normals[f'{view.image_name}'] = warped_normals  # 4HW
                            args.flow_bwd[f'{view.image_name}'] = flow_bwd  # 2HW
                    else:
                        warped_normals_batch = raft_model.raft_warp(image1_batch, rgb_batch, normals_batch, args.save_snapshot)
                        for view, warped_normals in zip(batch_views, warped_normals_batch):
                            args.warped_normals[f'{view.image_name}'] = warped_normals  # 4HW

    opt.densification_interval = max(opt.densification_interval, len(scene.getTrainCameras()))
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iter_s1 + opt.iter_s2), desc="Training progress", mininterval=5.0)
    first_iter += 1
    iter_start_time=time.time()
    
    # Train stage 1 and 2
    total_iter = opt.iter_s1 + opt.iter_s2
    for iteration in range(first_iter, total_iter + 1):        
        iter_start.record()

        if iteration == opt.iter_s1 + 1:
            # switch from s1 to s2
            if opt.iter_s1 > 0: gaussians.update_by_ntc()
            gaussians.training_one_frame_s2_setup(opt)

        if (iteration < opt.iter_s1 + 1): # s1
            gaussians.query_ntc()
        else: #s2
            gaussians.update_learning_rate(iteration - opt.iter_s1)
        
        # increase the levels of SH up to a maximum degree
        # if iteration - opt.iter_s1 == opt.iter_s1 // 10:
        #     gaussians.oneupSHdegree()
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(1).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        background = torch.rand((3), dtype=torch.float32, device="cuda") if dataset.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, normal, depth, opac, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        mask_gt = viewpoint_cam.get_gtMask(use_mask)
        gt_image = viewpoint_cam.get_gtImage(background, use_mask)
        mask_vis = (opac.detach() > 1e-5) # rendered_mask
        normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis
        d2n = depth2normal(depth, mask_vis, viewpoint_cam)
        mono = viewpoint_cam.mono if dataset.mono_normal else None
        if mono is not None:
            mono *= mask_gt
            monoN = mono[:3]

        # Loss
        loss_dict = {}

        Ll1 = l1_loss(image, gt_image)
        loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss_dict["loss_rgb"] = loss_rgb

        bce_loss_func = torch.nn.BCELoss()
        loss_mask = bce_loss_func(opac, mask_gt) * 0.1 # 0.01
        loss_dict["loss_mask"] = loss_mask
              
        loss = 1 * loss_rgb
        loss += 1 * loss_mask
        
        if iteration > opt.iter_s1:            
            ### the same effect with the original
            opac_ = gaussians.get_opacity
            opac_mask = torch.gt(opac_, 0.51) * torch.le(opac_, 0.99)
            opac_ = opac_ - 0.5
            loss_opac = torch.exp(-(opac_ * opac_) * 20)
            loss_opac = (loss_opac * opac_mask).mean()
            loss += loss_opac * 0.01
            loss_dict["loss_opac"] = loss_opac * 0.01

            # depth-normal consistency
            loss_surface = cos_loss(normal, d2n)  
            loss += (0.01 + 0.1 * min(2 * iteration / total_iter, 1)) * loss_surface    
            loss_dict["loss_surface"] = (0.01 + 0.1 * min(2 * iteration / total_iter, 1)) * loss_surface

            # spatial smoothness
            if opt.lambda_smooth > 0:
                curv_n = normal2curv(normal, mask_vis)
                loss_dict["loss_spatial_sm"] = l1_loss(curv_n * 1, 0) * 0.005 * opt.lambda_smooth
                loss += l1_loss(curv_n * 1, 0) * 0.005 * opt.lambda_smooth

            if mono is not None:
                loss_monoN = cos_loss(normal, monoN, weight=mask_gt)
                loss += (0.04 - ((iteration / total_iter)) * 0.03) * loss_monoN
                loss_dict["loss_monoN"] = (0.04 - ((iteration / total_iter)) * 0.03) * loss_monoN
            
            ## normal coherence
            if args.normals_rendered:
                curv_rendered = compute_curvature(normal) # 1HW
                curv_warped = compute_curvature(torch.nn.functional.normalize(args.warped_normals[f'{viewpoint_cam.image_name}'][0:3], dim=0))
                mask_vis = (opac.detach() > 1e-1) #1e-5
                mask_n = mask_gt*mask_vis # in [0, 1.0], 1HW
                mask_n = erode_mask(mask_n.float(), 9) # 1HW
                mask_n = (mask_n * args.warped_normals[f'{viewpoint_cam.image_name}'][3:]).detach()
                loss_n_coher = F.mse_loss(curv_rendered[mask_n>0], curv_warped[mask_n>0]) * args.l_coh
                loss += (0.04 - ((iteration / total_iter)) * 0.02) * loss_n_coher
                loss_dict["loss_n_coher"] = (0.04 - ((iteration / total_iter)) * 0.02) * loss_n_coher
                
        loss_dict["total_loss"] = loss        
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss_rgb.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}, Pts={len(gaussians._xyz)}"})
                progress_bar.update(10)
            if iteration == total_iter:
                progress_bar.close()

            # Log and save
            test_background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
            res = None
            if tb_writer:
                res = training_report(tb_writer, iteration, loss_dict, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, pipe, test_background, use_mask)
            if res is not None:
                test_res.append(res)
                if (iteration == 1) and (opt.iter_s1 == 0):
                    for _ in range(3): test_res.append(res)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration=iteration, save_type='all')
                                
            if iteration > opt.iter_s1: # s2
                # Densification: prune -> densify -> reset_opacity
                if iteration - opt.iter_s1 < opt.densify_until_iter and iteration - opt.iter_s1 > opt.densify_from_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if (iteration - opt.iter_s1) % opt.densification_interval == 0:
                        min_opac = 0.1
                        gaussians.adaptive_prune(min_opac, scene.cameras_extent)
                        gaussians.adaptive_densify(opt.densify_grad_threshold, scene.cameras_extent, True)
                    
                    if (iteration - opt.iter_s1 - 1) % opt.opacity_reset_interval == 0 and opt.opacity_lr > 0:
                        gaussians.reset_opacity(0.12)

            if (iteration - 1) % 200 == 0 and False:                
                normal_wrt = normal2rgb(normal, mask_vis)
                depth_wrt = depth2rgb(depth, mask_vis)
                img_wrt = torch.cat([gt_image, image, normal_wrt * opac, depth_wrt * opac], 2)
                os.makedirs(os.path.join(args.output_path, f'training_output'), exist_ok=True)
                save_image(img_wrt.cpu(), os.path.join(args.output_path, f'training_output/{iteration-1}.png'))

            # Optimizer step
            if iteration < opt.iter_s1 + 1:
                # s1
                gaussians.ntc_optimizer.step()
                gaussians.ntc_optimizer.zero_grad()
            else: # iteration < opt.iter_s1 + opt.iter_s2 + 1:
                # s2
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad()

    iter_end_time=time.time()
    
    pre_time = iter_start_time - start_time
    frame_training_time = iter_end_time - start_time # - iter_start_time

    if args.optical_flow_normals:
        with torch.no_grad():
            views = scene.getTrainCameras(1)
            background4output = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")
            # loop through all training cams
            for idx, view in enumerate(tqdm(views, desc="Rendering normals", mininterval=5.0)):
                render_pkg = render(view, gaussians, pipe, background4output)
                normal = render_pkg["normal"]
                opac = render_pkg["opac"]
                mask_gt = view.get_gtMask(use_mask) > 0
                mask_vis = (opac.detach() > 1e-1) #1e-5
                mask = mask_vis * mask_gt
                normal = torch.nn.functional.normalize(normal, dim=0) * mask
                args.rgb_chw[f'{view.image_name}'] = view.get_gtImage(background4output, False).cuda()
                args.normals[f'{view.image_name}'] = normal
                if args.save_snapshot:
                    # save rendered normals
                    normal_wrt = normal2rgb(normal, mask, background4output) # normal does not change, no need to clone
                    os.makedirs(os.path.join(args.output_path, f'rendered_normals'), exist_ok=True)
                    save_image(normal_wrt.cpu(), os.path.join(args.output_path, f'rendered_normals/{view.image_name}.png'))
                
                    if len(args.warped_normals) > 0:
                        # save warped normals
                        warped_normals = args.warped_normals[f'{view.image_name}'] # 4HW
                        mask_warp = warped_normals[3:]
                        normal_warped = torch.nn.functional.normalize(warped_normals[0:3], dim=0) * mask_warp * mask  # 3HW
                        normal_wrt = normal2rgb(normal_warped, mask_warp * mask, background4output)  # normal does not change, no need to clone
                        os.makedirs(os.path.join(args.output_path, f'warped_normals'), exist_ok=True)
                        save_image(normal_wrt.cpu(), os.path.join(args.output_path, f'warped_normals/{view.image_name}.png'))
                        
                        # save flow    
                        flow_bwd = args.flow_bwd[f'{view.image_name}'] * mask_warp * mask # 2HW
                        os.makedirs(os.path.join(args.output_path, f'flow_bwd'), exist_ok=True)
                        Image.fromarray(flow_viz.flow_to_image(flow_bwd.cpu().numpy().transpose(1, 2, 0))).save(os.path.join(args.output_path, f'flow_bwd/{view.image_name}.png'))

                        # save opac
                        os.makedirs(os.path.join(args.output_path, f'rendered_opac'), exist_ok=True)
                        save_image(opac.cpu(), os.path.join(args.output_path, f'rendered_opac/{view.image_name}.png'))
                        
                        # save gt mask
                        os.makedirs(os.path.join(args.output_path, f'gt_mask'), exist_ok=True)
                        save_image(mask_gt.float().cpu(), os.path.join(args.output_path, f'gt_mask/{view.image_name}.png'))
                        
                        # save rendered img
                        os.makedirs(os.path.join(args.output_path, f'rendered_rgb'), exist_ok=True)
                        save_image(render_pkg["render"].cpu(), os.path.join(args.output_path, f'rendered_rgb/{view.image_name}.png'))
                        
                        # save gt img
                        os.makedirs(os.path.join(args.output_path, f'gt_rgb'), exist_ok=True)
                        save_image(view.get_gtImage(background4output,True).cpu(), os.path.join(args.output_path, f'gt_rgb/{view.image_name}.png'))
                        
                        # save rendered curvature
                        import matplotlib.pyplot as plt
                        from matplotlib.colors import LinearSegmentedColormap
                        original_cmap = plt.cm.viridis
                        colors = original_cmap(np.linspace(0, 1, 256))
                        colors[0] = [1, 1, 1, 1]  # RGBA for white
                        custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', colors)
                        curv_rendered = compute_curvature(normal) # 1HW
                        mask_eroded = erode_mask(mask.float(), 9)
                        # save_image(mask.float().cpu(), os.path.join(args.output_path, f'gt_mask/mask{view.image_name}.png'))
                        # save_image(mask_eroded.cpu(), os.path.join(args.output_path, f'gt_mask/mask_eroded{view.image_name}.png'))
                        curv_rendered = curv_rendered * mask_eroded
                        array = curv_rendered.squeeze(0).cpu().numpy()
                        fig, ax = plt.subplots(figsize=(array.shape[1] / 100, array.shape[0] / 100), dpi=100)
                        cax = ax.imshow(array, cmap=custom_cmap)
                        ax.axis('off')
                        os.makedirs(os.path.join(args.output_path, f'rendered_curv'), exist_ok=True)
                        plt.savefig(os.path.join(args.output_path, f'rendered_curv/{view.image_name}.png'), bbox_inches='tight', pad_inches=0, dpi=100)
                        plt.close()
                        
                        # save warped curvature
                        curv_warped = compute_curvature(normal_warped) # 1HW
                        curv_warped = curv_warped * mask_eroded
                        array = curv_warped.squeeze(0).cpu().numpy()
                        fig, ax = plt.subplots(figsize=(array.shape[1] / 100, array.shape[0] / 100), dpi=100)
                        cax = ax.imshow(array, cmap=custom_cmap)
                        ax.axis('off')
                        os.makedirs(os.path.join(args.output_path, f'warped_curv'), exist_ok=True)
                        plt.savefig(os.path.join(args.output_path, f'warped_curv/{view.image_name}.png'), bbox_inches='tight', pad_inches=0, dpi=100)
                        plt.close()
                        
                        # save rendered depth
                        depth = render_pkg["depth"]
                        depth_wrt = depth2rgb(depth, mask, background4output)
                        os.makedirs(os.path.join(args.output_path, f'rendered_depths'), exist_ok=True)
                        save_image(depth_wrt.cpu(), os.path.join(args.output_path, f'rendered_depths/{view.image_name}.png'))

            args.normals_rendered = True

    if args.output_mesh:
        with torch.no_grad():
            poisson_depth = 9
            grid_dim = 512 
            occ_grid, grid_shift, grid_scale, grid_dim = gaussians.to_occ_grid(0.0, grid_dim, None)

            resampled = []
            # loop through all cams
            for idx, view in enumerate(tqdm(scene.getTrainCameras(1), desc="Rendering progress", mininterval=5.0)):
                render_pkg = render(view, gaussians, pipe, background)
                image, normal, depth, opac, viewspace_point_tensor, visibility_filter, radii = \
                    render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
                    render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                
                mask_gt = view.get_gtMask(use_mask)
                gt_image = view.get_gtImage(background, use_mask).cuda()
                mask_vis = (opac.detach() > 1e-1) #1e-5
                depth_range = [0, 20]
                mask_clip = (depth > depth_range[0]) * (depth < depth_range[1])
                normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis

                # unproject filtered depth map to 3D points in world space
                # [H, W, 9(xyz_in_world, normals, rgb)]
                pts = resample_points(view, depth, normal, image, mask_vis * mask_gt * mask_clip)
                # prune points by occupancy grid
                grid_mask = grid_prune(occ_grid, grid_shift, grid_scale, grid_dim, pts[..., :3], thrsh=0)
                pts = pts[grid_mask]
                resampled.append(pts.cpu())

            resampled = torch.cat(resampled, 0)
            mesh_dir = os.path.join(args.output_path, "..", "meshes")
            os.makedirs(mesh_dir, exist_ok=True)
            mesh_path = os.path.join(mesh_dir, f"Frame_{args.frame_id:06d}.ply")
            use_pymeshlab = True
            poisson_mesh(mesh_path, resampled[:, :3], resampled[:, 3:6], resampled[:, 6:], poisson_depth, use_pymeshlab, args.hhi, args.n_faces, args.add_floor_pc)
        
    return test_res, pre_time, frame_training_time

# per-frame logger
def prepare_per_frame_logger(args):
    tb_writer = None
    if TENSORBOARD_FOUND and args.eval:
        print("per-frame output folder: {}".format(args.output_path))
        os.makedirs(args.output_path, exist_ok = True)
        tb_writer = SummaryWriter(args.output_path)
    else:
        print("Not logging progress")
    return tb_writer

def prepare_global_logger(output_global_path, args):   
    tb_writer = None
    if TENSORBOARD_FOUND and args.eval: 
        print("Global Output folder for all frames: {}".format(output_global_path))
        os.makedirs(output_global_path, exist_ok = True)
        tb_writer = SummaryWriter(output_global_path)
    else:
        print("Not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss_dict, l1_loss, elapsed, testing_iterations, scene : Scene, pipe, bg, use_mask):
    for loss_name, loss_value in loss_dict.items():
        tb_writer.add_scalar(f'train_loss_patches/{loss_name}', loss_value.item(), iteration)
    tb_writer.add_scalar('iter_time', elapsed, iteration)
    tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, )
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                masked_psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(render(viewpoint, scene.gaussians, pipe, bg)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.get_gtImage(bg, with_mask=use_mask), 0.0, 1.0)
                    if idx < 5:
                        tb_writer.add_image(config['name'] + "_view_{}/render".format(viewpoint.image_name), image, global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_image(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image, global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    masked_psnr_test += masked_psnr(image, gt_image)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])  
                masked_psnr_test /= len(config['cameras'])        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                tb_writer.add_scalar(config['name'] + '/l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + '/psnr', psnr_test, iteration)
                tb_writer.add_scalar(config['name'] + '/masked_psnr', masked_psnr_test, iteration)
                if config['name'] == 'test':
                    avg_test_psnr = psnr_test
                    rendering_of_last_test_cam = image

        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()
        
        return {'avg_test_psnr':avg_test_psnr.cpu().numpy()
                , 'rendering_of_last_test_cam':rendering_of_last_test_cam.cpu()
                , 'points_num':scene.gaussians.get_xyz.shape[0]
                }

def train_one_frame(lp,op,pp,args):
    print("Optimizing " + args.output_path)
    res_dict={}
    ress, pre_time, frame_training_time = training_one_frame(lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from)
    print("\nTraining complete.")
    print(f"Preparation: {pre_time}")
    print(f"frame_training_time: {frame_training_time}")
    
    if ress !=[]:
        for idx, res in enumerate(ress):
            if False:
                save_tensor_img(res['rendering_of_last_test_cam'],os.path.join(args.output_path,f'rendering_test{idx}'))
            res_dict[f'psnr_{idx}']=res['avg_test_psnr']
            res_dict[f'points_num_{idx}']=res['points_num']
        res_dict[f'time']=frame_training_time
    return res_dict 

def train_frames(lp, op, pp, args):
    safe_state(args.quiet)
    output_path=args.output_path # global
    source_path=args.source_path # global
    sub_paths = os.listdir(source_path)
    tb_global = prepare_global_logger(args.output_global_path, lp.extract(args))
    pattern = re.compile(r'frame_(\d+)')
    dict_frame_dirs = {}
    for frame_dir in os.listdir(source_path): 
        if pattern.match(frame_dir):
            dict_frame_dirs[int(pattern.match(frame_dir).group(1))] = frame_dir
            
    print(f"Training from frame {args.frame_start}", f" to frame {args.frame_end-1}")
    for frame_id in range(args.frame_start+1, args.frame_end):
        print(f"Training frame {frame_id}")
        args.frame_id = frame_id
        start_time = time.time()
        args.source_path = os.path.join(source_path, dict_frame_dirs[frame_id]) # per-frame
        args.output_path = os.path.join(output_path, dict_frame_dirs[frame_id]) # per-frame
        args.model_path = os.path.join(output_path, dict_frame_dirs[frame_id-1]) # per-frame
        res_dict = train_one_frame(lp,op,pp,args)
        print(f"Frame {frame_id} finished in {time.time()-start_time} seconds.")
        if tb_global:
            tb_global.add_scalar('psnr_0', res_dict['psnr_0'], frame_id)
            tb_global.add_scalar('psnr_1', res_dict['psnr_1'], frame_id)
            tb_global.add_scalar('psnr_2', res_dict['psnr_2'], frame_id)
            tb_global.add_scalar('time', res_dict['time'], frame_id)
            tb_global.add_scalar('points_num', res_dict['points_num_2'], frame_id)
        torch.cuda.empty_cache()
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--frame_start', type=int, default=1)
    parser.add_argument('--frame_end', type=int, default=150)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument('--load_iteration', type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--config_path", type=str, default = None)
    parser.add_argument("--optical_flow_normals", type=str, default=True)
    parser.add_argument('--l_coh', type=float, default=1.0)
    parser.add_argument("--save_snapshot", action="store_true") 
    parser.add_argument("--output_mesh", type=str, default="False")
    parser.add_argument("--hhi", type=str, default="False")
    parser.add_argument("--n_faces", type=int, default=None)
    parser.add_argument("--add_floor_pc", type=str, default="False")
    args = parser.parse_args(sys.argv[1:])

    if args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config = json5.load(f)
        for key, value in config.items():
            setattr(args, key, value)
    args.optical_flow_normals = str2bool(args.optical_flow_normals)
    args.mono_normal = str2bool(args.mono_normal)
    args.output_mesh = str2bool(args.output_mesh)
    args.hhi = str2bool(args.hhi)
    args.add_floor_pc = str2bool(args.add_floor_pc)
    
    # resume training
    _, frame_done = get_min_max_subfolder_numbers(config["output_path"])
    if frame_done:
        if (frame_done == config["frame_end"] - 1): exit()
        args.frame_start = max(frame_done -1, args.frame_start)

    # set other parameters:
    if args.output_global_path == '':
        args.output_global_path = args.output_path.replace('output', 'output_global').replace('recon_mesh', 'recon_mesh_global')
    if len(args.test_iterations) == 0:
        # iterations are 1-based
        # args.test_iterations = [1, args.iter_s1//3, args.iter_s1//3*2, args.iter_s1, args.iter_s1 + args.iter_s2//3, args.iter_s1 + args.iter_s2//3*2, args.iter_s1 + args.iter_s2]
        args.test_iterations = [1, args.iter_s1, args.iter_s1 + args.iter_s2]
    if len(args.save_iterations) == 0:
        # args.save_iterations = [args.iter_s1, args.iter_s1 + args.iter_s2]
        args.save_iterations = [args.iter_s1 + args.iter_s2]
        
    # lr
    args.position_lr_max_steps = args.iter_s2   
    args.position_lr_init *= args.lr_scale
    args.position_lr_final *= args.lr_scale
    args.feature_lr *= args.lr_scale
    args.opacity_lr *= args.lr_scale
    args.scaling_lr *= args.lr_scale
    args.rotation_lr *= args.lr_scale

    # densify
    args.densify_from_iter = 30
    args.densify_until_iter = int(args.iter_s2 / 2)
    args.densification_interval = 30 # opt.densification_interval = max(opt.densification_interval, len(scene.getTrainCameras()))
    args.opacity_reset_interval = int(args.iter_s2 / 3.9) 

    args.normals_rendered = False
    args.normals = {}
    args.rgb_chw = {}
    args.warped_normals = {}
    args.flow_bwd = {}

    os.makedirs(args.output_global_path, exist_ok = True)
    train_frames(lp,op,pp,args)
    print("\nTraining complete.")
