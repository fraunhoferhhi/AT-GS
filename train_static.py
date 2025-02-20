## adapted from gaussian surfels

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, cos_loss
from gaussian_renderer import render
import numpy as np
import sys
import json5
from scene import Scene, GaussianModel
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, depth2rgb, normal2rgb, depth2normal, masked_psnr, resize_image, normal2curv, resample_points, grid_prune
from torchvision.utils import save_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import find_min_numbered_subfolder, poisson_mesh, str2bool
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, shuffle=False, resolution_scales=[1]) #[1, 2, 4])
    use_mask = dataset.use_mask
    gaussians.training_one_frame_s2_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif use_mask: 
        # gaussians.mask_prune(scene.getTrainCameras(), 4)
        None

    opt.densification_interval = max(opt.densification_interval, len(scene.getTrainCameras()))

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", mininterval=5.0)
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 2):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        scale = 1

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale).copy()[:]
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
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
        Ll1 = l1_loss(image, gt_image)
        loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        bce_loss_func = torch.nn.BCELoss()
        loss_mask = bce_loss_func(opac, mask_gt)
        
        # depth-normal consistency
        loss_surface = cos_loss(normal, d2n)
        
        opac_ = gaussians.get_opacity
        opac_mask = torch.gt(opac_, 0.51) * torch.le(opac_, 0.99)
        opac_ = opac_ - 0.5
        loss_opac = torch.exp(-(opac_ * opac_) * 20)
        loss_opac = (loss_opac * opac_mask).mean()

        loss = 1 * loss_rgb
        loss += 0.1 * loss_mask
        loss += 0.01 * loss_opac
        loss += (0.01 + 0.1 * min(2 * iteration / opt.iterations, 1)) * loss_surface   

        loss_dict = {
            "total_loss": loss,
            "loss_rgb": loss_rgb,
            "loss_mask": loss_mask,
            "loss_opac": loss_opac * 0.01,
            "loss_surface": (0.01 + 0.1 * min(2 * iteration / opt.iterations, 1)) * loss_surface
        }

        if opt.lambda_smooth > 0:
            curv_n = normal2curv(normal, mask_vis)
            loss_curv = l1_loss(curv_n * 1, 0) * 0.005 * opt.lambda_smooth
            loss += loss_curv

        if mono is not None:
            loss_monoN = cos_loss(normal, monoN, weight=mask_gt)
            loss += (0.04 - ((iteration / opt.iterations)) * 0.02) * loss_monoN

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss_rgb.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}, Pts={len(gaussians._xyz)}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            test_background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
            training_report(tb_writer, iteration, loss_dict, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, pipe, test_background, use_mask)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration % opt.densification_interval == 0:
                    min_opac = 0.1
                    gaussians.adaptive_prune(min_opac, scene.cameras_extent)
                    gaussians.adaptive_densify(opt.densify_grad_threshold, scene.cameras_extent)
                
                if (iteration - 1) % opt.opacity_reset_interval == 0 and opt.opacity_lr > 0:
                    gaussians.reset_opacity(0.12)

            if (iteration - 1) % 1000 == 0:                
                normal_wrt = normal2rgb(normal, mask_vis)
                depth_wrt = depth2rgb(depth, mask_vis)
                img_wrt = torch.cat([gt_image, image, normal_wrt * opac, depth_wrt * opac], 2)
                os.makedirs(os.path.join(args.output_path, f'training_output'), exist_ok=True)
                save_image(img_wrt.cpu(), os.path.join(args.output_path, f'training_output/{iteration-1}.png'))
            
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.output_path + "/chkpnt" + str(iteration) + ".pth")
    
    if args.output_mesh:    
# dataset.model_path, True, "train", scene.loaded_iter, scene.getTrainCameras(scales[0]), gaussians, pipeline, background, write_image, poisson_depth, frame_id
# def create_mesh(model_path, use_mask, name, iteration, views, gaussians, pipeline, background, write_image, poisson_depth, frame_id):
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
            mesh_path = os.path.join(mesh_dir, f"Frame_{args.frame_start:06d}.ply")
            use_pymeshlab = True
            poisson_mesh(mesh_path, resampled[:, :3], resampled[:, 3:6], resampled[:, 6:], poisson_depth, use_pymeshlab, args.hhi, args.n_faces, args.add_floor_pc)


def prepare_output_and_logger(args):    
    # Set up output folder
    print("Output folder: {}".format(args.output_path))
    os.makedirs(args.output_path, exist_ok = True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.output_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss_dict, l1_loss, elapsed, testing_iterations, scene : Scene, pipe, bg, use_mask):
    if tb_writer:
        for loss_name, loss_value in loss_dict.items():
            tb_writer.add_scalar(f'train_loss_patches/{loss_name}', loss_value.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()[::8]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                masked_psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(render(viewpoint, scene.gaussians, pipe, bg)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.get_gtImage(bg, with_mask=use_mask), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    masked_psnr_test += masked_psnr(image, gt_image)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])  
                masked_psnr_test /= len(config['cameras'])        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/masked_psnr', masked_psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            
            # log gradient scales
            scale_min = scene.gaussians.get_scaling[:,:2].min().mean().item()
            scale_max = scene.gaussians.get_scaling[:,:2].max().mean().item()
            axis_3 = scene.gaussians.get_scaling[:,2].mean().item()
            tb_writer.add_scalar('scale/min_avg', scale_min, iteration)
            tb_writer.add_scalar('scale/max_avg', scale_max, iteration)
            tb_writer.add_scalar('scale/3rd_axis_avg', axis_3, iteration)
            
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[15_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--config_path", type=str, default = None)
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
    args.mono_normal = str2bool(args.mono_normal)
    args.output_mesh = str2bool(args.output_mesh)
    args.hhi = str2bool(args.hhi)
    args.add_floor_pc = str2bool(args.add_floor_pc)

    args.save_iterations.append(args.iterations)
    args.source_path = os.path.join(args.source_path, f'frame_{args.frame_start}')
    args.output_path = os.path.join(args.output_path, os.path.basename(args.source_path))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    
    print("\nTraining complete.")
