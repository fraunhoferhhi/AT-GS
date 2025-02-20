import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state, poisson_mesh, str2bool
from utils.image_utils import psnr, depth2rgb, normal2rgb, depth2normal, resample_points, mask_prune, grid_prune, depth2viewDir, img2video
from argparse import ArgumentParser
from torchvision.utils import save_image
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel
import sys
import json
import json5
import time
import re

def render_set(model_path, use_mask, name, iteration, views, gaussians, pipeline, background, write_image, poisson_depth, frame_id):

    if name == 'train':
        grid_dim = 512 if poisson_depth <=9 else 1024
        occ_grid, grid_shift, grid_scale, grid_dim = gaussians.to_occ_grid(0.0, grid_dim, None)

    resampled = []
    psnr_all = []
    # loop through all cams
    for idx, view in enumerate(tqdm(views, desc="Rendering progress", mininterval=5.0)):
        render_pkg = render(view, gaussians, pipeline, background)

        image, normal, depth, opac, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        mask_vis = (opac.detach() > 0.5).float() # rendered_mask
        mask_gt = view.get_gtMask(use_mask)
        gt_image = view.get_gtImage(background, use_mask).cuda()

        if args.out_img4eval and name == 'test':
            view_name = view.image_name.zfill(4)
            view_render_path = os.path.join(args.output_path, "eval_imgs", view_name)
            os.makedirs(view_render_path, exist_ok=True)
            save_image((torch.cat([gt_image, mask_gt], 0)).cpu(), os.path.join(view_render_path, f"frame_{frame_id:06d}_gt.png"))
            # save_image((torch.cat([image, mask_gt], 0)).cpu(), os.path.join(view_render_path, f"frame_{frame_id:06d}_pred.png"))
            save_image((torch.cat([image, mask_vis], 0)).cpu(), os.path.join(view_render_path, f"frame_{frame_id:06d}_pred.png"))
            continue

        psnr_all.append(psnr((gt_image).to(torch.float64), (image).to(torch.float64)).mean().cpu().numpy())
        mask_vis = (opac.detach() > 1e-1) #1e-5
        depth_range = [0, 20]
        mask_clip = (depth > depth_range[0]) * (depth < depth_range[1])
        normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis

        if name == 'train':
            # unproject filtered depth map to 3D points in world space
            # [H, W, 9(xyz_in_world, normals, rgb)]
            pts = resample_points(view, depth, normal, image, mask_vis * mask_gt * mask_clip)
            # prune points by occupancy grid
            grid_mask = grid_prune(occ_grid, grid_shift, grid_scale, grid_dim, pts[..., :3], thrsh=args.occ_thrsh)
            pts = pts[grid_mask]
            resampled.append(pts.cpu())

        if write_image:
            render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
            gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
            info_path = os.path.join(model_path, name, "ours_{}".format(iteration), "info")
            makedirs(render_path, exist_ok=True)
            makedirs(gts_path, exist_ok=True)
            makedirs(info_path, exist_ok=True)

            d2n = depth2normal(depth, mask_vis, view)
            normal_wrt = normal2rgb(normal, mask_vis)
            depth_wrt = depth2rgb(depth, mask_vis)
            d2n_wrt = normal2rgb(d2n, mask_vis)
            normal_wrt += background[:, None, None] * (~mask_vis).expand_as(image) * mask_gt
            depth_wrt += background [:, None, None]* (~mask_vis).expand_as(image) * mask_gt
            d2n_wrt += background[:, None, None] * (~mask_vis).expand_as(image) * mask_gt
            outofmask = mask_vis * (1 - mask_gt)
            mask_vis_wrt = outofmask * (opac - 1) + mask_vis
            img_wrt = torch.cat([gt_image, image, normal_wrt, depth_wrt], 2)
            wrt_mask = torch.cat([opac * mask_gt, mask_vis_wrt, mask_vis_wrt, mask_vis_wrt], 2)
            img_wrt = torch.cat([img_wrt, wrt_mask], 0)
            save_image(img_wrt.cpu(), os.path.join(info_path, '{}'.format(view.image_name) + f".png"))
            save_image(image.cpu(), os.path.join(render_path, '{}'.format(view.image_name) + ".png"))
            save_image((torch.cat([gt_image, mask_gt], 0)).cpu(), os.path.join(gts_path, '{}'.format(view.image_name) + ".png"))

    if name == 'train':
        resampled = torch.cat(resampled, 0)
        mesh_dir = os.path.join(model_path, "..", "meshes")
        os.makedirs(mesh_dir, exist_ok=True)
        mesh_path = os.path.join(mesh_dir, f"Frame_{frame_id:06d}.ply")
        poisson_mesh(mesh_path, resampled[:, :3], resampled[:, 3:6], resampled[:, 6:], poisson_depth, args.use_pymeshlab, args.hhi, args.n_faces, args.add_floor_pc)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, write_image: bool, poisson_depth: int, frame_id):
    with torch.no_grad():
        gaussians = GaussianModel(dataset)
        scales = [1]
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=scales)

        bg_color = [1,1,1] if not args.out_img4eval else [0, 0, 0] # if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, True, "train", scene.loaded_iter, scene.getTrainCameras(scales[0]), gaussians, pipeline, background, write_image, poisson_depth, frame_id)

        if not skip_test:
             render_set(dataset.model_path, True, "test", scene.loaded_iter, scene.getTestCameras(scales[0]), gaussians, pipeline, background, write_image, poisson_depth, frame_id)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--img", action="store_true")
    parser.add_argument("--poisson_depth", default=9, type=int)
    parser.add_argument("--config_path", type=str, default = None)
    parser.add_argument("--occ_thrsh", default=0., type=float)
    parser.add_argument("--use_pymeshlab", type=bool, default=False)
    parser.add_argument("--out_img4eval", action="store_true")
    parser.add_argument("--n_faces", type=int, default=None)
    parser.add_argument("--hhi", type=str, default="False")
    parser.add_argument("--add_floor_pc", type=str, default="False")
    args = parser.parse_args(sys.argv[1:])

    if args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config = json5.load(f)
        for key, value in config.items():
            setattr(args, key, value)
    args.hhi = str2bool(args.hhi)
    args.add_floor_pc = str2bool(args.add_floor_pc)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    pattern = re.compile(r'frame_(\d+)')
    dict_out_sub_dirs = {}
    for out_sub_dir in os.listdir(args.output_path): 
        if pattern.match(out_sub_dir):
            dict_out_sub_dirs[int(pattern.match(out_sub_dir).group(1))] = out_sub_dir

    source_path = args.source_path
    for frame_id in range(args.frame_start, args.frame_end):
        start_time = time.time()
        args.source_path = os.path.join(source_path, dict_out_sub_dirs[frame_id]) 
        args.model_path = os.path.join(args.output_path, dict_out_sub_dirs[frame_id])

        if False:
            # skip if the mesh of this frame already exists
            obj_path = os.path.join(args.model_path, "..", "meshes", f"Frame_{frame_id:06d}.obj")
            ply_path = os.path.join(args.model_path, "..", "meshes", f"Frame_{frame_id:06d}.ply")
            if os.path.exists(obj_path) or os.path.exists(ply_path):
                print(f"Frame {frame_id} already exists.")
                continue

        print("Rendering " + args.model_path)
        render_sets(model.extract(args), -1, pipeline.extract(args), args.skip_train, args.skip_test, args.img, args.poisson_depth, 
                    frame_id = frame_id)
        print(f"Frame {frame_id} finished in {time.time()-start_time} seconds.")