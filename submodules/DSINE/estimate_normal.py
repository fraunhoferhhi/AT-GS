import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np

import argparse
from pathlib import Path
import glob
import sys
from tqdm import tqdm
import cv2
import shutil
import os
import glob

this_files_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(this_files_dir)
import utils.utils as utils
import projects.dsine.config as config
from utils.projection import intrins_from_fov, intrins_from_txt

device = torch.device('cuda')
args = config.get_args(test=True)
assert os.path.exists(args.ckpt_path)

if args.NNET_architecture == 'v00':
    from models.dsine.v00 import DSINE_v00 as DSINE
elif args.NNET_architecture == 'v01':
    from models.dsine.v01 import DSINE_v01 as DSINE
elif args.NNET_architecture == 'v02':
    from models.dsine.v02 import DSINE_v02 as DSINE
elif args.NNET_architecture == 'v02_kappa':
    from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
else:
    raise Exception('invalid arch')

model = DSINE(args).to(device)
model = utils.load_checkpoint(args.ckpt_path, model)
model.eval()

img_paths = glob.glob(args.img_dir+'/*.jpg') + glob.glob(args.img_dir+'/*.png') + glob.glob(args.img_dir+'/*.JPG')
output_dir = f"{args.img_dir}/../dsine"
os.makedirs(output_dir, exist_ok=True)
output_visualization_dir = f"{args.img_dir}/../dsine_vis"
os.makedirs(output_visualization_dir, exist_ok=True)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

with torch.no_grad():
    for img_path in tqdm(img_paths, desc=f"Estimating normals", mininterval=5.0):
        print(img_path)
        ext = os.path.splitext(img_path)[1]
        filename = os.path.splitext(os.path.basename(img_path))[0]

        img = Image.open(img_path).convert('RGB')

        if args.resize_img != 1:
            img = img.resize((int(img.width*args.resize_img), int(img.height*args.resize_img)), resample=Image.Resampling.LANCZOS)

        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        # pad input
        _, _, orig_H, orig_W = img.shape
        lrtb = utils.get_padding(orig_H, orig_W)
        img = F.pad(img, lrtb, mode="constant", value=0.0)
        img = normalize(img)

        # get intrinsics
        intrins_path = img_path.replace(ext, '.txt')
        if os.path.exists(intrins_path):
            # NOTE: camera intrinsics should be given as a txt file
            # it should contain the values of fx, fy, cx, cy
            intrins = intrins_from_txt(intrins_path, device=device).unsqueeze(0)
        else:
            # NOTE: if intrins is not given, we just assume that the principal point is at the center
            # and that the field-of-view is 60 degrees (feel free to modify this assumption)
            intrins = intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)
        intrins[:, 0, 2] += lrtb[0]
        intrins[:, 1, 2] += lrtb[2]

        pred_norm = model(img, intrins=intrins)[-1]
        pred_norm = pred_norm[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

        pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy() #[-1,1], [1,H,W,3]
        
        # save to npy file
        np.save(os.path.join(output_dir, f'{filename}_normal.npy'), pred_norm.transpose(0, 3, 1, 2)*(-1)) # [1,3,H,W]

        # NOTE: by saving the prediction as uint8 png format, you lose a lot of precision
        # if you want to use the predicted normals for downstream tasks, we recommend saving them as float32 NPY files
        pred_norm = (((pred_norm + 1) * 0.5) * 255).astype(np.uint8)
        im = Image.fromarray(pred_norm[0,...])
        target_path = os.path.join(output_visualization_dir, f'{filename}_normal.jpg')
        im.save(target_path)
