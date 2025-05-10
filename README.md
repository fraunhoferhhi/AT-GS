# AT-GS

Official repository of the paper "Adaptive and Temporally Consistent Gaussian Surfels for Multi-view Dynamic Reconstruction".

WACV 2025 Oral

| [Project](https://fraunhoferhhi.github.io/AT-GS/) 
| [arXiv](https://arxiv.org/abs/2411.06602) |
<!-- | [Paper](https://arxiv.org/abs/2411.06602)  -->


## Environment Setup
Tested on: Ubuntu 22.04, CUDA 11.8, Python 3.10, PyTorch 2.3.1.

Create conda environment:
```shell
conda env create --file environment.yml
conda activate AT-GS
```

## Download Model for Optical Flow 
Pretrained model for Optical Flow Estimation using [RAFT](https://github.com/princeton-vl/RAFT) can be downloaded from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing). Put the model to the path `models/raft-things.pth`.

## Data Preparation
To test on NHR and DNA-Rendering datasets, please refer to [4K4D's guide](https://github.com/zju3dv/4K4D?tab=readme-ov-file#dna-rendering-nhr-and-zju-mocap-datasets) to download these datasets. Alternatively, you can test on other custom datasets.

After downloading the datasets, like most Gaussian Splatting based methods, we need to convert the datasets to the COLMAP format:
   ```
   <frame_000000>
   |---images
   |   |---<image 0>
   |   |---<image 1>
   |   |---...
   |---masks
   |   |---<mask 0>
   |   |---<mask 1>
   |   |---...
   |---sparse
       |---0
           |---cameras.bin
           |---images.bin
           |---points3D.bin
   <frame_000001>
   ...
   ```

## Training
1. Prepear a config file to specify parameters such as the input folder, output folder and more. For example, see `configs/sport_1.json`. Refer to `arguments/__init__.py` for a comprehensive list of configurable hyper-parameters.

2. Train the first frame separately:
    ```shell
    python train_static.py --config_path {cfg_file}
    ```

3. Following [3DGStream](https://github.com/SJoJoK/3DGStream), we initialize the NTC by:
    ```shell
    python cache_warmup.py --config_path {cfg_file}
    ```

4. Train the full sequence:
    ```shell
    python train.py --config_path {cfg_file}
    ```

5. Render images and extract dynamic meshes from the trianed models:
    ```shell
    python render.py --config_path {cfg_file}
    ```
    The meshes are in the folder `{output_path}/meshes/`


## Acknowledgements
This project is built upon [gaussian_surfels](https://github.com/turandai/gaussian_surfels) and [3DGStream](https://github.com/SJoJoK/3DGStream). We thank all the authors for their great work and repos. 

## Citation
If you find our code or paper helpful, please consider citing it:
```bibtex
@inproceedings{chen2025adaptive,
  title={Adaptive and Temporally Consistent Gaussian Surfels for Multi-View Dynamic Reconstruction},
  author={Chen, Decai and Oberson, Brianne and Feldmann, Ingo and Schreer, Oliver and Hilsmann, Anna and Eisert, Peter},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={742--752},
  year={2025},
  organization={IEEE}
}

