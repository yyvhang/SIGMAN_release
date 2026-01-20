[![Website Badge](https://raw.githubusercontent.com/referit3d/referit3d/eccv/images/project_website_badge.svg)](https://yyvhang.github.io/SIGMAN_3D/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.06982-b31b1b.svg?style=plastic)](http://arxiv.org/abs/2504.06982)
# SIGMAN: Scaling 3D Human Gaussian Generation with Millions of Assets (ICCV2025)
PyTorch implementation of **Scaling 3D Human Gaussian Generation with Millions of Assets**.

## 📖 To Do List
1. - [x] release the pretrained model and inference code.
2. - [x] release the training code (VAE and DiT).
3. - [ ] release the HGS-1M dataset with processing pipeline. (contact [liufengqi@sjtu.edu.cn](https://scholar.google.com/citations?user=po8B17IAAAAJ&hl=en))

# Setup
1. Install the dependencies
```
# xformers is required! please refer to https://github.com/facebookresearch/xformers for details.
# for example, we use torch 2.1.0 + cuda 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

git clone https://github.com/graphdeco-inria/gaussian-splatting.git

# for mesh extraction
pip install git+https://github.com/NVlabs/nvdiffrast

# other dependencies
pip install -r requirements.txt
```

2. Download the [SMPLX](https://smpl-x.is.tue.mpg.de/) and put it into the folder `/core/modules/deformers/smplx/SMPLX`; Then, download the following files (refer to [E3Gen](https://github.com/olivia23333/E3Gen?tab=readme-ov-file)) and place them in `/core/modules/deformers/template`:
- [SMPL-X segmentation file](https://github.com/Meshcapade/wiki/blob/main/assets/SMPL_body_segmentation/smplx/smplx_vert_segmentation.json)(smplx_vert_segmentation.json)
- [SMPL-X UV](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_uv.zip)(smplx_uv.obj)
- [SMPL-X FLAME Correspondence](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_mano_flame_correspondences.zip)(SMPL-X__FLAME_vertex_ids.npy)
- [FLAME with mouth Mesh Template](https://github.com/philgras/neural-head-avatars/blob/main/assets/flame/head_template_mesh_mouth.obj)(head_template_mesh_mouth.obj)
- [FLAME Mesh Template](https://github.com/yfeng95/DECA/blob/master/data/head_template.obj)(head_template.obj)
- [FLAME Mask](https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip)(FLAME_masks.pkl) 

3. Extract templates
```
cd core/modules/deformers

# preprocess for uv, obtain new uv for smplx_mouth.obj
python preprocess_smplx.py

# save subdivide smplx mesh and corresponding uv
python subdivide_smplx.py

# save parameters for init
python utils_smplx.py
python utils_uvpos.py
```

4. Download the pretrained weights
Our VAE and DiT weights can be download from the [huggingface](https://huggingface.co/Mr-Hang/SIGMAN/tree/main):
```
# VAE
cd ckpt/autoencoder
wget https://huggingface.co/Mr-Hang/SIGMAN/blob/main/autoencoder.safetensors

# DiT
cd ckpt/transformer
wget https://huggingface.co/Mr-Hang/SIGMAN/blob/main/transformer.safetensors

# Image Encoder
mkdir ckpt/sapiens_1b
cd ckpt/sapiens_1b
wget https://huggingface.co/facebook/sapiens-pretrain-1b-torchscript/blob/main/sapiens_1b_epoch_173_torchscript.pt2
```

# Inference
```
# Generation
python scripts/test_DiT.py --image_path demo/images/demo.jpg --pose_path demo/poses/smplx_demo.npz

# Eval of VAE
python scripts/test_vae.py
```
 
# Training
We use an `.npy` file to record the root directory of all items for training and evaluation. To train the model, you should should organize each data item in the following format:
```
├── UV
├── camera_full_calibration.json
├── smplx.npz
├── rgb_map
│   ├── 0000.jpg
│   ├── 0001.jpg
│   ├── ...
│   ├── 0090.jpg
└── mask_map
│   ├── 0000.png
│   ├── 0001.png
│   ├── ...
│   ├── 0090.png
```

To obtain UVmap for training the VAE, please referr to:
```
bash core/proj_UV/runs.sh
```

Then, training the VAE or DiT:
```
# VAE
# --disc_start specifies the step to employ the Gan loss
accelerate launch --config_file configs/training.yaml --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} --machine_rank=${RANK} \
train_vae.py vae_b --workspace /output_folder --batch_size 8 --wandb_name xxx --disc_start xxx


# DiT
accelerate launch --config_file configs/training.yaml --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} --machine_rank=${RANK} \
    train_DiT.py DiT --workspace /output_folder --batch_size xxx
```

### Acknowledgements
- [LGM](https://github.com/3DTopia/LGM)
- [E3Gen](https://github.com/olivia23333/E3Gen?tab=readme-ov-file)
- [CogVideo](https://github.com/zai-org/CogVideo)

## 💌 Citation
```
@article{yang2025sigman,
  title={SIGMAN: Scaling 3D Human Gaussian Generation with Millions of Assets},
  author={Yang, Yuhang and Liu, Fengqi and Lu, Yixing and Zhao, Qin and Wu, Pingyu and Zhai, Wei and Yi, Ran and Cao, Yang and Ma, Lizhuang and Zha, Zheng-Jun and others},
  journal={arXiv preprint arXiv:2504.06982},
  year={2025}
}
```