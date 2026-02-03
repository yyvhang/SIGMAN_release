import tyro
import random
import wandb
import os
import sys
import json
import argparse
import numpy as np
import torch
import cv2
import math
import kiui
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Optional, Tuple
from PIL import Image
from core.model_config.DiT_DDPM import AllConfigs
from core.modules.DiT_utils import Load_VAE, Render
from core.modules.DiT import DiT3DModel
from core.dataset.dataloader_test import HGS_1M as Dataset
from core.loss.eval import eval_metrics
from core.modules.sample_pipeline.DDPM_sample_pipeline import SamplesPipeline
from diffusers.utils.torch_utils import is_compiled_module
from diffusers import CogVideoXDDIMScheduler
from accelerate import Accelerator
from safetensors.torch import load_file
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.models.embeddings import get_2d_rotary_pos_embed

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def _parse_cli():

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image_path", type=str, default="./demo/images/demo.jpg")
    parser.add_argument("--pose_path", type=str, default="./demo/poses/smplx_demo.npz")
    known, remaining = parser.parse_known_args(sys.argv[1:])

    opt = tyro.cli(AllConfigs, args=remaining)
    return opt, known

def set_modules_requires_grad(modules, requires_grad):
    for module in modules:
        module.requires_grad_(requires_grad)

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 512,
    base_width: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_2d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width)
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

def load_ckpt(model, ckpt_path):
    if ckpt_path.endswith('safetensors'):
        ckpt = load_file(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    # tolerant load (only load matching shapes)
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict: 
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
        else:
            print(f'[WARN] unexpected param {k}: {v.shape}')
    model.load_state_dict(state_dict, strict=False)

def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        model = torch.jit.load(checkpoint)
        # Set requires_grad to False for all parameters
        for param in model.parameters():
            param.requires_grad = False
        return model
    else:
        return torch.export.load(checkpoint).module()
    
def load_pose(path):
    smpl_param_path = np.load(path, allow_pickle=True)
    seq_long = smpl_param_path['betas'].shape[0]
    poses = []
    for seq_id in range(seq_long):
        betas = torch.tensor(smpl_param_path['betas'][seq_id], dtype=torch.float32)
        body_pose = torch.tensor(smpl_param_path['body_pose'][seq_id], dtype=torch.float32)
        global_orient = torch.tensor(smpl_param_path['global_orient'][seq_id], dtype=torch.float32)
        transl = torch.tensor(smpl_param_path['transl'][seq_id], dtype=torch.float32)
        expression = torch.tensor(smpl_param_path['expression'][seq_id], dtype=torch.float32)
        left_hand_pose = torch.tensor(smpl_param_path['left_hand_pose'][seq_id], dtype=torch.float32)
        right_hand_pose = torch.tensor(smpl_param_path['right_hand_pose'][seq_id], dtype=torch.float32)
        jaw_pose = torch.tensor(smpl_param_path['jaw_pose'][seq_id],dtype=torch.float32)
        leye_pose = torch.tensor(smpl_param_path['leye_pose'][seq_id],dtype=torch.float32)
        reye_pose = torch.tensor(smpl_param_path['reye_pose'][seq_id],dtype=torch.float32)
        smpl_params = torch.cat([transl, global_orient, betas, body_pose, expression, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose], dim=-1)
        poses.append(smpl_params)
    poses = torch.stack(poses,dim=0)

    return poses

def load_pose_single(path):
    smplx_param_path = np.load(path, allow_pickle=True)

    betas = torch.tensor(smplx_param_path['betas'], dtype=torch.float32)
    body_pose = torch.tensor(smplx_param_path['body_pose'], dtype=torch.float32)
    global_orient = torch.tensor(smplx_param_path['global_orient'], dtype=torch.float32)
    transl = torch.tensor(smplx_param_path['transl'], dtype=torch.float32)
    expression = torch.tensor(smplx_param_path['expression'], dtype=torch.float32)
    left_hand_pose = torch.tensor(smplx_param_path['left_hand_pose'], dtype=torch.float32)
    right_hand_pose = torch.tensor(smplx_param_path['right_hand_pose'], dtype=torch.float32)
    jaw_pose = torch.tensor(smplx_param_path['jaw_pose'],dtype=torch.float32)
    leye_pose = torch.tensor(smplx_param_path['leye_pose'],dtype=torch.float32)
    reye_pose = torch.tensor(smplx_param_path['reye_pose'],dtype=torch.float32)

    smpl_params = torch.cat([transl, global_orient, betas, body_pose, expression, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose], dim=-1)
    if len(smpl_params.shape) == 1:
        smpl_params = smpl_params.unsqueeze(0)
    return smpl_params

def load_amass(npz_path):

    data = dict(np.load(npz_path, allow_pickle = True))
    smplx_data_expression = np.zeros((data['pose_body'].shape[0], 10))
    smplx_data_betas = np.tile(data['betas'], (data['pose_body'].shape[0], 1))
    smplx_data_body_pose = data['pose_body']
    smplx_data_transl = data['trans']
    smplx_data_global_orient = data['root_orient']
    smplx_data_left_hand_pose = data['pose_hand'][:,:45]
    smplx_data_right_hand_pose = data['pose_hand'][:,45:]
    smplx_data_jaw_pose = data['pose_jaw']
    smplx_data_leye_pose = data['pose_eye'][:,:3]
    smplx_data_reye_pose = data['pose_eye'][:,3:]

    expression = torch.from_numpy(smplx_data_expression).float()
    transl = torch.from_numpy(smplx_data_transl).float()
    global_orient = torch.from_numpy(smplx_data_global_orient).float()
    body_pose = torch.from_numpy(smplx_data_body_pose).float()
    betas = torch.zeros(smplx_data_betas.shape[0],10).float()
    left_hand_pose = torch.from_numpy(smplx_data_left_hand_pose).float()
    right_hand_pose = torch.from_numpy(smplx_data_right_hand_pose).float()
    jaw_pose = torch.from_numpy(smplx_data_jaw_pose).float()
    leye_pose = torch.from_numpy(smplx_data_leye_pose).float()
    reye_pose = torch.from_numpy(smplx_data_reye_pose).float()

    smplx_params = torch.cat([transl, global_orient, betas, body_pose, expression, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose], dim=-1)
    return smplx_params

def getProjectionMatrix(znear, zfar, fovX, fovY, K = None, img_h = None, img_w = None):
    if K is None:
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right
    else:
        near_fx = znear / K[0, 0]
        near_fy = znear / K[1, 1]

        left = - (img_w - K[0, 2]) * near_fx
        right = K[0, 2] * near_fx
        bottom = (K[1, 2] - img_h) * near_fy
        top = K[1, 2] * near_fy

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def inference():
    opt, known = _parse_cli()
    seed = 42  # You can change this to any integer value
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    '''
    load model
    '''
    VAE = Load_VAE(opt)
    model = DiT3DModel(
        num_attention_heads=opt.num_attention_heads,
        attention_head_dim=opt.attention_head_dim,
        in_channels=opt.in_channels,
        out_channels=opt.out_channels,
        num_layers=opt.num_layers,
        text_embed_dim=opt.text_embed_dim,
        sample_width=opt.sample_width,
        sample_height=opt.sample_height,
        sample_frames=opt.sample_frames,
        patch_size=opt.patch_size,
        max_text_seq_length=opt.max_text_seq_length,
        temporal_compression_ratio=opt.temporal_compression_ratio,
        use_rotary_positional_embeddings=opt.use_rotary_positional_embeddings,
        use_learned_positional_embeddings=opt.use_learned_positional_embeddings,
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained("./core", subfolder="scheduler")
    renderer = Render(opt)
    image_encoder = load_model('.ckpt/sapiens_1b/sapiens_1b_epoch_173_torchscript.pt2', True)

    device = accelerator.device
    if opt.load_ckpt:
        load_ckpt(model, opt.dit_path)
    VAE.requires_grad_(False)
    model.requires_grad_(False)
    VAE.to(device)
    model.to(device)
    renderer.to(device)
    image_encoder.to(device)
    image_encoder.eval()

    if opt.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    model = accelerator.prepare(model)
    '''
    loda data
    '''
    intr = np.array([
        [
            1100.0,
            0.0,
            512.0
        ],
        [
            0.0,
            1100.0,
            512.0
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ])
    proj_matrix = getProjectionMatrix(opt.znear, opt.zfar, opt.FoVx, opt.FoVy, 
                                                K = intr, img_h = 1024, img_w = 1024).transpose(0, 1)
    data = {}
    camera = json.load(open('core/dataset/camera_full_calibration.json','r'))
    vids = [30, 37, 45, 53, 65, 85, 0, 6, 15, 24, 34, 41, 49, 57, 60, 68, 72, 75, 80, 83] 
    cam_poses = []

    for vid in vids:
        camera_pose = camera[f'{str(vid).zfill(4)}']
        camera_matrix = torch.eye(4)
        camera_matrix[:3, :3] = torch.tensor(camera_pose['R'])
        camera_matrix[:3, 3] = torch.tensor(camera_pose['T'])
        w2c = camera_matrix
        w2c = w2c.to(dtype=torch.float32).reshape(4, 4)
        cam_poses.append(w2c)
    cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

    cam_view = cam_poses.transpose(1, 2) # [V, 4, 4] w2c
    cam_view_proj = cam_view @ proj_matrix # [V, 4, 4] c2p
    cam_pos = torch.linalg.inv(cam_poses)[:, :3, 3] # [V, 3]
    
    data['cam_view'] = cam_view
    data['cam_view_proj'] = cam_view_proj
    data['cam_pos'] = cam_pos

    img_path = known.image_path
    pose_path = known.pose_path
    
    with torch.no_grad():
        sample_folder = f'{opt.workspace}/outputs'
        os.makedirs(sample_folder, exist_ok=True)
        
        # 加载pose
        pose = load_pose_single(pose_path)
        #pose = load_poses(pose_path)  #for sequence
        #pose = load_amass(pose_path)  #for amass pose
        
        pipe = SamplesPipeline(
            opt,
            transformer=model,
            vae=VAE,
            renderer=renderer,
            scheduler=scheduler,
            img_encoder=image_encoder,
            guidance_scale=3.5,
            vae_scale_factor=opt.vae_scaling_factor
        )
        pose = pose.to(device)
        data['smpl_params'] = pose
        
        # 使用PIL读取图片并resize到512x512
        pil_image = Image.open(img_path).convert('RGB')
        pil_image = pil_image.resize((512, 512), Image.LANCZOS)
        image = torch.from_numpy(np.array(pil_image).astype(np.float32) / 255)
        image = image.permute(2, 0, 1) # [3, 512, 512]
        image = image.unsqueeze(0).contiguous() # [1, 3, 512, 512]

        sapines_input = F.interpolate(image, size=(1024, 1024), mode='bilinear', align_corners=False)
        sapines_input = TF.normalize(sapines_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD).to(device)
        out = pipe(data, sapines_input, sapines_input, dtype=weight_dtype, height = opt.height, width = opt.width, inference=True)
        pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]

        pred_images_view = pred_images[0]
        os.makedirs(f'{sample_folder}/0', exist_ok=True)
        for view in range(pred_images_view.shape[0]):
            kiui.write_image(f'{sample_folder}/0/{view}.jpg', pred_images_view[view].transpose(1, 2, 0))
        
        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
        image_name = os.path.basename(img_path)
        kiui.write_image(f'{sample_folder}/{image_name}', pred_images)
        
    torch.cuda.empty_cache()

def eval():    
    opt, _known = _parse_cli()
    seed = 42  # You can change this to any integer value
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # model
    VAE = Load_VAE(opt)
    model = DiT3DModel(
        num_attention_heads=opt.num_attention_heads,
        attention_head_dim=opt.attention_head_dim,
        in_channels=opt.in_channels,
        out_channels=opt.out_channels,
        num_layers=opt.num_layers,
        text_embed_dim=opt.text_embed_dim,
        sample_width=opt.sample_width,
        sample_height=opt.sample_height,
        sample_frames=opt.sample_frames,
        patch_size=opt.patch_size,
        max_text_seq_length=opt.max_text_seq_length,
        temporal_compression_ratio=opt.temporal_compression_ratio,
        use_rotary_positional_embeddings=opt.use_rotary_positional_embeddings,
        use_learned_positional_embeddings=opt.use_learned_positional_embeddings,
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained(opt.DiT_pretrain, subfolder="scheduler")
    renderer = Render(opt)
    image_encoder = load_model('.ckpt/sapiens_1b/sapiens_1b_epoch_173_torchscript.pt2', True)

    if opt.load_ckpt:
        load_ckpt(model, opt.dit_path)
    VAE.requires_grad_(False)
    model.requires_grad_(False)
    VAE.to(accelerator.device)
    model.to(accelerator.device)
    renderer.to(accelerator.device)
    image_encoder.to(accelerator.device)
    image_encoder.eval()

    if opt.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    eval_process = eval_metrics(opt, accelerator)

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    model, test_dataloader, eval_process = accelerator.prepare(
        model, test_dataloader, eval_process
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    # eval

    with torch.no_grad():
        total_psnr = 0
        total_lpips = 0
        total_ssim = 0
        for i, data in enumerate(test_dataloader):
            ids = data['item']
            pipe = SamplesPipeline(
                opt,
                transformer=unwrap_model(model),
                vae=unwrap_model(VAE),
                renderer=unwrap_model(renderer),
                scheduler=scheduler,
                img_encoder=unwrap_model(image_encoder),
                guidance_scale=3.5,
                vae_scale_factor = opt.vae_scaling_factor
            )
            out = pipe(data, data['input'][:,0:1,:,:,:], data['sapines_input'],dtype=weight_dtype, height = opt.height, width = opt.width)
            lpips, psnr, ssim = eval_process(out)
            total_psnr += psnr.detach()
            total_lpips += lpips.detach()
            total_ssim += ssim

            gt_images = out['images_gt'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
            pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
            points = out['gaussian_pts'].float().detach().cpu().numpy()

            B = gt_images.shape[0]
            all_images = []
            for b in range(B):
                gt_img = gt_images[b] # [V, 3, output_size, output_size]
                gt_img = gt_img.transpose(2, 0, 3, 1).reshape(gt_img.shape[2], -1, 3) # [output_size, V*output_size, 3]
                
                pred_img = pred_images[b]
                pred_img = pred_img.transpose(2, 0, 3, 1).reshape(pred_img.shape[2], -1, 3)

                combined = np.concatenate([gt_img, pred_img], axis=0)
                all_images.append(combined)

                # gaussian_point = points[b]
                # pts = o3d.geometry.PointCloud()
                # pts.points = o3d.utility.Vector3dVector(gaussian_point)
            
                # # Save as PLY file
                # o3d.io.write_point_cloud(f'{ply_folder}/pointcloud_{i}.ply', pts)
            
            final_image = np.concatenate(all_images, axis=0)
            kiui.write_image(f'{opt.workspace}/Eval/{i}.jpg', final_image)

    torch.cuda.empty_cache()
    total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
    total_lpips = accelerator.gather_for_metrics(total_lpips).mean()
    if accelerator.is_main_process:
        total_psnr /= len(test_dataloader)
        total_ssim /= len(test_dataloader)
        total_lpips /= len(test_dataloader)

    print(f"psnr: {total_psnr:.4f} ssim: {total_ssim:.4f} lpips: {total_lpips:.4f}")

if __name__ == "__main__":
    inference()
    #eval()
