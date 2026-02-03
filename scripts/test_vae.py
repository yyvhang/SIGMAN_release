import tyro
import numpy as np
import torch
import os
import kiui

from core.model_config.VAE import AllConfigs
from core.dataset.dataloader_test import HGS_1M as Dataset
from accelerate import Accelerator
from safetensors.torch import load_file
from core.modules.DiT_utils import Load_VAE
from core.loss.eval import eval_metrics
from tqdm import tqdm


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

def main():
    opt = tyro.cli(AllConfigs)
    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )
    eval_process = eval_metrics(opt, accelerator)

    model = Load_VAE(opt)
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
        model.load_state_dict(ckpt, strict=False)
    
    # data
    if opt.data_mode == 's3':
        from core.dataset.dataloader_test import ObjaverseDataset as Dataset
    else:
        raise NotImplementedError

    test_dataset = Dataset(opt, training=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    # accelerate
    model, test_dataloader, eval_process = accelerator.prepare(
        model, test_dataloader, eval_process
    )

    pose = load_pose_single('./demo/smplx.npz')
    with torch.no_grad():
        model.eval()
        total_psnr = 0
        total_lpips = 0
        total_ssim = 0
        for i, data in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            #data['smpl_params'] = torch.zeros_like(pose).to(data['input'].dtype).to(data['input'].device)  # an example to load pose
            latent = model.encode(data['input'], data['UV_inital'])
            out = model.decode_uv(latent, data)

            gt_images = out['images_gt'].detach().cpu().numpy()
            pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]

            B = gt_images.shape[0]
            os.makedirs(f'{opt.workspace}/{i}',exist_ok=True)
            for b in range(B):
                pred_ = pred_images[b]
                views = pred_.shape[0]
                for view in range(views):
                    view_img = pred_[view].transpose(1,2,0)
                    kiui.write_image(f'{opt.workspace}/{i}/view_{view}.jpg', view_img)

            lpips, psnr, ssim = eval_process(out)
            total_psnr += psnr.detach()
            total_lpips += lpips.detach()
            total_ssim += ssim

        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        total_lpips = accelerator.gather_for_metrics(total_lpips).mean()
        total_psnr /= len(test_dataloader)
        total_ssim /= len(test_dataloader)
        total_lpips /= len(test_dataloader)
        torch.cuda.empty_cache()
    print(f"psnr: {total_psnr:.4f} ssim: {total_ssim:.4f} lpips: {total_lpips:.4f}")

if __name__ == "__main__":
    main()