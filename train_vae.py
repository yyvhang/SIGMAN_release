import tyro
import time
import random
import wandb
import os
import pdb
import numpy as np

import torch
import torch.nn as nn
import kiui

from core.model_config.VAE import AllConfigs
from core.modules.autoencoder import VAE
from core.loss.whole_loss import LPIPSWithDiscriminator
from core.loss.eval import eval_metrics
from core.dataset.dataloader_VAE import HGS_1M as Dataset

from accelerate import Accelerator
from safetensors.torch import load_file, save_file

def set_modules_requires_grad(modules, requires_grad):
    for module in modules:
        module.requires_grad_(requires_grad)

def main():    
    opt = tyro.cli(AllConfigs)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
    # Set seeds
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if opt.wandb:
        if accelerator.is_main_process:
            wandb.init(project="vae_ablation", entity='yuhang_team', name=opt.wandb_name, config=opt)

    # model
    model = VAE(opt)
    loss_cls = LPIPSWithDiscriminator(opt)
    eval_process = eval_metrics(opt, accelerator)

    best_psnr = 0
    best_psnr_epoch = -1
    best_ssim = 0
    best_ssim_epoch = -1
    best_lpips = 1000000
    best_lpips_epoch = -1
    # resume
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')['model_state_dict']
        
        # tolerant load (only load matching shapes)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    print(f'[WARN] param shape unmatch {k}: {v.shape}')
            else:
                print(f'[WARN] unexpected param {k}: {v.shape}')
        model.load_state_dict(state_dict, strict=False)
    
    VAE_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    num_trainable_parameters = sum(param.numel() for param in VAE_parameters)

    VAE_encoder_parameters = list(filter(lambda p: p.requires_grad, model.autoencoder.encoder.parameters()))
    num_encoder_trainable_parameters = sum(param.numel() for param in VAE_encoder_parameters)

    VAE_decoder_parameters = list(filter(lambda p: p.requires_grad, model.autoencoder.decoder.parameters()))
    num_decoder_trainable_parameters = sum(param.numel() for param in VAE_decoder_parameters)

    if accelerator.is_main_process:
        accelerator.print(f"Training started with {accelerator.num_processes} processes")
        accelerator.print(f"VAE trainable parameters: {num_trainable_parameters / 1e9:.2f} billion")
        accelerator.print(f"Encoder trainable parameters: {num_encoder_trainable_parameters / 1e9:.2f} billion")
        accelerator.print(f"Decoder trainable parameters: {num_decoder_trainable_parameters / 1e9:.2f} billion")

    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # optimizer
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
    optimizer_d = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, loss_cls.discriminator.parameters()), lr=opt.lr, weight_decay=0.01
    )
    modules_to_train = [module for module in model.get_model()]
    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 5000 / total_steps

    # accelerate
    model, loss_cls, optimizer_g, optimizer_d, train_dataloader, test_dataloader, eval_process = accelerator.prepare(
        model, loss_cls, optimizer_g, optimizer_d, train_dataloader, test_dataloader, eval_process
    )
    global_step = 0
    # loop
    for epoch in range(opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                if (
                    global_step % 2 == 1
                    and global_step >= loss_cls.module.discriminator_iter_start
                ):
                    set_modules_requires_grad(modules_to_train, False)
                    step_gen = False
                    step_dis = True
                else:
                    set_modules_requires_grad(modules_to_train, True)
                    step_gen = True
                    step_dis = False

                if step_gen:
                    optimizer_g.zero_grad()
                    out = model(data, step_ratio)
                    g_loss, g_log = loss_cls(
                        out,
                        out['posterior'],
                        optimizer_idx=0,
                        global_step=global_step,
                        last_layer=model.module.get_last_layer(),
                    )

                    psnr = out['psnr']
                    loss = g_log['loss']
                    loss_l1 = g_log['L1']
                    loss_kl = g_log['kl']
                    loss_lpips = g_log['lpips']
                    loss_ganG = g_log['GAN_G']
                    accelerator.backward(g_loss)
                    if accelerator.is_main_process:
                        print(f"[train] index: {i} epoch: {epoch} loss: {g_loss:.6f} psnr: {psnr:.4f}")
                    # gradient clipping
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                    optimizer_g.step()

                    total_loss += g_loss.detach()
                    total_psnr += psnr.detach()
                    if accelerator.is_main_process:
                        if opt.wandb:
                            wandb.log({
                                "step": global_step,
                                "train_step_loss": g_loss.item(),
                                "train_step_loss_l1": loss_l1.item(),
                                "train_step_loss_kl": loss_kl.item(),
                                "train_step_loss_lpips": loss_lpips.item(),
                                "train_step_loss_ganG": loss_ganG.item(),
                                "train_step_psnr": psnr.item(),
                                "train_step_ganG": loss_ganG.item()
                            })
                
                if step_dis:
                    optimizer_d.zero_grad()
                    out = model(data, step_ratio)
                    d_loss, d_log = loss_cls(
                        out,
                        out['posterior'],
                        optimizer_idx=1,
                        global_step=global_step,
                        last_layer=model.module.get_last_layer()
                    )
                    total_loss += d_loss.detach()
                    accelerator.backward(d_loss)
                    if accelerator.is_main_process:
                        print(f"[train] index: {i} epoch: {epoch} d_loss: {d_loss:.6f}")
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(loss_cls.module.discriminator.parameters(), opt.gradient_clip)

                    optimizer_d.step()
                    if accelerator.is_main_process:
                        if opt.wandb:
                            wandb.log({
                                "step": global_step,
                            "train_step_ganD": d_loss.item()
                        })

                global_step += 1
                if global_step % 200 == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        model_path = os.path.join(opt.workspace, 'autoencoder')
                        os.makedirs(model_path, exist_ok=True)
                        save_file(unwrapped_model.state_dict(), os.path.join(model_path, f"autoencoder.safetensors"))

                        if global_step >= loss_cls.module.discriminator_iter_start:
                            unwrapped_discriminator = accelerator.unwrap_model(loss_cls.module.discriminator)
                            discriminator_path = os.path.join(opt.workspace, 'discriminator')
                            os.makedirs(discriminator_path, exist_ok=True)
                            save_file(unwrapped_discriminator.state_dict(), os.path.join(discriminator_path, f"discriminator.safetensors"))

            if accelerator.is_main_process:
                # logging
                if i % 400 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}")
                
                # save log images
                if i % 400 == 0:
                    gt_images = out['images_gt'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    B = gt_images.shape[0]
                    all_images = []
                    for b in range(B):
                        gt_img = gt_images[b] # [V, 3, output_size, output_size]
                        gt_img = gt_img.transpose(2, 0, 3, 1).reshape(gt_img.shape[2], -1, 3) # [output_size, V*output_size, 3]

                        pred_img = pred_images[b]
                        pred_img = pred_img.transpose(2, 0, 3, 1).reshape(pred_img.shape[2], -1, 3)

                        combined = np.concatenate([gt_img, pred_img], axis=0)
                        all_images.append(combined)
                    
                    # 水平拼接所有batch的图像
                    final_image = np.concatenate(all_images, axis=0)
                    kiui.write_image(f'{opt.workspace}/train_{global_step}.jpg', final_image)

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            if global_step >= loss_cls.module.discriminator_iter_start:
                total_psnr = total_psnr / len(train_dataloader) * 2
            else:
                total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
            if opt.wandb:
                wandb.log(
                    {
                    "epoch": epoch,
                    "train_total_loss": total_loss.item(),
                    "train_total_psnr": total_psnr.item(),
                }
            )        

        accelerator.wait_for_everyone()

        # eval
        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                total_psnr = 0
                total_lpips = 0
                total_ssim = 0
                for i, data in enumerate(test_dataloader):

                    out = model(data)
                    lpips, psnr, ssim = eval_process(out)
                    total_psnr += psnr.detach()
                    total_lpips += lpips.detach()
                    total_ssim += ssim
                    # save some images
                    if accelerator.is_main_process and i % 20 == 0:
                        gt_images = out['images_gt'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        B = gt_images.shape[0]
                        all_images = []
                        for b in range(B):
                            gt_img = gt_images[b] # [V, 3, output_size, output_size]
                            gt_img = gt_img.transpose(2, 0, 3, 1).reshape(gt_img.shape[2], -1, 3) # [output_size, V*output_size, 3]
                            
                            pred_img = pred_images[b]
                            pred_img = pred_img.transpose(2, 0, 3, 1).reshape(pred_img.shape[2], -1, 3)

                            combined = np.concatenate([gt_img, pred_img], axis=0)
                            all_images.append(combined)
                        
                        final_image = np.concatenate(all_images, axis=0)
                        kiui.write_image(f'{opt.workspace}/eval_{global_step}.jpg', final_image)

                torch.cuda.empty_cache()

                total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
                total_lpips = accelerator.gather_for_metrics(total_lpips).mean()
                if accelerator.is_main_process:
                    total_psnr /= len(test_dataloader)
                    total_ssim /= len(test_dataloader)
                    total_lpips /= len(test_dataloader)
                    accelerator.print(f"[eval] epoch: {epoch} psnr: {total_psnr:.4f} ssim: {total_ssim:.4f} lpips: {total_lpips:.4f}")
                    if opt.wandb:
                        wandb.log({
                            "epoch": epoch,
                            "eval_total_psnr": total_psnr.item(),
                            "eval_total_ssim": total_ssim.item(),
                            "eval_total_lpips": total_lpips.item(),
                        })
                    if total_psnr > best_psnr:
                        best_psnr = total_psnr
                        best_psnr_epoch = epoch
                    if total_ssim > best_ssim:
                        best_ssim = total_ssim
                        best_ssim_epoch = epoch
                    if total_lpips < best_lpips:
                        best_lpips = total_lpips
                        best_lpips_epoch = epoch

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        model_path = os.path.join(opt.workspace, 'autoencoder')
        save_file(unwrapped_model.state_dict(), os.path.join(model_path, f"autoencoder.safetensors"))
        accelerator.print(f"Training finished. Best PSNR: {best_psnr:.4f} at epoch {best_psnr_epoch}")
        if opt.wandb:
            wandb.run.summary["best_psnr"] = best_psnr
            wandb.run.summary["best_psnr_epoch"] = best_psnr_epoch
            wandb.run.summary["best_ssim"] = best_ssim
            wandb.run.summary["best_ssim_epoch"] = best_ssim_epoch
            wandb.run.summary["best_lpips"] = best_lpips
            wandb.run.summary["best_lpips_epoch"] = best_lpips_epoch
    wandb.finish()

if __name__ == "__main__":
    main()
