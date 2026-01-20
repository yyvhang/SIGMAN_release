"""
DiT (Diffusion Transformer) DDPM Training Script
This script trains a DiT model using DDPM (Denoising Diffusion Probabilistic Models) approach.
"""

# Standard library imports
import os
import random
import time
from typing import Optional, Tuple

# Third-party imports
import numpy as np
import torch
import tyro
import wandb
import kiui
from accelerate import Accelerator
from diffusers import CogVideoXDDIMScheduler
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.utils.torch_utils import is_compiled_module
from safetensors.torch import load_file, save_file
from core.dataset.dataloader_DiT import HGS_1M as Dataset

# Local imports
from core.model_config.DiT_DDPM import AllConfigs
from core.modules.DiT import DiT3DModel
from core.modules.sample_pipeline.DDPM_sample_pipeline import SamplesPipeline
from core.modules.DiT_utils import Load_VAE, Render
from core.modules.encode.embeddings import encode_image_SAPIENS


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
    """Prepare rotary positional embeddings for the model.
    
    Args:
        height: Height of the input image
        width: Width of the input image
        vae_scale_factor_spatial: VAE spatial scale factor
        patch_size: Size of patches
        attention_head_dim: Dimension of attention heads
        device: Device to place tensors on
        base_height: Base height for grid calculation
        base_width: Base width for grid calculation
        
    Returns:
        Tuple of cosine and sine frequency tensors
    """
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid(
        (grid_height, grid_width), 
        base_size_width, 
        base_size_height
    )
    
    freqs_cos, freqs_sin = get_2d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width)
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def load_model(checkpoint: str, use_torchscript: bool = False):
    """Load a model from checkpoint.
    
    Args:
        checkpoint: Path to the checkpoint file
        use_torchscript: Whether to load as TorchScript model
        
    Returns:
        Loaded model
    """
    if use_torchscript:
        model = torch.jit.load(checkpoint)
        # Set requires_grad to False for all parameters
        for param in model.parameters():
            param.requires_grad = False
        return model
    else:
        return torch.export.load(checkpoint).module()

def encode_input(vae, image_encoder, input_tensor, uv_initial, images, opt, cfg=False):
    """Simplified encoding function for input images.
    
    Args:
        vae: VAE model for encoding
        image_encoder: Image encoder model
        input_tensor: Input tensor [B, V, C, H, W]
        uv_initial: Initial UV coordinates
        images: Conditioning images [B, V, C, H, W]
        opt: Configuration options
        cfg: Whether to use classifier-free guidance
        
    Returns:
        Tuple of multi-view latents and image embeddings
    """
    dtype = input_tensor.dtype
    multi_view_latents = vae.encode(input_tensor, uv_initial.to(dtype)) * opt.vae_scaling_factor
    cond_image = images.clone().to(dtype=torch.float32)
    image_embeddings = encode_image_SAPIENS(cond_image, image_encoder, cfg, opt.condition_mode)

    if random.random() < opt.noised_condition_dropout:
        image_embeddings = torch.zeros_like(image_embeddings)
    image_embeddings = image_embeddings.to(dtype=dtype)

    return multi_view_latents, image_embeddings

def unwrap_model(model, accelerator):
    """Unwrap model from accelerator and compiled module if needed.
    
    Args:
        model: Model to unwrap
        accelerator: Accelerator instance
        
    Returns:
        Unwrapped model
    """
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_visualization(sample_folder, step_name, gt_images, pred_images, vae_images):
    """Save visualization of ground truth, VAE reconstruction, and predicted images.
    
    Args:
        sample_folder: Folder to save visualizations
        step_name: Name/identifier for the current step
        gt_images: Ground truth images
        pred_images: Predicted images
        vae_images: VAE reconstructed images
    """
    # Process each batch sample separately
    B = gt_images.shape[0]
    all_images = []
    for b in range(B):
        # Process ground truth images
        gt_img = gt_images[b]  # [V, 3, output_size, output_size]
        gt_img = gt_img.transpose(2, 0, 3, 1).reshape(gt_img.shape[2], -1, 3)  # [output_size, V*output_size, 3]
        
        # Process predicted images
        pred_img = pred_images[b]
        pred_img = pred_img.transpose(2, 0, 3, 1).reshape(pred_img.shape[2], -1, 3)

        # Process VAE reconstructed images
        vae_img = vae_images[b]
        vae_img = vae_img.transpose(2, 0, 3, 1).reshape(vae_img.shape[2], -1, 3)

        # Stack images vertically: GT, VAE, Prediction
        combined = np.concatenate([gt_img, vae_img, pred_img], axis=0)
        all_images.append(combined)
    
    # Concatenate all batch samples horizontally
    final_image = np.concatenate(all_images, axis=0)
    kiui.write_image(f'{sample_folder}/{step_name}.jpg', final_image)


def main():
    """Main training function."""
    # Parse command line arguments
    opt = tyro.cli(AllConfigs)
    
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    # Initialize wandb for experiment tracking
    if opt.wandb and accelerator.is_main_process:
        wandb.init(project="DiT", entity='SIGMAN', name=opt.wandb_name, config=opt)

    weight_dtype = None
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Initialize models
    vae = Load_VAE(opt)
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
    image_encoder = load_model(
        '.ckpt/sapiens_1b/sapiens_1b_epoch_173_torchscript.pt2', 
        True
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained("./core", subfolder="scheduler")
    renderer = Render(opt)

    # Set up model parameters and devices
    vae.requires_grad_(False)
    for param in image_encoder.parameters():
        param.requires_grad = False
    
    vae.to(accelerator.device)
    model.to(accelerator.device)
    renderer.to(accelerator.device)
    image_encoder.to(accelerator.device)
    image_encoder.eval()

    # Enable gradient checkpointing if specified
    if opt.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Resume from checkpoint if specified
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')['model_state_dict']
        
        # Load checkpoint with shape tolerance
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(
                        f'[WARN] Mismatching shape for param {k}: '
                        f'ckpt {v.shape} != model {state_dict[k].shape}, ignored.'
                    )
            else:
                accelerator.print(f'[WARN] Unexpected param {k}: {v.shape}')
        model.load_state_dict(state_dict, strict=False)
    
    # Create data loaders
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
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=opt.lr, 
        weight_decay=0.05, 
        betas=(0.9, 0.95)
    )
    
    # Get trainable parameters for logging
    trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    # Prepare models and data loaders with accelerator
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )
    
    # Log training information
    if accelerator.is_main_process:
        num_trainable_parameters = sum(param.numel() for param in trainable_parameters)
        accelerator.print(f"Training started with {accelerator.num_processes} processes")
        accelerator.print(f"Num trainable parameters: {num_trainable_parameters / 1e9:.2f} billion")

    # Create sample directory
    sample_folder = f'{opt.workspace}/samples'
    os.makedirs(sample_folder, exist_ok=True)
    
    # Initialize global step counter
    global_step = 0
    
    # Training loop
    for epoch in range(opt.num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Calculate step ratio for logging
                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Get input data
                input_tensor = data['input']
                cond_images = data['sapines_input']
                
                # Encode inputs
                mv_latents, encoder_hidden_states = encode_input(
                    vae, image_encoder, input_tensor, data['UV_inital'], cond_images, opt, cfg=False
                )
                
                # Convert to appropriate dtype
                mv_latents = mv_latents.to(weight_dtype)
                encoder_hidden_states = encoder_hidden_states.to(weight_dtype)
                
                # Get batch dimensions
                batch_size, num_channels, height, width = mv_latents.shape
                
                # Sample noise and timesteps
                noise = torch.randn_like(mv_latents)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (batch_size,), device=input_tensor.device
                ).long()
                
                # Add noise to latents
                noisy_mv_latents = scheduler.add_noise(mv_latents, noise, timesteps)
                
                # Prepare rotary embeddings if needed
                image_rotary_emb = None
                if opt.use_rotary_positional_embeddings:
                    image_rotary_emb = prepare_rotary_positional_embeddings(
                        height=opt.height,
                        width=opt.width,
                        vae_scale_factor_spatial=8,
                        patch_size=opt.patch_size,
                        attention_head_dim=opt.attention_head_dim,
                        device=accelerator.device,
                        base_height=opt.height,
                        base_width=opt.width
                    )
                
                # Forward pass through model
                model_output = model(
                    hidden_states=noisy_mv_latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                
                # Calculate loss (DDPM v-prediction)
                model_pred = scheduler.get_velocity(model_output, noisy_mv_latents, timesteps)
                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                
                # Expand weights to match model prediction dimensions
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)
                
                # Calculate weighted MSE loss
                target = mv_latents
                loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
                loss = loss.mean()
                
                # Backward pass
                accelerator.backward(loss)
                total_loss += loss.detach()
                
                # Log training progress
                if accelerator.is_main_process:
                    print(f"[train] index: {i} epoch: {epoch} loss: {loss:.6f}")
                
                # Apply gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
                    global_step += 1
                
                # Log to wandb
                if accelerator.is_main_process and opt.wandb:
                    wandb.log({
                        "train_step_loss": loss.item(),
                        "train_step_lr": optimizer.param_groups[0]['lr']
                    })
                
                # Update model parameters
                optimizer.step()
            
            # Periodic logging and visualization
            if accelerator.is_main_process:
                if global_step % 400 == 0:
                    # Log memory usage
                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    print(
                        f"[INFO] {i}/{len(train_dataloader)} "
                        f"mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G "
                        f"step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}"
                    )
                    
                    # Create visualization pipeline
                    pipe = SamplesPipeline(
                        opt,
                        transformer=unwrap_model(model, accelerator),
                        vae=unwrap_model(vae, accelerator),
                        renderer=unwrap_model(renderer, accelerator),
                        scheduler=scheduler,
                        img_encoder=unwrap_model(image_encoder, accelerator),
                        guidance_scale=3.5,
                        vae_scale_factor=opt.vae_scaling_factor
                    )
                    
                    # Generate samples
                    out = pipe(
                        data=data, 
                        condition_img=data['sapines_input'], 
                        dtype=weight_dtype, 
                        height=opt.height, 
                        width=opt.width
                    )
                    
                    # Get images for visualization
                    gt_images = out['images_gt'].detach().cpu().numpy()
                    pred_images = out['images_pred'].detach().cpu().numpy()
                    
                    # Get VAE reconstruction
                    vae_latent = vae.encode(data['input'], data['UV_inital'])
                    vae_out = vae.decode_uv(vae_latent, data)
                    vae_images = vae_out['images_pred'].detach().cpu().numpy()
                    
                    # Save visualization
                    save_visualization(sample_folder, f"train_{global_step}", gt_images, pred_images, vae_images)
                
                # Save checkpoint periodically
                if global_step % opt.save_ckpt_steps == 0:
                    save_folder = os.path.join(opt.workspace, 'transformer')
                    os.makedirs(save_folder, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(model)
                    save_file(
                        unwrapped_model.state_dict(), 
                        os.path.join(save_folder, f"transformer.safetensors")
                    )
            
            # Periodic evaluation
            if global_step % opt.eval_steps == 0 and global_step != 0:
                model.eval()
                # Create evaluation pipeline
                pipe = SamplesPipeline(
                    opt,
                    transformer=unwrap_model(model, accelerator),
                    vae=unwrap_model(vae, accelerator),
                    renderer=unwrap_model(renderer, accelerator),
                    scheduler=scheduler,
                    img_encoder=unwrap_model(image_encoder, accelerator),
                    guidance_scale=3.5,
                    vae_scale_factor=opt.vae_scaling_factor
                )
                
                eval_loss = 0.0
                eval_count = 0
                
                # Evaluation loop
                with torch.no_grad():
                    for i, data in enumerate(test_dataloader):
                        # Get input data
                        input_tensor = data['input']
                        cond_images = data['sapines_input']
                        
                        # Encode inputs
                        mv_latents, encoder_hidden_states = encode_input(
                            vae, image_encoder, input_tensor, data['UV_inital'], cond_images, opt, cfg=False
                        )
                        mv_latents = mv_latents.to(weight_dtype)
                        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)
                        
                        # Sample noise and timesteps
                        noise = torch.randn_like(mv_latents)
                        timesteps = torch.randint(
                            0, scheduler.config.num_train_timesteps, (batch_size,), device=input_tensor.device
                        ).long()
                        noisy_mv_latents = scheduler.add_noise(mv_latents, noise, timesteps)
                        
                        # Prepare rotary embeddings if needed
                        image_rotary_emb = None
                        if opt.use_rotary_positional_embeddings:
                            image_rotary_emb = prepare_rotary_positional_embeddings(
                                height=opt.height,
                                width=opt.width,
                                vae_scale_factor_spatial=8,
                                patch_size=opt.patch_size,
                                attention_head_dim=opt.attention_head_dim,
                                device=accelerator.device,
                                base_height=opt.height,
                                base_width=opt.width
                            )
                        
                        # Forward pass
                        model_output = model(
                            hidden_states=noisy_mv_latents,
                            encoder_hidden_states=encoder_hidden_states,
                            timestep=timesteps,
                            image_rotary_emb=image_rotary_emb,
                            return_dict=False,
                        )[0]
                        
                        # Calculate loss
                        model_pred = scheduler.get_velocity(model_output, noisy_mv_latents, timesteps)
                        alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                        weights = 1 / (1 - alphas_cumprod)
                        
                        while len(weights.shape) < len(model_pred.shape):
                            weights = weights.unsqueeze(-1)
                        
                        target = mv_latents
                        loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
                        loss = loss.mean()
                        
                        eval_loss += loss.item()
                        eval_count += 1
                
                # Calculate average evaluation loss
                avg_eval_loss = eval_loss / eval_count if eval_count > 0 else 0
                
                # Log evaluation results
                if accelerator.is_main_process and opt.wandb:
                    wandb.log({
                        "eval_loss": avg_eval_loss,
                        "global_step": global_step
                    })
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Switch back to training mode
                model.train()
        
        # Calculate and log epoch-level metrics
        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            if opt.wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_total_loss": total_loss.item()
                })

    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    
    # Save final model
    if accelerator.is_main_process:
        save_folder = os.path.join(opt.workspace, 'transformer')
        os.makedirs(save_folder, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        save_file(
            unwrapped_model.state_dict(), 
            os.path.join(save_folder, f"transformer-final.safetensors")
        )
    
    # Finish wandb run
    if opt.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
