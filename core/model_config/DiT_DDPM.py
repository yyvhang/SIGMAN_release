import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional


@dataclass
class Options:
    '''
    VAE
    '''
    # Unet image input size
    input_size: int = 512
    input_size_h: int = 512
    input_size_w: int = 512
    splat_size: int = 128
    # gaussian render size
    output_size: int = 512
    output_size_h: int = 512
    output_size_w: int = 512
    # lpips loss weight
    lambda_lpips: float = 1.0
    # kl loss weight
    lambda_kl: float = 1e-6
    # discriminator loss weight
    disc_factor: float = 1.0
    disc_weight: float = 0.5
    # discriminator start iter
    disc_start: int = 0
    vae_scaling_factor: float = 0.6909025648433997
    self_attention_layers: int = 6
    latent_channels: int = 16
    vae_out_channels: int = 64

    '''
    text encoder
    '''
    text_encoder_path: str = './ckpt/T5'

    '''
    DiT
    '''
    height: int = 512
    width: int = 512
    vae_path: str = './ckpt/autoencoder/autoencoder.safetensors' 
    dit_path: str = '/ckpt/transformer/transformer.safetensors'
    num_attention_heads: int = 32
    attention_head_dim: int = 64
    text_embed_dim: int = 1536
    max_text_seq_length: int = 256
    in_channels: int = 16
    out_channels: Optional[int] = 16
    num_layers: int = 30
    sample_width: int = 64
    sample_height: int = 64
    sample_frames: int = 6
    patch_size: int = 2
    temporal_compression_ratio: int = 1
    use_rotary_positional_embeddings: bool = True
    use_learned_positional_embeddings: bool = False
    save_ckpt_steps: int = 200
    eval_steps: int = 3000
    noised_condition_dropout: float = 0.05
    condition_mode: str = 'cls'

    '''
    Rendering
    '''
    # fovy of the dataset
    FoVy: float = 0.8712626851529752
    FoVx: float = 0.8712626851529752
    # camera near plane
    znear: float = 0.1
    # camera far plane
    zfar: float = 100
    # number of all views (input + output)
    num_views: int = 10
    # number of views
    num_input_views: int = 6
    # camera radius
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False


    '''
    training
    '''
    # data mode (only support s3 now)
    data_mode: Literal['s3'] = 's3'
    # num workers
    num_workers: int = 8
    # workspace
    workspace: str = './workspace'
    # resume
    resume: Optional[str] = None
    # batch size (per-GPU)
    batch_size: int = 1
    # gradient checkpointing
    gradient_checkpointing: bool = True
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 30
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'bf16'
    # learning rate
    lr: float = 1e-4
    #lr_scheduler
    lr_scheduler: str = 'cosine'
    #warmup steps
    lr_warmup_steps: int = 2000
    #num_cycles
    lr_num_cycles: int = 1
    #power
    lr_power: float = 1.0
    # wandb
    wandb: bool = True
    wandb_name: str = 'DDPM_training'
    # overfit
    overfit: bool = False
    # rgb shuffle
    rgb_shuffle: bool = False
    #epoch
    num_epochs: int = 100
    #load_ckpt
    load_ckpt: bool = True

# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['vae'] = 'vae model'
config_defaults['vae'] = Options(
    input_size=256,
    splat_size=128,
    output_size=512, # render & supervise Gaussians at a higher resolution.
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
    num_epochs=250
)

config_doc['DiT'] = 'DiT model'
config_defaults['DiT'] = Options(
    num_epochs=100,
    batch_size=8,
    num_views=10,
    wandb=False,
    condition_mode='patch',
    height=512,
    width=512
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
