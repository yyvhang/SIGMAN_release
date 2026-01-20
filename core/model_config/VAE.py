import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional


@dataclass
class Options:
    ### model
    # Unet image input size
    input_size: int = 512
    input_size_h: int = 512
    input_size_w: int = 512
    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    latent_channels: int = 16
    vae_out_channels: int = 64
    splat_size: int = 64
    # gaussian render size
    output_size: int = 512
    output_size_h: int = 512 #376
    output_size_w: int = 512 #512
    self_attention_layers: int = 6

    vae_path: str = './ckpt/autoencoder/autoencoder.safetensors'

    data_mode: Literal['s3'] = 's3'
    # fovy of the dataset
    FoVy: float = 0.8712626851529752
    FoVx: float = 0.8712626851529752
    # camera near plane
    znear: float = 0.1
    # camera far plane
    zfar: float = 100
    # number of all views (input + output)
    num_views: int = 12
    # number of views
    num_input_views: int = 4
    # camera radius
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8
    # temporal compression ratio
    temporal_compression_ratio: float = 1

    ### training
    # workspace
    workspace: str = './workspace'
    # resume
    resume: Optional[str] = None
    # batch size (per-GPU)
    batch_size: int = 1
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 30
    # lpips loss weight
    lambda_lpips: float = 1.0
    # kl loss weight
    lambda_kl: float = 1e-6
    lambda_face: float = 5.0
    # discriminator loss weight
    disc_factor: float = 1.0
    disc_weight: float = 1000
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'bf16'
    # learning rate
    lr: float = 3e-6   #1e-5 #4e-4  
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5

    ### testing
    # test image path
    test_path: Optional[str] = None

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False
    # discriminator start iter
    disc_start: int = 50000000
    # wandb
    wandb: bool = True
    wandb_name: str = 'vae'
    # overfit
    overfit: bool = False
    # rgb shuffle
    rgb_shuffle: bool = False

# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['vae_s'] = 'vae model'
config_defaults['vae_s'] = Options(
    input_size=256,
    splat_size=128,
    output_size=512, # render & supervise Gaussians at a higher resolution.
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
    num_epochs=250
)

config_doc['vae_b'] = 'big model with higher resolution'
config_defaults['vae_b'] = Options(
    input_size=512,
    splat_size=128,
    output_size=512, # render & supervise Gaussians at a higher resolution.
    num_views=10,
    num_input_views=6,
    temporal_compression_ratio=1,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
    num_epochs=100,
    lambda_lpips=1.0,
    wandb=False,
    rgb_shuffle=False
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
