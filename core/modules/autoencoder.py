import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.gaussians.gs import GaussianRenderer
from core.model_config.VAE import Options

from typing import Optional, Tuple
from .vae_utils import VAE_Encoder3D_atten, VAE_Decoder2D, DiagonalGaussianDistribution, VAE_CrossAttention
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin

from .deformers import SMPLXDeformer


class Conv_VAE(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images. Used in
    [CogVideoX](https://github.com/THUDM/CogVideo).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to `1.15258426`):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True
    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 14,
        height: int = 32,
        width: int = 32,
        down_block_types: Tuple[str] = (
            "DownBlock3D",
            "DownBlock3D",
            "DownBlock3D",
            "DownBlock3D"
        ),
        up_block_types: Tuple[str] = (
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
        block_out_channels: Tuple[int] = (128, 256, 256, 512),
        decoder_block_out_channels: Tuple[int] = (128, 256, 256, 512),
        latent_channels: int = 16,
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 1,
        attention_head_dim: int = 64,
        num_attention_heads: int = 8,
        qk_norm: bool = True,
        attention_bias: bool = False,
        attention_out_bias: bool = True,
        self_attention_layers: int = 4,
        force_upcast: float = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
    ):
        super().__init__()

        self.encoder = VAE_Encoder3D_atten(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )

        self.height = height
        self.width = width
        self.uv_latent = nn.Parameter(torch.randn(1, height*width, block_out_channels[-1]))
        self.uv_encoding = nn.Sequential(
            nn.Conv2d(3,block_out_channels[-1],kernel_size=8,stride=8),
            nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6),
            nn.SiLU()
            )

        self.register_buffer("pos_embedding", self._get_sincos_pos_embedding(height*width, block_out_channels[-1]*2))
        self.attention = VAE_CrossAttention(
            height=height,
            width=width,
            query_dim=block_out_channels[-1]*2,
            cross_attention_dim=block_out_channels[-1],
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            self_attn_layers=self_attention_layers,
            dropout=0.1,
            bias=attention_bias,
            out_bias=attention_out_bias
        )
        self.projection = nn.Linear(block_out_channels[-1]*2, latent_channels*2)

        self.decoder = VAE_Decoder2D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=decoder_block_out_channels,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

    def _get_sincos_pos_embedding(self, n_position, d_hid):
        ''' Sinusoidal Position Embedding
        Args:
            n_position: sequence length
            d_hid: embedding dimension
        Returns:
            [1, n_position, d_hid]
        '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        # 添加batch维度并转换为tensor
        return torch.FloatTensor(sinusoid_table)[None, :, :]
    
    def encode(self, x: torch.Tensor, inital_uv):
        h = self.encoder(x)
        bs = h.shape[0]
        h = h.permute(0,2,3,4,1).view(bs,-1,h.shape[1]).contiguous()
        uv_encoding = self.uv_encoding(inital_uv).view(bs,-1,self.uv_latent.shape[-1])
        query = torch.cat([self.uv_latent.repeat(bs,1,1), uv_encoding], dim=-1)
        query_with_pos = query + self.pos_embedding
        atten_ = self.attention(query_with_pos, h)
        projection = self.projection(atten_)
        projection = projection.permute(0,2,1).view(bs, -1, self.height, self.width)
        if self.quant_conv is not None:
            projection = self.quant_conv(projection)
        posterior = DiagonalGaussianDistribution(projection)

        return posterior
    
    def decode(self, z: torch.Tensor):
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        decoded = self.decoder(z)
        return decoded

    def forward(self, x, inital_uv):
        # x: [B, Cin, H, W]
        # encoder
        posterior = self.encode(x, inital_uv)

        z = posterior.sample() # [B, 1024, C]
        dec = self.decode(z)

        return dec, posterior
    

DEFORMER = SMPLXDeformer(gender='neutral')

class VAE(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        # unet
        self.autoencoder = Conv_VAE(
                               in_channels=9, 
                               out_channels=opt.vae_out_channels,
                               latent_channels=opt.latent_channels, 
                               temporal_compression_ratio=opt.temporal_compression_ratio,
                               height=opt.input_size_h // 8,
                               width=opt.input_size_w // 8,
                               block_out_channels = (128, 256, 256, 512),
                               decoder_block_out_channels = (256, 512, 512, 1024),
                               attention_head_dim=64,
                               num_attention_heads = 8,
                               self_attention_layers=opt.self_attention_layers,
                               use_quant_conv=False,
                               use_post_quant_conv=False
                            ) #[B, C=64, H, W]
        out_channels = opt.vae_out_channels
        smplx_uv = torch.as_tensor(np.load('./core/modules/deformers/template/init_uv_smplx_thu.npy'))
        self.register_buffer('smplx_uvcoord', smplx_uv.unsqueeze(0)*2.-1.)

        t_pose_pcd = torch.as_tensor(np.load('./core/modules/deformers/template/init_pcd_smplx_thu.npy'))
        self.register_buffer('smplx_tpose_pcd', t_pose_pcd.unsqueeze(0), persistent=False)

        per_point_init_rot = torch.as_tensor(np.load('./core/modules/deformers/template/init_rot_smplx_thu.npy'))
        self.register_buffer('init_rot', per_point_init_rot, persistent=False)

        face_mask = torch.as_tensor(np.load('./core/modules/deformers/template/face_mask_thu.npy'))
        self.register_buffer('face_mask', face_mask.unsqueeze(0), persistent=False)

        hands_mask = torch.as_tensor(np.load('./core/modules/deformers/template/hands_mask_thu.npy'))
        self.register_buffer('hands_mask', hands_mask.unsqueeze(0), persistent=False)

        outside_mask = torch.as_tensor(np.load('./core/modules/deformers/template/outside_mask_thu.npy'))
        self.register_buffer('outside_mask', outside_mask.unsqueeze(0), persistent=False)

        self.sigmoid_saturation = 0.001

        self.decode_gaussian_geo = nn.Conv2d(out_channels // 2, 10, kernel_size=3, padding=1)
        self.decode_gaussian_rgb = nn.Conv2d(out_channels // 2, 3, kernel_size=3, padding=1)
        self.sigmoid = lambda x: torch.sigmoid(x)

        self.gs = GaussianRenderer(opt)

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict


    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
        
    def forward_gaussians(self, images, inital_uv, smpl_params=None):
        '''
        x: [b, c, h, w]
        '''
        bs, V, C, H, W = images.shape
        images = images.permute(0,2,1,3,4)
        x, posterior = self.autoencoder(images, inital_uv)
        x_geo, x_rgb = x.chunk(2,dim=1)
        gaussian_feats = self.decode_gaussian_geo(x_geo)
        rgb = self.decode_gaussian_rgb(x_rgb)
        opacity, offset, scale, rot = gaussian_feats.split([1,3,3,3], dim=1)
        opacity, rgb, scale, rot = self.sigmoid(opacity), self.sigmoid(rgb), self.sigmoid(scale), self.sigmoid(rot)

        select_coord = self.smplx_uvcoord.unsqueeze(1).repeat(bs, 1, 1, 1)
        select_coord = select_coord * torch.tensor([1.0, -1.0], device=select_coord.device).view(1, 1, 1, 2) # flip y, opengl to pytorch
        init_pcd = self.smplx_tpose_pcd.repeat(bs, 1, 1)
        output = torch.cat([opacity, offset, rgb, scale, rot], dim=1)
        output_attr = F.grid_sample(output, select_coord, mode='bilinear', padding_mode='border', align_corners=False).reshape(bs, 13, -1).permute(0, 2, 1)
        opacity, offset, rgbs, scale, rot = output_attr.split([1, 3, 3, 3, 3], dim=2)
        rgb_uv = rgb

        if self.sigmoid_saturation > 0:
            rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
        
        scale = (scale - 0.5) * 2
        rot = (rot - 0.5) * np.pi
        canon_pcd = init_pcd + offset
        defm_pcd, tfs = self.xyz_transform(canon_pcd, smpl_params=smpl_params, num_scenes=bs)

        return defm_pcd, opacity, rgbs, offset, scale, tfs, rot, posterior, rgb_uv
    
    def xyz_transform(self, xyz, smpl_params=None, num_scenes=1):
        assert xyz.dim() == 3
        deformer = DEFORMER
        deformer = deformer.to(xyz.device)
        deformer.prepare_deformer(smpl_params, num_scenes, device=xyz.device)
        xyz, tfs = deformer(xyz, mask=(self.face_mask+self.hands_mask+self.outside_mask), cano=False)
        xyz = xyz.squeeze(-2)
        return xyz, tfs

    
    def forward(self, data, step_ratio=1, test=False):

        results = {}

        images = data['input'] # [B, 4, 9, h, W], input features
        bs = images.shape[0]

        defm_pcd, opacity, rgbs, offset, scale, tfs, rot, posterior, rgb_uv = self.forward_gaussians(images, data['UV_inital'], data['smpl_params']) # [B, N, 14]
        R_delta = self.batch_rodrigues(rot.reshape(-1, 3))
        R = torch.bmm(self.init_rot.repeat(bs, 1, 1), R_delta)
        R_def = torch.bmm(tfs.flatten(0, 1)[:,:3,:3], R)
        normals = (R_def[:, :, -1]).reshape(bs, -1, 3)
        R_def_batch = R_def.reshape(bs, -1, 3, 3)

        gaussians = {
            'position': defm_pcd,
            'opacity': opacity,
            'scale': scale,
            'cov3d': R_def_batch,
            'rgb': rgbs,
        }
        bg_color = torch.ones(3, dtype=torch.float32, device=defm_pcd.device)
        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks
        try:
            results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
            pred_images = results['image'] # [B, V, C, output_size, output_size]
            pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

            with torch.no_grad():
                psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
                results['psnr'] = psnr
        except:
            print(data['item'])
            pred_images = torch.zeros_like(data['images_output'],device=defm_pcd.device)
            pred_alphas = torch.zeros_like(data['masks_output'],device=defm_pcd.device)
            results['psnr'] = 0

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['posterior'] = posterior

        results['images_gt'] = gt_images
        results['masks_gt'] = gt_masks

        return results
    
    def decode_uv(self, x, data, inference=False, caninocal=False):
        bs = x.shape[0]
        x_geo, x_rgb = x.chunk(2,dim=1)
        gaussian_feats = self.decode_gaussian_geo(x_geo)
        rgb = self.decode_gaussian_rgb(x_rgb)
        opacity, offset, scale, rot = gaussian_feats.split([1,3,3,3], dim=1)
        opacity, rgb, scale, rot = self.sigmoid(opacity), self.sigmoid(rgb), self.sigmoid(scale), self.sigmoid(rot)

        select_coord = self.smplx_uvcoord.unsqueeze(1).repeat(bs, 1, 1, 1)
        select_coord = select_coord * torch.tensor([1.0, -1.0], device=select_coord.device).view(1, 1, 1, 2) # flip y, opengl to pytorch
        init_pcd = self.smplx_tpose_pcd.repeat(bs, 1, 1)
 
        output = torch.cat([opacity, offset, rgb, scale, rot], dim=1)

        select_coord = select_coord.to(output.dtype)

        output_attr = F.grid_sample(output, select_coord, mode='bilinear', padding_mode='border', align_corners=False).reshape(bs, 13, -1).permute(0, 2, 1)
        opacity, offset, rgbs, scale, rot = output_attr.split([1, 3, 3, 3, 3], dim=2)
        rgb_uv = rgb
        if self.sigmoid_saturation > 0:
            rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
        
        scale = (scale - 0.5) * 2
        rot = (rot - 0.5) * np.pi
        canon_pcd = init_pcd + offset
        defm_pcd, tfs = self.xyz_transform(canon_pcd, smpl_params=data['smpl_params'], num_scenes=bs)
        if caninocal:
            defm_pcd = canon_pcd
        tfs = tfs.to(output.dtype)

        R_delta = self.batch_rodrigues(rot.reshape(-1, 3))
        R = torch.bmm(self.init_rot.repeat(bs, 1, 1), R_delta)
        R_def = torch.bmm(tfs.flatten(0, 1)[:,:3,:3], R)
        normals = (R_def[:, :, -1]).reshape(bs, -1, 3)
        R_def_batch = R_def.reshape(bs, -1, 3, 3)

        '''
        no smplx infer
        R_delta = self.batch_rodrigues(rot.reshape(-1, 3))
        R = torch.bmm(self.init_rot.repeat(bs, 1, 1), R_delta)
        normals = (R[:, :, -1]).reshape(bs, -1, 3)
        R_def_batch = R.reshape(bs, -1, 3, 3)
        defm_pcd = canon_pcd
        '''

        gaussians = {
            'position': defm_pcd,
            'opacity': opacity,
            'scale': scale,
            'cov3d': R_def_batch,
            'rgb': rgbs,
        }
        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=defm_pcd.device)
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)

        gaussian_points = gaussians['position']
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]


        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['gaussian_pts'] = gaussian_points
        if inference:
            return results
        #results['posterior'] = posterior
        results['rgb_uv'] = rgb_uv

        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        results['images_gt'] = gt_images
        results['masks_gt'] = gt_masks
        # metric
        with torch.no_grad():
            mse = torch.mean((pred_images.detach()*gt_masks - gt_images*gt_masks)**2)
            max_val = (pred_images.detach()*gt_masks).max()
            psnr = 10 * torch.log10(max_val**2 / mse)
            results['psnr'] = psnr
        
        return results

    def batch_rodrigues(self, rot_vecs, epsilon = 1e-8):
        ''' Calculates the rotation matrices for a batch of rotation vectors
            Parameters
            ----------
            rot_vecs: torch.tensor Nx3
                array of N axis-angle vectors
            Returns
            -------
            R: torch.tensor Nx3x3
                The rotation matrices for the given axis-angle parameters
        '''

        batch_size = rot_vecs.shape[0]
        device, dtype = rot_vecs.device, rot_vecs.dtype

        angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
        rot_dir = rot_vecs / angle

        cos = torch.unsqueeze(torch.cos(angle), dim=1)
        sin = torch.unsqueeze(torch.sin(angle), dim=1)

        # Bx1 arrays
        rx, ry, rz = torch.split(rot_dir, 1, dim=1)
        K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

        zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
            .view((batch_size, 3, 3))

        ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
        rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
        return rot_mat

    def get_model(self):
        #return [self.unet]
        return [self.autoencoder, self.decode_gaussian_geo, self.decode_gaussian_rgb]
    
    def get_last_layer(self):

        if hasattr(self.autoencoder.decoder.conv_out, "module"):
            return self.autoencoder.decoder.conv_out.module.weight
        else:
            return self.autoencoder.decoder.conv_out.weight
