import torch
from torch import nn
import torch.nn.functional as F
import pdb
from .lpips import LPIPS
from einops import rearrange
from .discriminator import weights_init, NLayerDiscriminator3D, NLayerDiscriminator2D
from core.model_config.VAE import Options

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1.0 - logits_real), dim=[1, 2, 3])
    loss_fake = torch.mean(F.relu(1.0 + logits_fake), dim=[1, 2, 3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow((x - y), 2)


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        opt: Options = None,
        disc_num_layers=4,
        disc_in_channels=3,
        use_actnorm=False,
        disc_loss="hinge",
        learn_logvar: bool = False,
        wavelet_weight=0.01
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.opt = opt

        self.wavelet_weight = wavelet_weight
        logvar_init = 0.0
        self.logvar = nn.Parameter(
            torch.full((), logvar_init), requires_grad=learn_logvar
        )

        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

        self.discriminator = NLayerDiscriminator2D(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = self.opt.disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = self.opt.disc_factor
        self.discriminator_weight = self.opt.disc_weight
        self.rec_loss = l1

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        layer = last_layer if last_layer is not None else self.last_layer[0]

        nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def calculate_adaptive_weight_inputs(self, inputs, nll_loss, g_loss, last_layer=None):
        pred = inputs['images_pred']
        nll_grads = torch.autograd.grad(nll_loss, pred, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, pred, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        posteriors,
        optimizer_idx,
        global_step,
        weights=None,
        last_layer=None,
        wavelet_coeffs=None
    ):
        bs = inputs['images_pred'].shape[0]
        t = inputs['images_pred'].shape[1]

        if optimizer_idx == 0:
            #rec_loss
            gt = rearrange(inputs['images_gt'], "b t c h w -> (b t) c h w").contiguous()
            pred = rearrange(inputs['images_pred'], "b t c h w -> (b t) c h w").contiguous()
            gt_masks = rearrange(inputs['masks_gt'], "b t c h w -> (b t) c h w").contiguous()
            alpha_pred = rearrange(inputs['alphas_pred'], "b t c h w -> (b t) c h w").contiguous()
            loss_l1 = self.rec_loss(pred*gt_masks, gt*gt_masks)

            if self.opt.lambda_lpips > 0:
                loss_lpips = self.lpips_loss(
                    F.interpolate(inputs['images_gt'].view(-1, 3, self.opt.output_size_h, self.opt.output_size_w) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                    F.interpolate(inputs['images_pred'].view(-1, 3, self.opt.output_size_h, self.opt.output_size_w) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                )
                loss_rec = loss_lpips*self.opt.lambda_lpips + loss_l1
                loss_lpips = torch.sum(loss_lpips) / loss_lpips.shape[0]
                loss_l1 = torch.sum(loss_l1) / loss_l1.shape[0]
                loss_rec = torch.sum(loss_rec) / loss_rec.shape[0]

            nll_loss = loss_rec / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights * nll_loss

            loss_kl = posteriors.kl()
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]

            loss_kl *= self.opt.lambda_kl

            if wavelet_coeffs:
                wl_loss_l2 = torch.sum(l1(wavelet_coeffs[0], wavelet_coeffs[1])) / bs
                wl_loss_l3 = torch.sum(l1(wavelet_coeffs[2], wavelet_coeffs[3])) / bs
                wl_loss = wl_loss_l2 + wl_loss_l3
            else:
                wl_loss = torch.tensor(0.0)

            pred = rearrange(inputs['images_pred'], "b t c h w -> b c t h w").contiguous()
            logits_fake = self.discriminator(pred)
            g_loss = -torch.mean(logits_fake)
            if global_step >= self.discriminator_iter_start:
                if self.disc_factor > 0.0:
                    d_weight = torch.tensor(1.0) * self.discriminator_weight
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)
            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = (
                weighted_nll_loss
                + loss_kl
                + d_weight * disc_factor * g_loss
                + self.wavelet_weight * wl_loss
            )

            loss_log = {
                'L1': loss_l1,
                'lpips': loss_lpips,
                'kl': loss_kl,
                'GAN_G': d_weight * disc_factor * g_loss,
                'loss': loss,
            }
            return loss, loss_log
        
        elif optimizer_idx == 1:
            logits_real = self.discriminator(inputs['images_gt'].permute(0, 2, 1, 3, 4).contiguous().detach())
            logits_fake = self.discriminator(inputs['images_pred'].permute(0, 2, 1, 3, 4).contiguous().detach())

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            loss_log = {
                'GAN_D': d_loss.clone().detach().mean()
            }
            return d_loss, loss_log
