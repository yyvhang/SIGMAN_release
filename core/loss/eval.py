import cv2
import numpy as np
import torch.nn as nn
from .lpips import LPIPS


def ssim(img1, img2):
    C1 = 0.01**2
    C2 = 0.03**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim_function(img1, img2):
    # [0,1]
    # ssim is the only metric extremely sensitive to gray being compared to b/w
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")

def trans(x):
    return x


def calculate_ssim(inputs1, inputs2):
    # videos [batch_size, timestamps, channel, h, w]

    assert inputs1.shape == inputs2.shape

    ssim_results = []
    for batch_num in range(inputs1.shape[0]):
        # get a video
        # video [timestamps, channel, h, w]
        img1 = inputs1[batch_num].numpy()
        img2 = inputs2[batch_num].numpy()

        ssim_results.append(calculate_ssim_function(img1, img2))

    ssim_results = np.array(ssim_results)

    return ssim_results.mean()


class eval_metrics(nn.Module):
    def __init__(self, opt, accelerator):
        super().__init__()
        self.opt = opt
        self.accelerator = accelerator
        self.lpips = LPIPS(net='alex')
        self.lpips.requires_grad_(False)

    def forward(self, out):
        lpips = self.lpips((out['images_gt'].view(-1, 3, self.opt.output_size_h, self.opt.output_size_w) * 2 - 1).detach(), 
            (out['images_pred'].view(-1, 3, self.opt.output_size_h, self.opt.output_size_w) * 2 - 1).detach(),
        ).mean()

        pnsr = out['psnr']
        gather_pred = self.accelerator.gather(out['images_pred'])
        gather_gt = self.accelerator.gather(out['images_gt'])
        ssim = calculate_ssim(gather_pred.view(-1, 3, self.opt.output_size_h, self.opt.output_size_w).detach().cpu(), gather_gt.view(-1, 3, self.opt.output_size_h, self.opt.output_size_w).detach().cpu())

        return lpips, pnsr, ssim