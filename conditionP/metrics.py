"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: metrics.py.py
    @Time: 2025/12/23 23:29
    @Email: None
"""
import torch
import torch.nn.functional as F


def to_01(x):
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def psnr(pred, target, eps=1e-8):
    pred = to_01(pred)
    target = to_01(target)
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.flatten(1).mean(1)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def _gaussian_kernel(ch, k=11, sigma=1.5, device="cpu", dtype=torch.float32):
    x = torch.arange(k, device=device, dtype=dtype) - k // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return k2d.repeat(ch, 1, 1, 1)


def ssim(pred, target, k=11, sigma=1.5, c1=0.01**2, c2=0.03**2):
    pred = to_01(pred)
    target = to_01(target)

    b, ch, h, w = pred.shape
    kernel = _gaussian_kernel(ch, k=k, sigma=sigma, device=pred.device, dtype=pred.dtype)

    mu_x = F.conv2d(pred, kernel, padding=k//2, groups=ch)
    mu_y = F.conv2d(target, kernel, padding=k//2, groups=ch)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=k//2, groups=ch) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=k//2, groups=ch) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=k//2, groups=ch) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return ssim_map.flatten(1).mean(1)
