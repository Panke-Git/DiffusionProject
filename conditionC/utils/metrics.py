# coding=utf-8
"""
    @Project: 
    @Author: PyCharm
    @FileName： metrics.py
    @Date：2025/12/25 16:15
    @Email: None
"""

import torch
import torch.nn.functional as F
import math

def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (x.clamp(-1, 1) + 1.0) / 2.0

@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    # pred/target: [0,1]
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1,2,3])
    val = 10.0 * torch.log10(1.0 / (mse + eps))
    return val.mean()

def _gaussian_window(window_size: int, sigma: float, device):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g

def _create_ssim_kernel(window_size: int = 11, sigma: float = 1.5, channels: int = 3, device="cpu"):
    g1d = _gaussian_window(window_size, sigma, device)
    g2d = (g1d[:, None] * g1d[None, :]).unsqueeze(0).unsqueeze(0)
    kernel = g2d.repeat(channels, 1, 1, 1)  # (C,1,ws,ws)
    return kernel

@torch.no_grad()
def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
    """
    pred/target: [0,1], shape (B,C,H,W)
    """
    device = pred.device
    b, c, h, w = pred.shape
    kernel = _create_ssim_kernel(window_size, sigma, c, device=device)

    mu1 = F.conv2d(pred, kernel, padding=window_size//2, groups=c)
    mu2 = F.conv2d(target, kernel, padding=window_size//2, groups=c)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size//2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=window_size//2, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size//2, groups=c) - mu12

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
    return ssim_map.mean(dim=[1,2,3]).mean()