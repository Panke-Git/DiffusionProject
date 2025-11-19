"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT
    @FileName: train_utils.py
    @Time: 2025/11/09
    @Email: None
"""
from __future__ import annotations
import os, random
import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def tensor_to_01(x: torch.Tensor) -> torch.Tensor:
    """
    把 [-1, 1] 映射到 [0, 1]
    """
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


def psnr_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    pred, target: [B, C, H, W] in [-1,1]
    返回一个 batch 的平均 PSNR (dB)
    """
    pred_01 = tensor_to_01(pred)
    target_01 = tensor_to_01(target)
    mse = F.mse_loss(pred_01, target_01, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # [B]
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr.mean().item()


def _gaussian_kernel(window_size: int, sigma: float,
                     channels: int, device: torch.device) -> torch.Tensor:
    """
    生成 2D 高斯卷积核，用于 SSIM 计算
    """
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.view(1, 1, -1) * g.view(1, -1, 1)  # [1,1,window,window]
    kernel_2d = kernel_2d.unsqueeze(0)                # [1,1,window,window]
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)   # [C,1,window,window]
    return kernel_2d


def ssim_batch(pred: torch.Tensor, target: torch.Tensor,
               window_size: int = 11, sigma: float = 1.5) -> float:
    """
    pred, target: [B, C, H, W] in [-1,1]
    返回一个 batch 的平均 SSIM
    """
    pred_01 = tensor_to_01(pred)
    target_01 = tensor_to_01(target)

    B, C, H, W = pred_01.shape
    device = pred_01.device
    window = _gaussian_kernel(window_size, sigma, C, device=device)
    padding = window_size // 2

    mu1 = F.conv2d(pred_01, window, padding=padding, groups=C)
    mu2 = F.conv2d(target_01, window, padding=padding, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_01 * pred_01, window, padding=padding, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target_01 * target_01, window, padding=padding, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred_01 * target_01, window, padding=padding, groups=C) - mu1_mu2

    L = 1.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()

