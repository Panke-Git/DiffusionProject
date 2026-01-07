import math
from typing import Tuple
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    """pred/target: (B, C, H, W) in [0, 1]. Returns (B,) PSNR."""
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.mean(dim=(1, 2, 3))
    psnr_val = 10.0 * torch.log10((data_range ** 2) / (mse + eps))
    return psnr_val


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_1d = g.unsqueeze(0)  # (1, W)
    kernel_2d = kernel_1d.t() @ kernel_1d
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute SSIM for batches of images.
    pred/target: (B, C, H, W) in [0, 1]
    Returns: (B,) SSIM
    """
    assert pred.shape == target.shape
    B, C, H, W = pred.shape
    device = pred.device
    dtype = pred.dtype

    kernel = _gaussian_kernel(window_size, sigma, device=device, dtype=dtype)
    kernel = kernel.view(1, 1, window_size, window_size)
    kernel = kernel.repeat(C, 1, 1, 1)  # (C,1,ws,ws)

    # reflect padding for stability
    pad = window_size // 2
    pred_pad = F.pad(pred, (pad, pad, pad, pad), mode="reflect")
    target_pad = F.pad(target, (pad, pad, pad, pad), mode="reflect")

    mu1 = F.conv2d(pred_pad, kernel, groups=C)
    mu2 = F.conv2d(target_pad, kernel, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_pad * pred_pad, kernel, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target_pad * target_pad, kernel, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred_pad * target_pad, kernel, groups=C) - mu1_mu2

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # SSIM map
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + eps)

    # average over C,H,W
    return ssim_map.mean(dim=(1, 2, 3))
