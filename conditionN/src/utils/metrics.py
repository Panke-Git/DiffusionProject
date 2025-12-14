"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: metrics.py
    @Time: 2025/12/13 22:31
    @Email: None
"""
# src/utils/metrics.py

import torch
import torch.nn.functional as F


def to_0_1(x):
    """
    将张量从 [-1,1] 或 [0,1] 范围映射到 [0,1]，并裁剪。
    """
    # 先假设输入在 [-1,1]，做线性变换
    x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def calculate_psnr(pred, target, eps=1e-8):
    """
    pred, target: (B, C, H, W)，值域可以是 [-1,1] 或 [0,1]
    返回: float，批次平均 PSNR（单位 dB）
    """
    pred = to_0_1(pred)
    target = to_0_1(target)

    # 按图片分别算 MSE 再取平均
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))  # (B,)
    psnr = -10.0 * torch.log10(mse + eps)                  # (B,)
    return psnr.mean().item()


def _gaussian_kernel(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    kernel_1d = g.unsqueeze(0)                   # (1, W)
    kernel_2d = kernel_1d.t() @ kernel_1d       # (W, W)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1,1,W,W)
    kernel = kernel_2d.repeat(channels, 1, 1, 1)     # (C,1,W,W) 用 groups=C 逐通道卷积
    return kernel


def calculate_ssim(pred, target, window_size=11, sigma=1.5):
    """
    pred, target: (B, C, H, W)，值域可以是 [-1,1] 或 [0,1]
    返回: float，批次平均 SSIM
    """
    pred = to_0_1(pred)
    target = to_0_1(target)

    b, c, h, w = pred.shape
    device = pred.device
    dtype = pred.dtype

    # 生成高斯窗口
    window = _gaussian_kernel(window_size, sigma, c, device, dtype)

    # 均值
    mu_x = F.conv2d(pred, window, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    # 方差和协方差
    sigma_x2 = F.conv2d(pred * pred, window, padding=window_size // 2, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=window_size // 2, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=window_size // 2, groups=c) - mu_xy

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )

    # 对空间、通道、batch 全局平均
    return ssim_map.mean().item()


