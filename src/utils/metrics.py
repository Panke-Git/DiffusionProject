"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT
    @FileName: metrics.py
    @Time: 2025/11/09
    @Email: None
"""
from __future__ import annotations
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_tm
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_tm

def to01(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]"""
    return torch.clamp((x + 1) / 2, 0, 1)

def psnr(x_hat01: torch.Tensor, x001: torch.Tensor) -> float:
    """PSNR（使用 torchmetrics 实现），输入需在 [0,1]"""
    # batched reduction='elementwise_mean' 等价：我们手动对 batch 求平均更直观
    val = psnr_tm(x_hat01, x001, data_range=1.0, dim=(1,2,3), reduction='elementwise_mean')
    return float(val.item())

def ssim(x_hat01: torch.Tensor, x001: torch.Tensor) -> float:
    val = ssim_tm(x_hat01, x001, data_range=1.0)
    return float(val.item())
