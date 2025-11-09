"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT
    @FileName: scheduler.py
    @Time: 2025/11/09
    @Email: None
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

class Diffusion:
    """
    训练：提供 q_sample(x0, t) 构造 x_t（仅对 GT 加噪）。
    推理：提供 DDIM 采样（支持从 y 加噪起点的 img2img）。
    """
    def __init__(self, steps: int = 1000, schedule: str = "cosine", device: str | torch.device = "cuda"):
        self.steps = steps
        self.device = torch.device(device)
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, steps)
        elif schedule == "cosine":
            betas = self.cosine_beta_schedule(steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        self.betas = betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    @staticmethod
    def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
        # Nichol & Dhariwal (cosine schedule)
        t = torch.linspace(0, steps, steps + 1)
        f = torch.cos(((t / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-8, 0.999)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise"""
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.alphas_cumprod[t]  # (B,)
        while a_bar.dim() < x0.dim():
            a_bar = a_bar.unsqueeze(-1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    @torch.no_grad()
    def ddim_sample(self, model, y: torch.Tensor, image_size: int, steps: int = 30, eta: float = 0.0,
                    start_from_y: bool = True, start_t: Optional[int] = None, save_intermediate: Optional[Path] = None) -> torch.Tensor:
        """
        DDIM 采样：
        - start_from_y=True：使用 y 加噪作为起点（img2img 风格，保留结构）。
        - start_t：起始噪声步（越大改动越大，默认 self.steps-1）。
        - eta：0 为确定性 DDIM；>0 增加随机性。
        """
        device = y.device
        H = y.size(2); W = y.size(3); B = y.size(0)
        if start_t is None:
            start_t = self.steps - 1

        # 选择 steps 个时间步（均匀）
        ts = torch.linspace(0, start_t, steps, dtype=torch.long, device=device).flip(0)
        # 起点
        x = torch.randn(B, 3, H, W, device=device)
        if start_from_y:
            x = self.q_sample(y, torch.full((B,), start_t, device=device, dtype=torch.long), noise=torch.randn_like(y))

        for i, t in enumerate(ts):
            t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
            eps = model(x, y, t_batch)
            a_t = self.alphas_cumprod[t_batch]  # (B,)
            a_prev = self.alphas_cumprod_prev[t_batch]
            while a_t.dim() < x.dim():
                a_t = a_t.unsqueeze(-1)
                a_prev = a_prev.unsqueeze(-1)

            # 预测 x0 并更新到上一时刻
            x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
            dir_term = torch.sqrt(1 - a_prev - sigma ** 2) * eps
            noise = torch.randn_like(x) if i < len(ts) - 1 else torch.zeros_like(x)
            x = torch.sqrt(a_prev) * x0_pred + dir_term + sigma * noise

            if save_intermediate is not None:
                save_intermediate.mkdir(parents=True, exist_ok=True)
                save_image(torch.clamp((x0_pred + 1) / 2, 0, 1), save_intermediate / f"step_{i:03d}.png")

        return x0_pred
