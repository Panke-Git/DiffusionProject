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
                    start_from_y: bool = True, start_t: Optional[int] = None,
                    save_intermediate: Optional[Path] = None) -> torch.Tensor:
        """
        返回 x0（[-1,1]），尺寸与 y 一致。
        """
        device = y.device
        B, C, H, W = y.shape
        if start_t is None:
            start_t = self.steps - 1

        # --- 正确生成严格递减且包含 0 的时间步 ---
        ts = torch.linspace(start_t, 0, steps, device=device)  # float
        ts = torch.round(ts).long()
        ts = torch.unique_consecutive(ts)  # 去重，仍然递减
        if ts[-1].item() != 0:
            ts = torch.cat([ts, torch.zeros(1, device=device, dtype=torch.long)], dim=0)
        steps = ts.numel()

        # --- 起点：从 y 加噪或纯噪 ---
        if start_from_y:
            t0 = torch.full((B,), start_t, device=device, dtype=torch.long)
            x = self.q_sample(y, t0, noise=torch.randn_like(y))
        else:
            x = torch.randn(B, 3, H, W, device=device)

        x0_pred = None
        for i, t in enumerate(ts):
            t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)

            # 关键：FP32 做所有公式，避免/√ā_t 下溢
            eps = model(x, y, t_batch).float()

            a_t = self.alphas_cumprod[t_batch].float().clamp_min(1e-5)
            a_prev = self.alphas_cumprod_prev[t_batch].float()
            # 边界：t=0 的前一项就是 1
            a_prev = torch.where(t_batch == 0, torch.ones_like(a_prev), a_prev).clamp_min(1e-5)

            while a_t.dim() < x.dim():
                a_t = a_t.unsqueeze(-1)
                a_prev = a_prev.unsqueeze(-1)

            # 预测 x0
            x0_pred = (x.float() - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            x0_pred = x0_pred.clamp(-1, 1)

            # 最后一步（t==0）直接返回 x0
            if i == steps - 1:
                if save_intermediate is not None:
                    from torchvision.utils import save_image
                    save_intermediate.mkdir(parents=True, exist_ok=True)
                    save_image(((x0_pred + 1) / 2).clamp(0, 1), save_intermediate / f"step_{i:03d}.png")
                return x0_pred

            # 计算 DDIM 过渡（带 clamp）
            if eta == 0.0:
                sigma = torch.zeros_like(a_t)
            else:
                sigma = eta * torch.sqrt(((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).clamp_min(0))
            coef = (1 - a_prev - sigma ** 2).clamp_min(0).sqrt()
            noise = torch.randn_like(x)

            x = torch.sqrt(a_prev) * x0_pred + coef * eps + sigma * noise
            x = x.clamp(-1, 1)

            if save_intermediate is not None:
                from torchvision.utils import save_image
                save_intermediate.mkdir(parents=True, exist_ok=True)
                save_image(((x0_pred + 1) / 2).clamp(0, 1), save_intermediate / f"step_{i:03d}.png")

        return x0_pred

