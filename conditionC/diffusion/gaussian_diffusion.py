# coding=utf-8
"""
    @Project: 
    @Author: PyCharm
    @FileName： gaussian_diffusion.py
    @Date：2025/12/25 16:15
    @Email: None
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999).float()

def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    # a: (T,), t: (B,)
    b = t.shape[0]
    out = a.gather(0, t).float()
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_schedule: str = "cosine"  # cosine / linear
    objective: str = "eps"         # eps (predict noise)
    p2_loss_weight_gamma: float = 0.0
    p2_loss_weight_k: float = 1.0

class GaussianDiffusion:
    def __init__(self, cfg: DiffusionConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        if cfg.beta_schedule == "cosine":
            betas = cosine_beta_schedule(cfg.timesteps)
        elif cfg.beta_schedule == "linear":
            betas = linear_beta_schedule(cfg.timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {cfg.beta_schedule}")

        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

        # optional p2 loss weight
        if cfg.p2_loss_weight_gamma > 0:
            self.p2_loss_weight = (cfg.p2_loss_weight_k + self.alphas_cumprod / (1 - self.alphas_cumprod)) ** (-cfg.p2_loss_weight_gamma)
        else:
            self.p2_loss_weight = torch.ones_like(self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_recip_alphas_cumprod, t, xt.shape) * xt - extract(self.sqrt_recipm1_alphas_cumprod, t, xt.shape) * eps

    def p_losses(self, model, x0: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise=noise)

        model_in = torch.cat([xt, cond], dim=1)  # conditional by concat
        pred = model(model_in, t)

        if self.cfg.objective == "eps":
            target = noise
        else:
            raise ValueError(f"Unknown objective: {self.cfg.objective}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss.mean(dim=[1,2,3])

        w = extract(self.p2_loss_weight, t, loss.shape)
        loss = loss * w.squeeze()
        return loss.mean()

    @torch.no_grad()
    def p_sample(self, model, xt: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # DDPM ancestral sampling: x_{t-1} = mean + sqrt(var)*z
        model_in = torch.cat([xt, cond], dim=1)
        eps = model(model_in, t)
        x0_pred = self.predict_x0_from_eps(xt, t, eps).clamp(-1, 1)

        mean = extract(self.posterior_mean_coef1, t, xt.shape) * x0_pred + extract(self.posterior_mean_coef2, t, xt.shape) * xt
        log_var = extract(self.posterior_log_variance_clipped, t, xt.shape)

        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().reshape(xt.shape[0], *((1,) * (len(xt.shape)-1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample_ddpm(self, model, cond: torch.Tensor) -> torch.Tensor:
        b, _, h, w = cond.shape
        xt = torch.randn((b, 3, h, w), device=cond.device)
        for i in reversed(range(self.cfg.timesteps)):
            t = torch.full((b,), i, device=cond.device, dtype=torch.long)
            xt = self.p_sample(model, xt, cond, t)
        return xt

    @torch.no_grad()
    def sample_ddim(self, model, cond: torch.Tensor, steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """
        Deterministic DDIM sampler for faster sampling.
        steps << timesteps.
        eta=0 => deterministic.
        """
        b, _, h, w = cond.shape
        x = torch.randn((b, 3, h, w), device=cond.device)

        T = self.cfg.timesteps
        times = torch.linspace(T - 1, 0, steps, device=cond.device).long()
        times_next = torch.cat([times[1:], torch.tensor([0], device=cond.device)])

        for t, t_next in zip(times, times_next):
            tt = torch.full((b,), int(t.item()), device=cond.device, dtype=torch.long)

            model_in = torch.cat([x, cond], dim=1)
            eps = model(model_in, tt)

            alpha = extract(self.alphas_cumprod, tt, x.shape)
            alpha_next = extract(self.alphas_cumprod, torch.full((b,), int(t_next.item()), device=cond.device, dtype=torch.long), x.shape)

            x0 = (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()
            x0 = x0.clamp(-1, 1)

            # DDIM update
            sigma = eta * ((1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = alpha_next.sqrt() * x0 + c * eps + sigma * noise

        return x