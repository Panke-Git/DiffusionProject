from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(
    schedule: str,
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    schedule = schedule.lower()
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    elif schedule == "cosine":
        # cosine schedule from Nichol & Dhariwal 2021
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(0.0001, 0.9999).float()
        return betas
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    clip_denoised: bool = True
    loss_type: str = "eps"  # eps


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """Extract values from a 1-D tensor for a batch of indices t."""
    out = a.gather(-1, t)
    return out.reshape((t.shape[0],) + (1,) * (len(x_shape) - 1))


class GaussianDiffusion(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        T = int(cfg.timesteps)
        betas = make_beta_schedule(cfg.beta_schedule, T, cfg.beta_start, cfg.beta_end)
        self.register_buffer("betas", betas)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32, device=betas.device), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        # log variance clipped for numerical stability
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ) * eps

    def p_mean_variance(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        eps_pred = model(x, t, cond)
        x0_pred = self.predict_x0_from_eps(x, t, eps_pred)
        if self.cfg.clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        model_mean = _extract(self.posterior_mean_coef1, t, x.shape) * x0_pred + _extract(
            self.posterior_mean_coef2, t, x.shape
        ) * x

        model_var = _extract(self.posterior_variance, t, x.shape)
        model_log_var = _extract(self.posterior_log_variance_clipped, t, x.shape)
        return model_mean, model_var, model_log_var, x0_pred

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        model_mean, _, model_log_var, _ = self.p_mean_variance(model, x, t, cond)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_var) * noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: Tuple[int, int, int, int], cond: torch.Tensor, device: torch.device):
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in reversed(range(self.cfg.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, cond)
        return img

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        cond: torch.Tensor,
        device: torch.device,
        steps: int = 50,
        eta: float = 0.0,
    ):
        """DDIM sampling (fast)."""
        b = shape[0]
        img = torch.randn(shape, device=device)

        # choose a subset of timesteps
        T = self.cfg.timesteps
        steps = int(steps)
        if steps <= 0:
            raise ValueError("steps must be > 0")
        times = torch.linspace(T - 1, 0, steps, device=device).long()

        for idx, i in enumerate(times):
            t = torch.full((b,), int(i.item()), device=device, dtype=torch.long)
            eps_pred = model(img, t, cond)
            x0_pred = self.predict_x0_from_eps(img, t, eps_pred)
            if self.cfg.clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            if idx == len(times) - 1:
                img = x0_pred
                continue

            t_prev = torch.full((b,), int(times[idx + 1].item()), device=device, dtype=torch.long)

            alpha_t = _extract(self.alphas_cumprod, t, img.shape)
            alpha_prev = _extract(self.alphas_cumprod, t_prev, img.shape)

            # DDIM parameters
            sigma = (
                eta
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt(1 - alpha_t / alpha_prev)
            )

            noise = torch.randn_like(img)
            # equation: x_{t-1} = sqrt(alpha_prev)*x0 + sqrt(1-alpha_prev - sigma^2)*eps + sigma*z
            pred_dir = torch.sqrt(1 - alpha_prev - sigma ** 2) * eps_pred
            img = torch.sqrt(alpha_prev) * x0_pred + pred_dir + sigma * noise

        return img

    def training_losses(self, model: nn.Module, x_start: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        eps_pred = model(x_noisy, t, cond)
        if self.cfg.loss_type == "eps":
            loss = F.mse_loss(eps_pred, noise)
        else:
            raise ValueError(f"Unknown loss_type: {self.cfg.loss_type}")
        return loss
