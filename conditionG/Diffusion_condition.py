# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
# [MOD] Completed missing diffusion pieces + adapted to RGB conditional enhancement:
#       - Only GT (target) is noised
#       - Condition stays clean
#       - Model predicts epsilon for GT (3 channels)

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v: torch.Tensor, t: torch.Tensor, x_shape):
    """
    v: (T,)
    t: (B,)
    returns (B, 1, 1, 1...) broadcastable
    """
    device = t.device
    out = torch.gather(v, dim=0, index=t).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer_cond(nn.Module):
    """
    Train by predicting noise epsilon.
    Input:
      gt:   (B,3,H,W) in [-1,1]
      cond: (B,3,H,W) in [-1,1]
    """
    def __init__(self, model, beta_1: float, beta_T: float, T: int):
        super().__init__()
        self.model = model
        self.T = int(T)

        betas = torch.linspace(beta_1, beta_T, T).double()
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def q_sample(self, x0, t, noise):
        return extract(self.sqrt_alphas_bar, t, x0.shape) * x0 + extract(self.sqrt_one_minus_alphas_bar, t, x0.shape) * noise

    def forward(self, gt, cond):
        """
        returns MSE loss between predicted noise and true noise
        """
        B = gt.shape[0]
        t = torch.randint(low=0, high=self.T, size=(B,), device=gt.device, dtype=torch.long)
        noise = torch.randn_like(gt)

        x_t = self.q_sample(gt, t, noise)
        x_in = torch.cat([x_t, cond], dim=1)  # [MOD] 6 channels

        pred = self.model(x_in, t)
        return F.mse_loss(pred, noise)


class GaussianDiffusionSampler_cond(nn.Module):
    """
    DDPM sampler p(x_{t-1} | x_t, cond)
    """
    def __init__(self, model, beta_1: float, beta_T: float, T: int):
        super().__init__()
        self.model = model
        self.T = int(T)

        betas = torch.linspace(beta_1, beta_T, T).double()
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1.0)[:T]

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('alphas_bar_prev', alphas_bar_prev)

        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # posterior variance (beta_tilde)
        posterior_variance = betas * (1. - alphas_bar_prev) / (1. - alphas_bar)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))

        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_bar_prev) * torch.sqrt(alphas) / (1. - alphas_bar))

    def predict_x0_from_eps(self, x_t, t, eps):
        return (x_t - extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape) * eps) / extract(torch.sqrt(self.alphas_bar), t, x_t.shape)

    def p_mean_variance(self, x_t, cond, t):
        """
        x_t: (B,3,H,W) current noisy target
        cond:(B,3,H,W) condition
        """
        x_in = torch.cat([x_t, cond], dim=1)  # [MOD]
        eps = self.model(x_in, t)
        x0_pred = self.predict_x0_from_eps(x_t, t, eps)
        x0_pred = torch.clamp(x0_pred, -1., 1.)

        mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var, x0_pred

    @torch.no_grad()
    def forward(self, cond, x_T=None):
        """
        cond: (B,3,H,W)
        x_T: optional initial noise (B,3,H,W)
        returns x_0 in [-1,1]
        """
        B, C, H, W = cond.shape
        if x_T is None:
            x_t = torch.randn((B, 3, H, W), device=cond.device, dtype=cond.dtype)
        else:
            x_t = x_T

        for time_step in reversed(range(self.T)):
            t = cond.new_full((B,), time_step, dtype=torch.long)
            mean, var, log_var, x0_pred = self.p_mean_variance(x_t, cond, t)
            if time_step > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.exp(0.5 * log_var) * noise
            else:
                x_t = mean
        return torch.clamp(x_t, -1., 1.)
