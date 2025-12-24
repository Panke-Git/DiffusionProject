"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: diffusion_core_full.py.py
    @Time: 2025/12/23 23:21
    @Email: None
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else d() if callable(d) else d


def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    if repeat:
        n = torch.randn((1, *shape[1:]), device=device)
        return n.repeat(shape[0], *((1,) * (len(shape) - 1)))
    return torch.randn(shape, device=device)


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        return torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float32)
    if schedule == "cosine":
        steps = n_timestep + 1
        x = torch.linspace(0, n_timestep, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / n_timestep) + cosine_s) / (1 + cosine_s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0, 0.999)
    raise NotImplementedError("Unknown beta schedule: %s" % schedule)


class GaussianDiffusionCore(nn.Module):
    """
    保留“原作者风格”的 diffusion core：
      - 训练：DDPM epsilon prediction
      - 采样：DDPM / DDIM

    eps_model 签名：
        eps = eps_model(x_in, t, extra_cond)

    条件扩散：
        x_in = cat([cond, x_t], dim=1)
    """
    def __init__(
        self,
        eps_model,
        timesteps=1000,
        beta_schedule="linear",
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        loss_type="l1",
        clip_denoised=True
    ):
        super().__init__()
        self.eps_model = eps_model
        self.num_timesteps = int(timesteps)
        self.clip_denoised = bool(clip_denoised)

        betas = make_beta_schedule(
            beta_schedule, self.num_timesteps,
            linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        if loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif loss_type in ("l2", "mse"):
            self.loss_fn = F.mse_loss
        else:
            raise ValueError("loss_type must be 'l1' or 'l2'/'mse'")

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_x0_from_eps(self, x_t, t, eps):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def _eps_pred(self, x_t, t, cond=None, extra_cond=None):
        if exists(cond):
            x_in = torch.cat([cond, x_t], dim=1)
        else:
            x_in = x_t
        return self.eps_model(x_in, t, extra_cond)

    def p_mean_variance(self, x_t, t, cond=None, extra_cond=None):
        eps = self._eps_pred(x_t, t, cond=cond, extra_cond=extra_cond)
        x0_pred = self.predict_x0_from_eps(x_t, t, eps)
        if self.clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)
        model_mean, model_var, model_log_var = self.q_posterior(x0_pred, x_t, t)
        return model_mean, model_var, model_log_var, x0_pred, eps

    @torch.no_grad()
    def p_sample_ddpm(self, x_t, t, cond=None, extra_cond=None, repeat_noise=False):
        b = x_t.shape[0]
        model_mean, _, model_log_var, _, _ = self.p_mean_variance(x_t, t, cond=cond, extra_cond=extra_cond)
        noise = noise_like(x_t.shape, x_t.device, repeat=repeat_noise)
        nonzero_mask = (t != 0).float().reshape(b, *((1,) * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_var).exp() * noise

    @torch.no_grad()
    def sample_loop_ddpm(self, shape, cond=None, extra_cond=None, return_all=False):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        imgs = [img] if return_all else None

        for i in range(self.num_timesteps - 1, -1, -1):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample_ddpm(img, t, cond=cond, extra_cond=extra_cond)
            if return_all:
                imgs.append(img)

        return imgs if return_all else img

    @torch.no_grad()
    @torch.no_grad()
    def p_sample_ddim(self, x_t, t, t_next, cond=None, extra_cond=None, eta=0.0):
        eps = self._eps_pred(x_t, t, cond=cond, extra_cond=extra_cond)
        alpha = extract(self.alphas_cumprod, t, x_t.shape)

        x0_pred = (x_t - (1 - alpha).sqrt() * eps) / alpha.sqrt()

        if self.clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)
            eps = (x_t - alpha.sqrt() * x0_pred) / (1 - alpha).sqrt()

        if (t_next < 0).all():
            return x0_pred

        alpha_next = extract(self.alphas_cumprod, t_next, x_t.shape)
        sigma = eta * ((1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next)).sqrt()

        noise = torch.randn_like(x_t)
        dir_xt = (1 - alpha_next - sigma ** 2).sqrt() * eps
        x_next = alpha_next.sqrt() * x0_pred + dir_xt + sigma * noise
        return x_next

    @torch.no_grad()
    def sample_loop_ddim(self, shape, cond=None, extra_cond=None, steps=50, eta=0.0, return_all=False):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        imgs = [img] if return_all else None

        steps = int(steps)
        steps = max(2, min(steps, self.num_timesteps))

        times = torch.linspace(self.num_timesteps - 1, 0, steps, device=device).long()
        times_next = torch.cat([times[1:], torch.tensor([-1], device=device, dtype=torch.long)], dim=0)

        for i, j in zip(times.tolist(), times_next.tolist()):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            t_next = torch.full((shape[0],), j, device=device, dtype=torch.long)
            img = self.p_sample_ddim(img, t, t_next, cond=cond, extra_cond=extra_cond, eta=eta)
            if return_all:
                imgs.append(img)

        return imgs if return_all else img

    def p_losses(self, x_start, cond=None, extra_cond=None, noise=None, t=None, reduction="mean"):
        b = x_start.shape[0]
        if not exists(t):
            t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        eps_pred = self._eps_pred(x_noisy, t, cond=cond, extra_cond=extra_cond)
        return self.loss_fn(eps_pred, noise, reduction=reduction)

    def forward(self, x_start, cond=None, extra_cond=None, **kwargs):
        return self.p_losses(x_start, cond=cond, extra_cond=extra_cond, **kwargs)
