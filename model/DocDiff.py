import math
import numpy as np
from functools import partial
from inspect import isfunction

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# helpers (from your diffusion.py)
# -----------------------------
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    def noise():
        return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


# -----------------------------
# DocDiff-style prior: make J from condition
# Output to diffusion UNet: cat(J, x_noisy) -> 6ch
# -----------------------------
def _safe_minmax_1ch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min = x.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def _gaussian_kernel2d(ks: int, sigma: float, device, dtype):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def blur_rgb_depthwise(x: torch.Tensor, ks: int = 15, sigma: float = 3.0) -> torch.Tensor:
    b, c, h, w = x.shape
    kernel2d = _gaussian_kernel2d(ks, sigma, device=x.device, dtype=x.dtype)
    weight = kernel2d.view(1, 1, ks, ks).repeat(c, 1, 1, 1)  # (C,1,ks,ks)
    pad = ks // 2
    return F.conv2d(x, weight, bias=None, stride=1, padding=pad, groups=c)

class BetaPredictor(nn.Module):
    """
    condition (B,3,H,W) -> pred_beta (B,3,1,1)
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(base_ch * 2, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        feat = self.net(condition)
        pooled = self.pool(feat)
        beta = self.fc(pooled)                 # (B,3)
        return beta.view(beta.shape[0], 3, 1, 1)

class DocDiff6Adapter(nn.Module):
    """
    Build J using a DocDiff-like physics prior.
    Given: condition (B,3,H,W) in same scale you train with (often [0,1] or [-1,1])
    Returns:
        J (B,3,H,W), T_direct, T_scatter, pred_beta, A
    """
    def __init__(self, eps: float = 1e-6, clamp_J: bool = True,
                 A_blur_ks: int = 15, A_blur_sigma: float = 3.0,
                 beta_base_ch: int = 32):
        super().__init__()
        self.eps = float(eps)
        self.clamp_J = bool(clamp_J)
        self.A_blur_ks = int(A_blur_ks)
        self.A_blur_sigma = float(A_blur_sigma)
        self.beta_predictor = BetaPredictor(3, beta_base_ch)

    @torch.no_grad()
    def _default_depth(self, condition: torch.Tensor) -> torch.Tensor:
        # If you don't have depth label, use luminance proxy as (B,1,H,W)
        y = 0.299 * condition[:, 0:1] + 0.587 * condition[:, 1:2] + 0.114 * condition[:, 2:3]
        return _safe_minmax_1ch(y, eps=self.eps)

    def forward(self, condition: torch.Tensor, depth: torch.Tensor | None = None):
        assert condition.dim() == 4 and condition.shape[1] == 3, f"condition must be (B,3,H,W), got {tuple(condition.shape)}"
        B, _, H, W = condition.shape

        if depth is None:
            depth_1 = self._default_depth(condition)  # (B,1,H,W)
        else:
            assert depth.dim() == 4 and depth.shape[0] == B and depth.shape[2:] == (H, W)
            if depth.shape[1] == 3:
                depth_1 = depth.mean(dim=1, keepdim=True)
            elif depth.shape[1] == 1:
                depth_1 = depth
            else:
                raise ValueError("depth channel must be 1 or 3")
            depth_1 = _safe_minmax_1ch(depth_1, eps=self.eps)

        pred_beta = self.beta_predictor(condition)     # (B,3,1,1)
        depth_3 = depth_1.repeat(1, 3, 1, 1)           # (B,3,H,W)

        T_direct = torch.exp(-pred_beta * depth_3).clamp(0.0, 1.0)
        T_scatter = (1.0 - torch.exp(-pred_beta * depth_3)).clamp(0.0, 1.0)

        # A-map (blurred condition) like your utils.get_A() style, but pure torch
        A = blur_rgb_depthwise(condition, ks=self.A_blur_ks, sigma=self.A_blur_sigma)

        J = (condition - T_scatter * A) / (T_direct + self.eps)
        if self.clamp_J:
            J = J.clamp(condition.min().item(), condition.max().item())  # keep same scale range
        return J, T_direct, T_scatter, pred_beta, A


# -----------------------------
# GaussianDiffusion with prior hook
# - prior(condition) -> J
# - feed UNet: cat(J, x_noisy)  (6ch)
# -----------------------------
class GaussianDiffusionWithDocDiffPrior(nn.Module):
    """
    Drop-in replacement of your GaussianDiffusion (diffusion.py),
    but applies DocDiff prior to condition before feeding UNet.

    Expect x_in dict:
        x_in['target']: (B,3,H,W)
        x_in['input']:  (B,3,H,W)  condition
        optional x_in['depth']: (B,1/3,H,W) if you want depth-aware prior
    """
    def __init__(
        self,
        denoise_fn: nn.Module,
        image_size: int,
        channels: int = 3,
        loss_type: str = 'l1',
        conditional: bool = True,
        schedule_opt=None,
        prior: nn.Module | None = None,
        prior_trainable: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.conditional = conditional
        self.loss_type = loss_type
        self.prior = prior
        self.prior_trainable = prior_trainable

        if self.prior is not None and (not prior_trainable):
            for p in self.prior.parameters():
                p.requires_grad = False

        if schedule_opt is not None:
            pass

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end']
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _apply_prior(self, condition_x: torch.Tensor, depth: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """
        condition_x: (B,3,H,W)
        returns:
            cond_used: (B,3,H,W)  -> feed into UNet as condition
            aux dict
        """
        # FIX dtype mismatch: make sure inputs are float32 like model weights
        if condition_x.dtype != torch.float32:
            condition_x = condition_x.float()
        if depth is not None and depth.dtype != torch.float32:
            depth = depth.float()

        if self.prior is None:
            return condition_x, {}

        if (not self.prior_trainable) or (not self.training):
            with torch.no_grad():
                J, T_d, T_s, beta, A = self.prior(condition_x, depth=depth)
        else:
            J, T_d, T_s, beta, A = self.prior(condition_x, depth=depth)

        aux = {"J": J, "T_direct": T_d, "T_scatter": T_s, "pred_beta": beta, "A": A}
        return J, aux  # <-- 用 J 替换 condition

    def p_losses(self, x_in, noise=None):
        x_start = x_in['target']     # (B,3,H,W)
        condition_x = x_in.get('input', None)  # (B,3,H,W)
        depth = x_in.get('depth', None)        # optional

        b, c, h, w = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if not self.conditional or condition_x is None:
            x_recon = self.denoise_fn(x_noisy, t)
        else:
            cond_used, _aux = self._apply_prior(condition_x, depth=depth)
            # your UNet expects 6ch: cat(condition, x_noisy)
            x_recon = self.denoise_fn(torch.cat([cond_used, x_noisy], dim=1), t)

        loss = self.loss_func(noise, x_recon)
        return loss

    @torch.no_grad()
    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, depth=None):
        if condition_x is not None:
            cond_used, _aux = self._apply_prior(condition_x, depth=depth)
            eps = self.denoise_fn(torch.cat([cond_used, x], dim=1), t)
            x_recon = self.predict_start_from_noise(x, t=t, noise=eps)
        else:
            eps = self.denoise_fn(x, t)
            x_recon = self.predict_start_from_noise(x, t=t, noise=eps)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None, depth=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, depth=depth
        )
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))

        if not self.conditional:
            shape = x_in
            b = shape[0]
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in reversed(range(0, self.num_timesteps)):
                img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
            return img

        else:
            condition_x = x_in
            shape = condition_x.shape
            b = shape[0]
            img = torch.randn(shape, device=device)
            ret_img = condition_x
            for i in reversed(range(0, self.num_timesteps)):
                img = self.p_sample(
                    img,
                    torch.full((b,), i, device=device, dtype=torch.long),
                    condition_x=condition_x
                )
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
            return ret_img if continous else ret_img[-1]

    @torch.no_grad()
    def restore(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)


# -----------------------------
# schedule (copied from your diffusion.py)
# -----------------------------
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':
        betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas
