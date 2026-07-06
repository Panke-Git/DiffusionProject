import math
from functools import partial
from inspect import isfunction

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from model.depth_estimator_admm import SceneDepthEstimatorADMM
from model.depth_guided_luminance_gate import DepthGuidedLuminanceGate


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


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
        return torch.randn(
            (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise():
        return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    """
    V10: Baseline diffusion + ADMM pseudo-depth + DLDG luminance gate.

    The U-Net is the original baseline denoiser. DLDG refines the predicted
    x0 image in Lab luminance space, then the gated x0 is used by both training
    loss and the reverse diffusion posterior.
    """

    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        depth_luminance_gate_opt=None,
        admm_depth_opt=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.conditional = conditional
        self.loss_type = loss_type
        self.loss_is_normalized = True

        depth_luminance_gate_opt = depth_luminance_gate_opt or {}
        admm_depth_opt = admm_depth_opt or {}
        self.use_depth_luminance_gate = bool(depth_luminance_gate_opt.get('enabled', True))

        self.depth_estimator = SceneDepthEstimatorADMM(
            alpha=float(admm_depth_opt.get('alpha', 0.15)),
            gamma=float(admm_depth_opt.get('gamma', 2.0)),
            mu=float(admm_depth_opt.get('mu', 0.5)),
            sigma=float(admm_depth_opt.get('sigma', 0.05)),
            max_iter=int(admm_depth_opt.get('max_iter', 20)),
            eps_stop=float(admm_depth_opt.get('eps_stop', 1e-4)),
            mip_kernel=int(admm_depth_opt.get('mip_kernel', 15)),
        )

        self.depth_lum_gate = DepthGuidedLuminanceGate(
            hidden_channels=int(depth_luminance_gate_opt.get('hidden_channels', 32)),
            alpha_init=float(depth_luminance_gate_opt.get('alpha_init', 0.05)),
            input_range=depth_luminance_gate_opt.get('input_range', '-1_1'),
            output_range=depth_luminance_gate_opt.get('output_range', '-1_1'),
        )
        self.latest_reg_info = {}

        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def estimate_depth(self, condition_x):
        condition_x_01 = torch.clamp((condition_x + 1.0) / 2.0, 0.0, 1.0)
        return self.depth_estimator(condition_x_01).to(dtype=condition_x.dtype)

    def apply_depth_luminance_gate(self, condition_x, x0_pred, depth_map):
        if not self.use_depth_luminance_gate:
            return x0_pred
        return self.depth_lum_gate(condition_x, x0_pred, depth_map).clamp(-1.0, 1.0)

    def noise_from_start(self, x_t, t, x_start):
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_alpha * x_start) / sqrt_one_minus_alpha.clamp(min=1e-12)

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
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

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
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, condition_depth=None):
        if condition_x is not None:
            noise_pred = self.denoise_fn(torch.cat([condition_x, x], dim=1), t)
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
            if clip_denoised:
                x_recon = x_recon.clamp(-1., 1.)
            if self.use_depth_luminance_gate:
                if condition_depth is None:
                    condition_depth = self.estimate_depth(condition_x)
                x_recon = self.apply_depth_luminance_gate(condition_x, x_recon, condition_depth)
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, t))
            if clip_denoised:
                x_recon = x_recon.clamp(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None, condition_depth=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            condition_x=condition_x,
            condition_depth=condition_depth,
        )
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))

        if not self.conditional:
            shape = x_in
            b = shape[0]
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, torch.full(
                    (b,), i, device=device, dtype=torch.long))
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
            return img
        else:
            x = x_in
            shape = x.shape
            b = shape[0]
            img = torch.randn(shape, device=device)
            ret_img = x
            condition_depth = self.estimate_depth(x) if self.use_depth_luminance_gate else None
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(
                    img,
                    torch.full((b,), i, device=device, dtype=torch.long),
                    condition_x=x,
                    condition_depth=condition_depth,
                )
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def restore(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['target']
        cond_img = x_in['input']
        b, c, h, w = x_start.shape

        t = torch.randint(
            0, self.num_timesteps, (b,),
            device=x_start.device
        ).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if not self.conditional:
            noise_pred = self.denoise_fn(x_noisy, t)
            diffusion_loss = self.loss_func(noise, noise_pred) / int(b * c * h * w)
            gated_delta = x_start.new_tensor(0.0)
        else:
            noise_pred = self.denoise_fn(torch.cat([cond_img, x_noisy], dim=1), t)
            raw_diffusion_loss = self.loss_func(noise, noise_pred) / int(b * c * h * w)

            if self.use_depth_luminance_gate:
                with torch.no_grad():
                    depth_map = self.estimate_depth(cond_img)

                x0_pred = self.predict_start_from_noise(x_noisy, t=t, noise=noise_pred)
                x0_pred = x0_pred.clamp(-1., 1.)
                x0_gated = self.apply_depth_luminance_gate(cond_img, x0_pred, depth_map)
                gated_delta = torch.mean(torch.abs(x0_gated - x0_pred))
                gated_noise_pred = self.noise_from_start(x_noisy, t, x0_gated)
                diffusion_loss = self.loss_func(noise, gated_noise_pred) / int(b * c * h * w)
            else:
                diffusion_loss = raw_diffusion_loss
                gated_delta = x_start.new_tensor(0.0)

        self.latest_reg_info = {
            'l_pix': diffusion_loss.detach().item(),
            'l_raw_pix': raw_diffusion_loss.detach().item() if self.conditional else diffusion_loss.detach().item(),
            'l_total': diffusion_loss.detach().item(),
            'l_dldg_delta': gated_delta.detach().item(),
            'alpha_dldg': torch.clamp(self.depth_lum_gate.alpha.detach(), 0.0, 1.0).item(),
        }

        return diffusion_loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
