import math
from functools import partial
from inspect import isfunction

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from model.depth_estimator_admm import SceneDepthEstimatorADMM
from model.depth_guided_adaptive_regularization import DepthGuidedAdaptiveRegularizer


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
    V8: Baseline diffusion + ADMM pseudo-depth guided adaptive regularization.

    The denoising network stays the original baseline U-Net. ADMM is only used
    to estimate a depth prior from the conditional image for the additional loss.
    """

    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        depth_regularization_opt=None,
        admm_depth_opt=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.conditional = conditional
        self.loss_type = loss_type
        self.loss_is_normalized = True

        depth_regularization_opt = depth_regularization_opt or {}
        admm_depth_opt = admm_depth_opt or {}
        self.use_depth_regularization = bool(depth_regularization_opt.get('enabled', True))
        self.lambda_reg = float(depth_regularization_opt.get('lambda_reg', 0.05))

        self.depth_estimator = SceneDepthEstimatorADMM(
            alpha=float(admm_depth_opt.get('alpha', 0.15)),
            gamma=float(admm_depth_opt.get('gamma', 2.0)),
            mu=float(admm_depth_opt.get('mu', 0.5)),
            sigma=float(admm_depth_opt.get('sigma', 0.05)),
            max_iter=int(admm_depth_opt.get('max_iter', 20)),
            eps_stop=float(admm_depth_opt.get('eps_stop', 1e-4)),
            mip_kernel=int(admm_depth_opt.get('mip_kernel', 15)),
        )

        self.depth_regularizer = DepthGuidedAdaptiveRegularizer(
            lambda_tv=float(depth_regularization_opt.get('lambda_tv', 1.0)),
            lambda_edge=float(depth_regularization_opt.get('lambda_edge', 0.3)),
            depth_gain=float(depth_regularization_opt.get('depth_gain', 1.5)),
            edge_suppress_tau=float(depth_regularization_opt.get('edge_tau', 0.15)),
        )
        self.latest_reg_info = {}

        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

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

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), t))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
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
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, torch.full(
                    (b,), i, device=device, dtype=torch.long), condition_x=x)
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
        else:
            noise_pred = self.denoise_fn(torch.cat([cond_img, x_noisy], dim=1), t)

        diffusion_loss_sum = self.loss_func(noise, noise_pred)
        diffusion_loss = diffusion_loss_sum / int(b * c * h * w)

        total_loss = diffusion_loss
        reg_loss = x_start.new_tensor(0.0)
        reg_info = {
            'reg_adaptive_tv': x_start.new_tensor(0.0),
            'reg_edge_align': x_start.new_tensor(0.0),
        }

        if self.use_depth_regularization and self.lambda_reg > 0:
            cond_img_01 = torch.clamp((cond_img + 1.0) / 2.0, 0.0, 1.0)
            with torch.no_grad():
                depth_map = self.depth_estimator(cond_img_01).to(dtype=cond_img.dtype)

            x0_pred = self.predict_start_from_noise(x_noisy, t=t, noise=noise_pred)
            x0_pred = torch.clamp(x0_pred, -1., 1.)
            x0_pred_01 = (x0_pred + 1.0) / 2.0

            reg_loss, reg_info = self.depth_regularizer(
                pred_x0=x0_pred_01,
                depth=depth_map,
                guide=cond_img_01
            )
            total_loss = diffusion_loss + self.lambda_reg * reg_loss

        self.latest_reg_info = {
            'l_pix': diffusion_loss.detach().item(),
            'l_depth_reg': reg_loss.detach().item(),
            'l_total': total_loss.detach().item(),
            'l_adaptive_tv': reg_info['reg_adaptive_tv'].detach().item(),
            'l_edge_align': reg_info['reg_edge_align'].detach().item(),
            'lambda_depth_reg': self.lambda_reg,
        }

        return total_loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
