"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT
    @FileName: ddpm_scheduler.py
    @Time: 2025/11/09
    @Email: None
"""
# ddpm_scheduler.py
import torch


class DDPMNoiseScheduler:
    """
    DDPM 噪声调度器：
      - 正向：q(x_t | x_0)
      - 反向：根据 ε_pred 计算 x_{t-1} 的采样（p_sample）
    """

    def __init__(self,
                 timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 device='cpu'):
        self.timesteps = timesteps
        self.device = torch.device(device)

        betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # \bar{α}_t
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]], dim=0
        )

        self.betas = betas                  # [T]
        self.alphas = alphas                # [T]
        self.alphas_cumprod = alphas_cumprod            # [T]
        self.alphas_cumprod_prev = alphas_cumprod_prev  # [T]

        # posterior variance: β̃_t
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # 避免数值问题
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        从 q(x_t | x_0) 采样：x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
        x0: [B, C, H, W]
        t:  [B] long
        noise: [B, C, H, W] or None
        """
        if noise is None:
            noise = torch.randn_like(x0)

        a_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        """
        由 x_t 和 ε 推回 x_0：
          x_0 = (x_t - √(1-ᾱ_t) ε) / √ᾱ_t
        """
        a_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1).clamp_min(1e-5)
        return (x_t - torch.sqrt(1.0 - a_bar) * eps) / torch.sqrt(a_bar)

    def q_posterior(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """
        q(x_{t-1} | x_t, x_0) 的均值和方差（闭式）
        返回:
          mean: [B,C,H,W]
          var:  [B,1,1,1]
        """
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        a_t = self.alphas[t].view(-1, 1, 1, 1)
        a_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        a_bar_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)

        # from Ho et al. appendix
        coef1 = torch.sqrt(a_bar_prev) * betas_t / (1.0 - a_bar_t)
        coef2 = torch.sqrt(a_t) * (1.0 - a_bar_prev) / (1.0 - a_bar_t)
        mean = coef1 * x0 + coef2 * x_t

        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        return mean, var

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor,
                 generator: torch.Generator = None):
        """
        单步反向采样：
          给定 x_t 和 模型预测的 ε_pred，采样 x_{t-1}
        t: [B]，batch 内所有元素应相同（同一个时间步）
        """
        # 先根据 ε_pred 反推出 x0_pred
        x0_pred = self.predict_start_from_noise(x_t, t, eps_pred)
        mean, var = self.q_posterior(x0_pred, x_t, t)

        # t=0 时不再加噪声
        # 假设整个 batch 的 t 都一样
        t_scalar = t[0].item()
        if t_scalar == 0:
            return mean

        if generator is None:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.randn_like(x_t, generator=generator)

        return mean + torch.sqrt(var) * noise

    def get_alpha_bar(self, t: torch.Tensor):
        """
        与之前接口兼容：返回 ᾱ_t 形状 [B,1,1,1]
        """
        a_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return a_bar
