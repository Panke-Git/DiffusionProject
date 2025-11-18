"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT
    @FileName: ddpm_scheduler.py
    @Time: 2025/11/09
    @Email: None
"""
# diffusion/ddpm_scheduler.py
import torch


class DDPMNoiseScheduler:
    """
    只负责正向 q(x_t|x_0) 的采样和 alpha_cumprod 的计算。
    """

    def __init__(self,
                 timesteps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 device='cpu'):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

    def q_sample(self, x0, t, noise=None):
        """
        x0: [B, C, H, W]  (GT 图像 y)
        t:  [B] long      (0 ~ T-1)
        noise: [B, C, H, W]，如果为 None，则自动采样标准高斯噪声
        """
        if noise is None:
            noise = torch.randn_like(x0)

        a_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise

    def get_alpha_bar(self, t):
        """
        返回 a_bar(t)，shape: [B,1,1,1]
        """
        a_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return a_bar
