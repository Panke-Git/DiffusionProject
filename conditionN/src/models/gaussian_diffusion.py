"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: gaussian_diffusion.py.py
    @Time: 2025/12/13 11:16
    @Email: None
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """
    根绝每个样本的时间步t[b], 从一条 1D 序列 a（长度 T）里取出对应的标量；
    从 1D tensor a 中按时间步 t 提取对应元素, 并 reshape 成可与 x 逐通道广播相加的形状:
        a: (T,)
        t: (B,)
        返回: (B, 1, 1, 1) 或 (B, 1, 1) 等, 维度与 x_shape 对齐
    """
    out = a.gather(-1, t.long())  # (B,)
    # reshape 为 (B, 1, 1, 1, ...)
    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)
    return out


class GaussianDiffusion(nn.Module):
    """
    DDPM 高斯前向扩散 + 反向采样核心:
    - 这里假设模型预测 eps (噪声)
    """

    def __init__(
            self,
            timesteps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 2e-2,
            beta_schedule: str = "linear",
            clip_x_start: bool = True,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.clip_x_start = clip_x_start
        # 得到一个长度为timesteps的1D向量;
        betas = self.make_beta_schedule(beta_schedule, beta_start, beta_end, timesteps)
        # 注册为 buffer 以便随模型一起搬到 GPU
        # 前向 schedule：
        self.register_buffer("betas", betas)  # 公式中的β
        self.register_buffer("alphas", 1.0 - betas)  # 公式中的1-β，也就是α
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))  # 从0~t步，所有α的累乘；
        self.register_buffer(
            "alphas_cumprod_prev",
            torch.cat(
                [torch.ones(1, device=betas.device),
                 torch.cumprod(1.0 - betas, dim=0)[:-1]],
                dim=0,
            ),
        )  # 构造一个带t=-1的起点；

        # 前向加噪 / 解析公式用：
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))  # 马尔可夫链的公式中用的，
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - self.alphas_cumprod),
        )  # 对1-α进行开放后累乘；也就是ε
        self.register_buffer(
            "sqrt_recip_alphas",
            torch.sqrt(1.0 / (1.0 - betas)),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas",
            torch.sqrt(1.0 / (1.0 - betas) - 1.0),
        )

        # posterior q(x_{t-1} | x_t, x_0) 的方差 & 对数方差
        # 后验 q(x_{t-1} | x_t, x_0)（采样用）：
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
        )
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(
                torch.clamp(
                    betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
                    min=1e-20,
                )
            ),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(1.0 - betas)
            / (1.0 - self.alphas_cumprod),
        )

    @staticmethod
    def make_beta_schedule(
            schedule: str, beta_start: float, beta_end: float, timesteps: int
    ) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Unknown beta schedule: {schedule}")

    # -------- 前向扩散 q(x_t | x_0) ----------
    def q_sample(
            self,
            x_start: torch.Tensor,
            t: torch.Tensor,
            noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        根据给定 x_0 和噪声 eps 生成 x_t:
            x_t = sqrt(a_bar_t) * x_0 + sqrt(1 - a_bar_t) * eps
        x_start: [-1,1]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_a_bar = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_a_bar = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_a_bar * x_start + sqrt_one_minus_a_bar * noise

    # -------- 一些辅助函数 ----------
    def predict_start_from_noise(self, x_t, t, noise, clip: bool = None):
        sqrt_a_bar = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_a_bar = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x0 = (x_t - sqrt_one_minus_a_bar * noise) / sqrt_a_bar

        if clip is None:
            clip = self.clip_x_start
        if clip:
            x0 = x0.clamp(-1.0, 1.0)
        return x0

    def q_posterior(
            self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 q(x_{t-1} | x_t, x_0) 的均值和方差:
            mu_tilde = coef1 * x_0 + coef2 * x_t
        """
        coef1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x_start + coef2 * x_t

        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    # -------- 反向采样 p_theta(x_{t-1} | x_t, cond) ----------
    @torch.no_grad()
    def p_mean_variance(
            self,
            model: nn.Module,
            x_t: torch.Tensor,
            cond: torch.Tensor,
            t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        给定当前 x_t 和 cond, 用模型预测 eps -> 得到 x_0 估计, 再计算 q_posterior.
        返回:
            model_mean, model_var, model_log_var, x0_pred
        """
        eps_pred = model(x_t, cond, t)  # 预测噪声
        x0_pred = self.predict_start_from_noise(x_t, t, eps_pred)
        mean, var, log_var = self.q_posterior(x0_pred, x_t, t)
        return mean, var, log_var, x0_pred

    @torch.no_grad()
    def p_sample(
            self,
            model: nn.Module,
            x_t: torch.Tensor,
            cond: torch.Tensor,
            t: torch.Tensor,
            deterministic: bool = False,
    ) -> torch.Tensor:
        """
        从 p_theta(x_{t-1} | x_t, cond) 采样一个 x_{t-1}
        """
        b = x_t.shape[0]
        model_mean, _, model_log_var, _ = self.p_mean_variance(model, x_t, cond, t)
        if deterministic or (t==0).all():
            return model_mean

        noise = torch.randn_like(x_t)
        return model_mean + torch.exp(0.5 * model_log_var) * noise

    @torch.no_grad()
    def p_sample_loop(
            self,
            model: nn.Module,
            cond: torch.Tensor,
            shape,
            device: torch.device,
            deterministic: bool = False,
    ) -> torch.Tensor:
        """
        从纯噪声开始, 迭代生成 x_0:
        - cond: (B, cond_channels, H, W)
        - shape: 生成的 x 的形状, 一般为 (B, C, H, W)
        """
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, cond, t, deterministic)

        return img

    @torch.no_grad()
    def p_sample_loop_from_x(
            self,
            model: nn.Module,
            cond: torch.Tensor,
            x_t: torch.Tensor,
            t_start: int,
            deterministic: bool = False,
    ) -> torch.Tensor:
        """
        从给定的 x_{t_start} 开始往回采样到 x_0:
            - x_t: (B,C,H,W)
            - t_start: 0 <= t_start < self.timesteps
        """
        device = x_t.device
        img = x_t
        b = x_t.shape[0]

        # 依次走 t_start, t_start-1, ..., 0
        for i in reversed(range(t_start + 1)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, cond, t, deterministic)

        return img

    @torch.no_grad()
    def sample_from_input(
            self,
            model: nn.Module,
            cond: torch.Tensor,
            t_start: int,
            deterministic: bool = False,
    ) -> torch.Tensor:
        """
        以 input 自身作为 x_0 的近似, 先加噪到 t_start, 再反向采样:
            - cond: (B,3,H,W), 既是条件图像, 也被当作近似 x_0_start
            - t_start: 0 ~ timesteps-1, 越大表示加的噪声越强
        """
        device = cond.device
        b, c, h, w = cond.shape

        # 限制 t_start 的范围
        t_start = int(max(0, min(t_start, self.timesteps - 1)))

        # 用 cond 近似 x_0, 先加噪得到 x_{t_start}
        t = torch.full((b,), t_start, device=device, dtype=torch.long)
        noise = torch.randn_like(cond)
        x_t = self.q_sample(cond, t, noise=noise)

        # 再从 x_{t_start} 反向采样到 x_0
        x0 = self.p_sample_loop_from_x(model, cond, x_t, t_start, deterministic)
        return x0

    @torch.no_grad()
    def sample(self, model: nn.Module, cond: torch.Tensor, deterministic: bool = False,) -> torch.Tensor:
        """
        对一批条件图像 cond 生成增强图像:
            cond: (B, 3, H, W)  [-1,1]
        返回:
            x0:   (B, 3, H, W)  [-1,1]
        """
        device = cond.device
        b, c, h, w = cond.shape
        x = self.p_sample_loop(model, cond, (b, c, h, w), device=device, deterministic=deterministic)
        return x

    @torch.no_grad()
    def ddim_sample_from_x(
            self,
            model,
            cond,
            x_t,
            t_start: int,
            steps: int = 50,
            eta: float = 0.0,
            clip_x0: bool = True,
    ):
        device = x_t.device
        b = x_t.size(0)

        # t_start -> 0 的 (steps+1) 个点，形成 steps 次更新
        times = torch.linspace(t_start, 0, steps + 1, device=device).round().long()
        times = torch.unique_consecutive(times)
        if times[-1].item() != 0:
            times = torch.cat([times, torch.zeros(1, device=device, dtype=torch.long)], dim=0)

        x = x_t
        for i in range(len(times) - 1):
            t = times[i].expand(b)
            t_prev = times[i + 1].expand(b)

            eps = model(x, cond, t)

            a_t = extract(self.alphas_cumprod, t, x.shape)
            a_prev = extract(self.alphas_cumprod, t_prev, x.shape)

            sqrt_a_t = a_t.sqrt()
            sqrt_1m_a_t = (1.0 - a_t).sqrt()

            x0 = (x - sqrt_1m_a_t * eps) / sqrt_a_t

            if clip_x0:
                x0 = x0.clamp(-1.0, 1.0)
                # 关键：clamp 后重算 eps，保持一致性
                eps = (x - sqrt_a_t * x0) / (sqrt_1m_a_t + 1e-8)

            if eta == 0.0:
                sigma = 0.0
                noise = 0.0
            else:
                sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(torch.clamp(1 - a_t / a_prev, min=0.0))
                noise = torch.randn_like(x)

            c = torch.sqrt(torch.clamp(1.0 - a_prev - sigma ** 2, min=0.0))
            x = a_prev.sqrt() * x0 + c * eps + sigma * noise

        return x

    @torch.no_grad()
    def sample_from_input_ddim(self, model, cond, t_start: int = 400, steps: int = 50, eta: float = 0.0):
        device = cond.device
        b = cond.size(0)
        t_start = int(max(0, min(t_start, self.timesteps - 1)))

        t = torch.full((b,), t_start, device=device, dtype=torch.long)
        noise = torch.randn_like(cond)
        x_t = self.q_sample(cond, t, noise=noise)  # 从 input 加噪到 t_start

        return self.ddim_sample_from_x(model, cond, x_t, t_start=t_start, steps=steps, eta=eta)

    @torch.no_grad()
    def ddim_sample(self, model, cond, steps: int = 50, eta: float = 0.0, clip_x0: bool = True):
        device = cond.device
        b, c, h, w = cond.shape

        # steps 次更新，所以用 steps+1 个时间点
        times = torch.linspace(self.timesteps - 1, 0, steps + 1, device=device).round().long()
        times = torch.unique_consecutive(times)
        if times[-1].item() != 0:
            times = torch.cat([times, torch.zeros(1, device=device, dtype=torch.long)], dim=0)

        x = torch.randn((b, c, h, w), device=device)

        for i in range(len(times) - 1):
            t = times[i].expand(b)
            t_prev = times[i + 1].expand(b)

            eps = model(x, cond, t)

            a_t = extract(self.alphas_cumprod, t, x.shape)
            a_prev = extract(self.alphas_cumprod, t_prev, x.shape)

            sqrt_a_t = a_t.sqrt()
            sqrt_1m_a_t = (1.0 - a_t).sqrt()

            # x0 pred
            x0 = (x - sqrt_1m_a_t * eps) / (sqrt_a_t + 1e-8)

            if clip_x0:
                x0 = x0.clamp(-1.0, 1.0)
                # 关键：clip 后重算 eps，保持 (x0, eps) 一致
                eps = (x - sqrt_a_t * x0) / (sqrt_1m_a_t + 1e-8)

            if eta == 0.0:
                sigma = 0.0
                noise = 0.0
            else:
                sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(torch.clamp(1 - a_t / a_prev, min=0.0))
                noise = torch.randn_like(x)

            c = torch.sqrt(torch.clamp(1.0 - a_prev - sigma ** 2, min=0.0))
            x = a_prev.sqrt() * x0 + c * eps + sigma * noise

        return x


if __name__ == "__main__":
    # 简单自测: 检查前向/反向的 shape 是否合理
    # 注意这里的 import 路径要和你的工程一致
    from cond_unet_ddpm import UNetConditional

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetConditional(
        in_channels=3,
        cond_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        dropout=0.0,
        attn_resolutions=(16,),
        num_heads=4,
        image_size=256,
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
    ).to(device)

    x0 = torch.randn(2, 3, 256, 256, device=device)
    cond = torch.randn(2, 3, 256, 256, device=device)
    t = torch.randint(0, diffusion.timesteps, (2,), device=device)

    x_t = diffusion.q_sample(x0, t)
    print("x_t shape:", x_t.shape)

    with torch.no_grad():
        eps_pred = model(x_t, cond, t)
        print("eps_pred shape:", eps_pred.shape)

        x0_pred = diffusion.predict_start_from_noise(x_t, t, eps_pred)
        print("x0_pred shape:", x0_pred.shape)
