"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: MIPTVDepthEstimator.py
    @Time: 2026/1/15 23:44
    @Email: None
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.get_A_utils import get_A

import numpy as np
import torch
from torch import nn
import math

class Swish(nn.Module):
    """
    ### Swish actiavation function
    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb


class ResidualBlock(nn.Module):
    """
    ### Residual block
    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float = 0.1,
        is_noise: bool = True,
    ):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.is_noise = is_noise
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # Group normalization and the second convolution layer

        self.act2 = Swish()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        if self.is_noise:
            self.time_emb = nn.Linear(time_channels, out_channels)
            self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(x))
        # Add time embeddings
        if self.is_noise:
            h += self.time_emb(self.time_act(t))[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        is_noise: bool = True,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels, out_channels, time_channels, is_noise=is_noise
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        return x


class UpBlock(nn.Module):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        is_noise: bool = True,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(
            in_channels + out_channels, out_channels, time_channels, is_noise=is_noise
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int, is_noise: bool = True):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels, n_channels, time_channels, is_noise=is_noise
        )
        self.dia1 = nn.Conv2d(
            n_channels, n_channels, 3, 1, dilation=2, padding=get_pad(16, 3, 1, 2)
        )
        self.dia2 = nn.Conv2d(
            n_channels, n_channels, 3, 1, dilation=4, padding=get_pad(16, 3, 1, 4)
        )
        self.dia3 = nn.Conv2d(
            n_channels, n_channels, 3, 1, dilation=8, padding=get_pad(16, 3, 1, 8)
        )
        self.dia4 = nn.Conv2d(
            n_channels, n_channels, 3, 1, dilation=16, padding=get_pad(16, 3, 1, 16)
        )
        self.res2 = ResidualBlock(
            n_channels, n_channels, time_channels, is_noise=is_noise
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.dia1(x)
        x = self.dia2(x)
        x = self.dia3(x)
        x = self.dia4(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Denoise_UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(
        self, input_channels, output_channels, n_channels, ch_mults, n_blocks, is_noise
    ):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(
            input_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.is_noise = is_noise
        if is_noise:
            self.time_emb = TimeEmbedding(n_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels, out_channels, n_channels * 4, is_noise=is_noise
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, is_noise=False)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = n_channels * ch_mults[i]
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels, out_channels, n_channels * 4, is_noise=is_noise
                    )
                )
            # Final block to reduce the number of channels
            in_channels = n_channels * (ch_mults[i - 1] if i >= 1 else 1)
            up.append(
                UpBlock(in_channels, out_channels, n_channels * 4, is_noise=is_noise)
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.act = Swish()
        self.final = nn.Conv2d(
            in_channels, output_channels, kernel_size=(3, 3), padding=(1, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor = torch.tensor([0]).cuda()):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        if self.is_noise:
            t = self.time_emb(t)
        else:
            t = None
        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                # print(x.shape, s.shape)
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(x))


class Beta_UNet(nn.Module):
    def __init__(self, input_channels, output_channels, n_channels, ch_mults, n_blocks):
        super().__init__()
        is_noise = False
        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(
            input_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels, out_channels, n_channels * 4, is_noise=is_noise
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, is_noise=False)
        self.act = Swish()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.transform = nn.Sequential(nn.Linear(128, 3), Swish(), nn.Linear(3, 3))

    def forward(self, x: torch.Tensor):
        t = None
        x = self.image_proj(x)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)
        x = self.middle(x, t)
        x = torch.sigmoid(self.transform(self.pool(x).squeeze()))
        return x.unsqueeze(-1).unsqueeze(-1)


class DocDiff(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        n_channels,
        ch_mults,
        n_blocks,
    ):
        super(DocDiff, self).__init__()
        self.beta_predictor = Beta_UNet(3, 3, n_channels, ch_mults, n_blocks)
        self.denoiser = Denoise_UNet(
            12, 3, n_channels, ch_mults, n_blocks, is_noise=True
        )

    def forward(self, x, condition, hist, depth, t, diffusion):
        pred_beta = self.beta_predictor(condition)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        T_direct = torch.clamp((torch.exp(-pred_beta * depth)), 0, 1)
        T_scatter = torch.clamp((1 - torch.exp(-pred_beta * depth)), 0, 1)
        atm_light = [get_A(item) for item in condition]
        atm_light = torch.stack(atm_light).to(x.device)
        J = torch.clamp(((condition - T_scatter * atm_light) / T_direct), 0, 1)

        noisy_image, noise_ref = diffusion.noisy_image(t, x)
        denoised_J = self.denoiser(
            torch.cat((noisy_image, condition.clone().detach(), J, hist), dim=1), t
        )
        return J, noise_ref, denoised_J, T_direct, T_scatter

class DocPriorWithMIPTV(nn.Module):
    """
    物理先验模块（从 DocDiff 提取改造）：
      输入: x, condition (可选 t，但这里不使用)
      输出: J，保证 [B,3,256,256]
    """

    def __init__(self, n_channels, ch_mults, n_blocks,
                 force_size=256, eps=1e-6,
                 depth_kwargs=None):
        super().__init__()
        self.beta_predictor = Beta_UNet(3, 3, n_channels, ch_mults, n_blocks)
        self.depth_estimator = MIPTVDepthEstimator(**(depth_kwargs or {}))
        self.force_size = force_size
        self.eps = eps

    def forward(self, x, condition, t=None):
        # # 1) 强制输入变成 256×256（如果你上游保证就是 256，可改成 assert 更干净）
        # if self.force_size is not None:
        #     target = (self.force_size, self.force_size)
        #     if x.shape[-2:] != target:
        #         x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        #     if condition.shape[-2:] != target:
        #         condition = F.interpolate(condition, size=target, mode="bilinear", align_corners=False)

        # 2) beta（每通道一个标量）：[B,3,1,1]
        pred_beta = self.beta_predictor(condition)


        depth = self.depth_estimator(condition)

        T_direct = torch.clamp(torch.exp(-pred_beta * depth), 0.0, 1.0)
        T_scatter = torch.clamp(1.0 - torch.exp(-pred_beta * depth), 0.0, 1.0)

        atm_light = torch.stack([get_A(item) for item in condition]).to(condition.device)

        J = (condition - T_scatter * atm_light) / (T_direct + self.eps)
        J = torch.clamp(J, 0.0, 1.0)
        return J

def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


# =========================
# MIPTVDepthEstimator (your code)
# =========================
class MIPTVDepthEstimator(nn.Module):
    """
    Depth estimator for SeaDiff replacement:
    1) Coarse normalized depth: N_hat = Fs(|D_mip| + R)
       where D_mip = local_max(R) - local_max(max(G,B))
    2) Weighted TV-L2 smoothing (Scheme A): W fixed from grad(N_hat)
       N = argmin ||N - N_hat||_2^2 + alpha * || W ∘ ∇N ||_1
    Output: D(x) = N in [0,1], shape (B,1,H,W)
    """

    def __init__(
        self,
        mip_kernel: int = 31,
        p_low: float = 0.01,
        p_high: float = 0.99,
        alpha: float = 0.15,
        mu: float = 10.0,
        sigma_mode: str = "median",
        sigma_fixed: float = 0.02,
        pd_iters: int = 30,
        tau: float = 0.125,
        sigma_pd: float = 0.125,
        theta: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert mip_kernel % 2 == 1, "mip_kernel should be odd"
        assert 0.0 <= p_low < p_high <= 1.0
        self.mip_kernel = mip_kernel
        self.p_low = p_low
        self.p_high = p_high
        self.alpha = alpha
        self.mu = mu
        self.sigma_mode = sigma_mode
        self.sigma_fixed = sigma_fixed
        self.pd_iters = pd_iters
        self.tau = tau
        self.sigma_pd = sigma_pd
        self.theta = theta
        self.eps = eps

    @staticmethod
    def _forward_grad(u: torch.Tensor):
        """Forward differences: returns dx, dy with same shape as u."""
        dx = u[..., :, 1:] - u[..., :, :-1]
        dy = u[..., 1:, :] - u[..., :-1, :]

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return dx, dy

    @staticmethod
    def _divergence(px: torch.Tensor, py: torch.Tensor):
        """Divergence of dual field (px, py) with backward differences."""
        div_x = px[..., :, :] - F.pad(px[..., :, :-1], (1, 0, 0, 0))
        div_y = py[..., :, :] - F.pad(py[..., :-1, :], (0, 0, 1, 0))
        return div_x + div_y

    def _robust_normalize(self, x: torch.Tensor):
        """Per-image robust normalization to [0,1] using percentiles."""
        B = x.shape[0]
        flat = x.view(B, -1)

        lo = torch.quantile(flat, self.p_low, dim=1, keepdim=True)
        hi = torch.quantile(flat, self.p_high, dim=1, keepdim=True)

        lo = lo.view(B, 1, 1, 1)
        hi = hi.view(B, 1, 1, 1)

        x = (x - lo) / (hi - lo + self.eps)
        return x.clamp(0.0, 1.0)

    def _estimate_sigma(self, grad_mag: torch.Tensor):
        """Estimate sigma per-image from grad magnitude."""
        if self.sigma_mode == "fixed":
            return torch.full_like(grad_mag[:, :, :1, :1], float(self.sigma_fixed))

        B = grad_mag.shape[0]
        flat = grad_mag.view(B, -1)
        if self.sigma_mode == "mean":
            s = flat.mean(dim=1, keepdim=True)
        else:
            s = torch.quantile(flat, 0.5, dim=1, keepdim=True)

        s = s.view(B, 1, 1, 1).clamp_min(self.eps)
        return s

    @torch.no_grad()
    def forward(self, Iy: torch.Tensor) -> torch.Tensor:
        """
        Iy: (B,3,H,W), expected in [0,1]
        return D: (B,1,H,W) in [0,1]
        """
        assert Iy.dim() == 4 and Iy.size(1) == 3, "Iy must be (B,3,H,W)"
        R = Iy[:, 0:1, :, :]
        G = Iy[:, 1:2, :, :]
        Bc = Iy[:, 2:3, :, :]

        # (1) local max pooling
        k = self.mip_kernel
        pad = k // 2
        Rmax = F.max_pool2d(R, kernel_size=k, stride=1, padding=pad)
        Gmax = F.max_pool2d(G, kernel_size=k, stride=1, padding=pad)
        Bmax = F.max_pool2d(Bc, kernel_size=k, stride=1, padding=pad)
        GBmax = torch.maximum(Gmax, Bmax)

        Dmip_abs = (Rmax - GBmax).abs()

        # (2) coarse depth
        N_hat = self._robust_normalize(Dmip_abs + R)

        # (3) weights from grad(N_hat)
        dxh, dyh = self._forward_grad(N_hat)
        grad_mag = torch.sqrt(dxh * dxh + dyh * dyh + self.eps)

        sigma = self._estimate_sigma(grad_mag)
        Wx = 1.0 + self.mu * torch.exp(-dxh.abs() / sigma)
        Wy = 1.0 + self.mu * torch.exp(-dyh.abs() / sigma)

        # (4) weighted TV-L2 via Chambolle–Pock
        u = N_hat.clone()
        u_bar = u.clone()
        px = torch.zeros_like(u)
        py = torch.zeros_like(u)

        tau = self.tau
        sig = self.sigma_pd
        theta = self.theta
        alpha = self.alpha

        for _ in range(self.pd_iters):
            dx, dy = self._forward_grad(u_bar)
            px = px + sig * dx
            py = py + sig * dy

            px = torch.clamp(px, -alpha * Wx, alpha * Wx)
            py = torch.clamp(py, -alpha * Wy, alpha * Wy)

            div_p = self._divergence(px, py)
            v = u + tau * div_p
            u_new = (v + tau * N_hat) / (1.0 + tau)

            u_bar = u_new + theta * (u_new - u)
            u = u_new

        return u.clamp(0.0, 1.0)