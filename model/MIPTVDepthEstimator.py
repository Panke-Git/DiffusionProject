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
from PIL import Image
from PIL.ImageFilter import GaussianBlur


def batch_gaussian_blur_pil(x: torch.Tensor, radius: float) -> torch.Tensor:
    """
    用 PIL.ImageFilter.GaussianBlur 对 batch tensor 做高斯模糊。
    x: (B,3,H,W), float, 值域期望 [0,1]
    return: (B,3,H,W), float, [0,1]，在 CPU 上做完再搬回原 device
    """
    assert x.dim() == 4 and x.size(1) == 3, f"Expect (B,3,H,W), got {x.shape}"
    device = x.device
    dtype = x.dtype

    x_cpu = x.detach().float().clamp(0, 1).cpu()  # PIL 只能 CPU + 不可微
    B, C, H, W = x_cpu.shape
    outs = []

    for i in range(B):
        img = x_cpu[i].permute(1, 2, 0).numpy()          # HWC, float
        img_u8 = (img * 255.0 + 0.5).astype(np.uint8)    # uint8
        pil = Image.fromarray(img_u8, mode="RGB")
        pil_blur = pil.filter(GaussianBlur(radius=radius))

        arr = np.asarray(pil_blur).astype(np.float32) / 255.0  # HWC float [0,1]
        ten = torch.from_numpy(arr).permute(2, 0, 1)           # CHW
        outs.append(ten)

    out = torch.stack(outs, dim=0).to(device=device, dtype=dtype)
    return out


# =========================
# Utils (from DocDiff, minimal)
# =========================
def get_pad(in_, ksize, stride, atrous=1):
    """
    Compute padding for "same" output size style conv when using dilation.
    This is kept from your DocDiff.py because MiddleBlock uses it.
    """
    out_ = np.ceil(float(in_) / float(stride))
    pad = int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)
    return pad


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """
    Minimal residual block used by Beta_UNet.
    Note: Beta_UNet uses is_noise=False, so time embedding branch is disabled.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float = 0.1,
        is_noise: bool = True,
    ):
        super().__init__()
        self.is_noise = is_noise

        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        if self.is_noise:
            self.time_emb = nn.Linear(time_channels, out_channels)
            self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None) -> torch.Tensor:
        h = self.conv1(self.act1(x))

        # Only used when is_noise=True
        if self.is_noise:
            if t is None:
                raise ValueError("t must be provided when is_noise=True")
            h = h + self.time_emb(self.time_act(t))[:, :, None, None]

        h = self.conv2(self.dropout(self.act2(h)))
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """Down block = ResidualBlock (attention removed in your Beta_UNet path)."""

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

    def forward(self, x: torch.Tensor, t: torch.Tensor | None) -> torch.Tensor:
        return self.res(x, t)


class Downsample(nn.Module):
    """Downsample by 2 using stride-2 conv."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None) -> torch.Tensor:
        _ = t  # kept for signature compatibility
        return self.conv(x)


class MiddleBlock(nn.Module):
    """
    Middle block used by Beta_UNet:
    ResidualBlock + several dilated convs + ResidualBlock
    """

    def __init__(self, n_channels: int, time_channels: int, is_noise: bool = True):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels, n_channels, time_channels, is_noise=is_noise
        )
        # keep exactly as your DocDiff.py style
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

    def forward(self, x: torch.Tensor, t: torch.Tensor | None) -> torch.Tensor:
        x = self.res1(x, t)
        x = self.dia1(x)
        x = self.dia2(x)
        x = self.dia3(x)
        x = self.dia4(x)
        x = self.res2(x, t)
        return x


# =========================
# Beta_UNet (minimal kept)
# =========================
class Beta_UNet(nn.Module):
    """
    Predict beta parameters from condition image.

    Input : (B, input_channels, H, W)  typically input_channels=3
    Output: (B, output_channels, 1, 1) typically output_channels=3
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        n_channels: int,
        ch_mults: list[int],
        n_blocks: int,
    ):
        super().__init__()
        is_noise = False  # as in your DocDiff.py

        n_resolutions = len(ch_mults)

        self.image_proj = nn.Conv2d(
            input_channels, n_channels, kernel_size=3, padding=1
        )

        down = []
        out_channels = in_channels = n_channels

        for i in range(n_resolutions):
            out_channels = n_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels, out_channels, time_channels=n_channels * 4, is_noise=is_noise
                    )
                )
                in_channels = out_channels

            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, time_channels=n_channels * 4, is_noise=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # IMPORTANT: adapt to actual out_channels (original code was nn.Linear(128,3))
        self.transform = nn.Sequential(
            nn.Linear(out_channels, output_channels),
            Swish(),
            nn.Linear(output_channels, output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = None
        x = self.image_proj(x)

        for m in self.down:
            x = m(x, t)

        x = self.middle(x, t)
        x = self.pool(x).squeeze(-1).squeeze(-1)       # (B, C)
        x = torch.sigmoid(self.transform(x))           # (B, output_channels) in (0,1)
        return x.unsqueeze(-1).unsqueeze(-1)           # (B, output_channels, 1, 1)


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