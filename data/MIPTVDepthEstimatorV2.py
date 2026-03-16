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


import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_01(x: torch.Tensor) -> torch.Tensor:
    # Accept [0,255] or [0,1]
    if x.dtype.is_floating_point:
        if x.max() > 1.5:
            return x / 255.0
        return x
    return x.float() / 255.0


def _minmax_norm_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: (B,1,H,W) or (B,H,W)
    if x.dim() == 3:
        x_ = x
    else:
        x_ = x.squeeze(1)

    B = x_.shape[0]
    x_flat = x_.view(B, -1)
    x_min = x_flat.min(dim=1).values.view(B, 1, 1)
    x_max = x_flat.max(dim=1).values.view(B, 1, 1)
    out = (x_ - x_min) / (x_max - x_min + eps)
    return out.unsqueeze(1)


def _max_filter2d(x: torch.Tensor, k: int) -> torch.Tensor:
    # x: (B,1,H,W)
    pad = k // 2
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)


def _forward_diff_circular(x: torch.Tensor):
    # x: (B,1,H,W), circular forward differences to match FFT solution
    dx = torch.roll(x, shifts=-1, dims=3) - x
    dy = torch.roll(x, shifts=-1, dims=2) - x
    return dx, dy


def _psf2otf(psf: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """
    psf: (h,w) real kernel
    return: OTF complex tensor (out_h,out_w) in torch.fft domain
    """
    device = psf.device
    dtype = psf.dtype
    h, w = psf.shape
    otf = torch.zeros((out_h, out_w), device=device, dtype=dtype)
    otf[:h, :w] = psf

    # circularly shift so that psf center is at (0,0)
    shift_y = -(h // 2)
    shift_x = -(w // 2)
    otf = torch.roll(otf, shifts=(shift_y, shift_x), dims=(0, 1))

    return torch.fft.fft2(otf)  # complex


def _shrink(x: torch.Tensor, thresh: torch.Tensor) -> torch.Tensor:
    # shrink(x,phi)=sign(x)*max(|x|-phi,0)
    return torch.sign(x) * torch.clamp(torch.abs(x) - thresh, min=0.0)


class SceneDepthEstimatorADMM(nn.Module):
    """
    Dense depth map D(x) from:
      - coarse depth: N_hat = Fs(D_mip + I^r)
      - ADMM TV optimization (Algorithm 3.1 in your figure):
            min_N ||N-N_hat||_2^2 + alpha || W ∘ ∇N ||_1
      - W = 1 + mu * exp(-|∇N|/sigma)  (adaptive weighting)

    Output:
      D in [0,1], shape (B,1,H,W)
    """

    def __init__(
        self,
        alpha: float = 0.15,      # α in (3.13)
        gamma: float = 2.0,       # γ in (3.16)
        mu: float = 0.5,          # μ in W definition
        sigma: float = 0.05,      # σ in W definition (assumes N in [0,1])
        max_iter: int = 30,       # ADMM iterations
        eps_stop: float = 1e-4,   # ε in (3.22)
        mip_kernel: int = 15,     # local max window for "最大强度"
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.max_iter = int(max_iter)
        self.eps_stop = float(eps_stop)
        self.mip_kernel = int(mip_kernel)

        # fixed gradient kernels for FFT closed-form (forward difference)
        # dx: [-1, 1], dy: [-1; 1]
        self.register_buffer("psf_dx", torch.tensor([[-1.0, 1.0]]))
        self.register_buffer("psf_dy", torch.tensor([[-1.0], [1.0]]))

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B,3,H,W)
        """
        img = _to_01(img)
        B, C, H, W = img.shape
        assert C == 3, "Expect RGB image, shape (B,3,H,W)."

        Ir = img[:, 0:1]  # red channel
        Ig = img[:, 1:2]
        Ib = img[:, 2:3]

        # ---- (A) Coarse depth: N_hat = Fs(D_mip + I^r) ----
        # D_mip: difference between local max of red and local max of (green/blue)
        r_max = _max_filter2d(Ir, self.mip_kernel)
        g_max = _max_filter2d(Ig, self.mip_kernel)
        b_max = _max_filter2d(Ib, self.mip_kernel)
        gb_max = torch.max(g_max, b_max)
        D_mip = r_max - gb_max

        N_hat = _minmax_norm_per_sample(D_mip + Ir)  # Fs(·)

        # ---- (B) ADMM TV optimization (Algorithm 3.1) ----
        # init
        N = N_hat.clone()
        bx = torch.zeros_like(N)
        by = torch.zeros_like(N)
        Xx = torch.zeros_like(N)
        Xy = torch.zeros_like(N)

        # FFT precompute
        otf_dx = _psf2otf(self.psf_dx.to(img.device, img.dtype), H, W)  # (H,W) complex
        otf_dy = _psf2otf(self.psf_dy.to(img.device, img.dtype), H, W)

        # denom = F(1) + α γ ( |Dx|^2 + |Dy|^2 )
        # F(1) is ones in freq domain
        denom = (1.0 + self.alpha * self.gamma * (torch.abs(otf_dx) ** 2 + torch.abs(otf_dy) ** 2)).to(img.dtype)

        # FFT of N_hat
        F_Nhat = torch.fft.fft2(N_hat.squeeze(1))  # (B,H,W) complex

        for _ in range(self.max_iter):
            N_prev = N

            # ∇N
            dNx, dNy = _forward_diff_circular(N)

            # W = 1 + mu * exp(-|∇N|/sigma)
            # here make directional weights (same idea, more stable)
            Wx = 1.0 + self.mu * torch.exp(-torch.abs(dNx) / (self.sigma + 1e-12))
            Wy = 1.0 + self.mu * torch.exp(-torch.abs(dNy) / (self.sigma + 1e-12))

            # (1) X update by shrinkage: X = shrink(∇N + b, W/(2γ))
            thresh_x = Wx / (2.0 * self.gamma)
            thresh_y = Wy / (2.0 * self.gamma)
            Xx = _shrink(dNx + bx, thresh_x)
            Xy = _shrink(dNy + by, thresh_y)

            # (2) N update in Fourier domain (3.20)
            # div_term = Dx^*(Xx - bx) + Dy^*(Xy - by)
            Fx = torch.fft.fft2((Xx - bx).squeeze(1))
            Fy = torch.fft.fft2((Xy - by).squeeze(1))
            div_term = torch.conj(otf_dx) * Fx + torch.conj(otf_dy) * Fy  # (B,H,W) complex

            numer = F_Nhat + (self.alpha * self.gamma) * div_term
            F_N = numer / denom  # broadcast denom (H,W)
            N = torch.fft.ifft2(F_N).real.unsqueeze(1)

            # (3) b update: b = b + ∇N - X
            dNx_new, dNy_new = _forward_diff_circular(N)
            bx = bx + dNx_new - Xx
            by = by + dNy_new - Xy

            # stop criterion (3.22): eps_N = ||N^j - N^{j-1}||_1 / ||N^{j-1}||_1
            num = torch.sum(torch.abs(N - N_prev), dim=(1, 2, 3))
            den = torch.sum(torch.abs(N_prev), dim=(1, 2, 3)) + 1e-8
            eps_N = (num / den).max().item()
            if eps_N < self.eps_stop:
                break

        # final clamp + renorm (keep consistent [0,1])
        N = torch.clamp(N, 0.0, 1.0)
        N = _minmax_norm_per_sample(N)  # safer, because FFT can introduce tiny drift
        return N


class Block1_MIPTVV2(nn.Module):
    def __init__(self, input_channels, output_channels, n_channels, ch_mults, n_blocks, eps=1e-6):
        super(Block1_MIPTVV2, self).__init__()
        self.beta_predictor = Beta_UNet(3, 3, n_channels, ch_mults, n_blocks)
        self.depth_estimator = SceneDepthEstimatorADMM()
        self.eps = float(eps)

    def forward(self, condition):
        pred_beta = self.beta_predictor(condition)   # (B,3,1,1)
        depth = self.depth_estimator(condition)      # (B,1,H,W)

        T_direct = torch.exp(-pred_beta * depth)
        T_direct = torch.clamp(T_direct, min=self.eps, max=1.0)

        T_scatter = 1.0 - T_direct

        atm_light = [get_A(item) for item in condition]
        atm_light = torch.stack(atm_light).to(condition.device, dtype=condition.dtype)

        if atm_light.dim() == 2:
            atm_light = atm_light.unsqueeze(-1).unsqueeze(-1)

        J = (condition - T_scatter * atm_light) / T_direct
        return torch.clamp(J, 0.0, 1.0)