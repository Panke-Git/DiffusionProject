"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: depth_estimator_admm.py
    @Time: 2026/3/24 22:53
    @Email: None
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_01(x: torch.Tensor) -> torch.Tensor:
    if x.dtype.is_floating_point:
        if x.max() > 1.5:
            return x / 255.0
        return x
    return x.float() / 255.0


def _minmax_norm_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.dim() == 3:
        x_ = x
    else:
        x_ = x.squeeze(1)

    b = x_.shape[0]
    x_flat = x_.view(b, -1)
    x_min = x_flat.min(dim=1).values.view(b, 1, 1)
    x_max = x_flat.max(dim=1).values.view(b, 1, 1)
    out = (x_ - x_min) / (x_max - x_min + eps)
    return out.unsqueeze(1)


def _max_filter2d(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)


def _forward_diff_circular(x: torch.Tensor):
    dx = torch.roll(x, shifts=-1, dims=3) - x
    dy = torch.roll(x, shifts=-1, dims=2) - x
    return dx, dy


def _psf2otf(psf: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    device = psf.device
    dtype = psf.dtype
    h, w = psf.shape
    otf = torch.zeros((out_h, out_w), device=device, dtype=dtype)
    otf[:h, :w] = psf
    shift_y = -(h // 2)
    shift_x = -(w // 2)
    otf = torch.roll(otf, shifts=(shift_y, shift_x), dims=(0, 1))
    return torch.fft.fft2(otf)


def _shrink(x: torch.Tensor, thresh: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - thresh, min=0.0)


class SceneDepthEstimatorADMM(nn.Module):
    """
    输入:
        img: (B, 3, H, W), RGB
    输出:
        depth: (B, 1, H, W), 归一化到 [0, 1]
    """
    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 2.0,
        mu: float = 0.5,
        sigma: float = 0.05,
        max_iter: int = 30,
        eps_stop: float = 1e-4,
        mip_kernel: int = 15,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.max_iter = int(max_iter)
        self.eps_stop = float(eps_stop)
        self.mip_kernel = int(mip_kernel)

        self.register_buffer("psf_dx", torch.tensor([[-1.0, 1.0]]))
        self.register_buffer("psf_dy", torch.tensor([[-1.0], [1.0]]))

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = _to_01(img)
        b, c, h, w = img.shape
        assert c == 3, "Expect RGB image with shape (B, 3, H, W)."

        ir = img[:, 0:1]
        ig = img[:, 1:2]
        ib = img[:, 2:3]

        # ----- coarse depth -----
        r_max = _max_filter2d(ir, self.mip_kernel)
        g_max = _max_filter2d(ig, self.mip_kernel)
        b_max = _max_filter2d(ib, self.mip_kernel)
        gb_max = torch.max(g_max, b_max)

        d_mip = r_max - gb_max
        n_hat = _minmax_norm_per_sample(d_mip + ir)

        # ----- ADMM -----
        n = n_hat.clone()
        bx = torch.zeros_like(n)
        by = torch.zeros_like(n)
        xx = torch.zeros_like(n)
        xy = torch.zeros_like(n)

        otf_dx = _psf2otf(self.psf_dx.to(img.device, img.dtype), h, w)
        otf_dy = _psf2otf(self.psf_dy.to(img.device, img.dtype), h, w)

        denom = (
            1.0
            + self.alpha * self.gamma * (torch.abs(otf_dx) ** 2 + torch.abs(otf_dy) ** 2)
        ).to(img.dtype)

        f_nhat = torch.fft.fft2(n_hat.squeeze(1))

        for _ in range(self.max_iter):
            n_prev = n

            dnx, dny = _forward_diff_circular(n)

            wx = 1.0 + self.mu * torch.exp(-torch.abs(dnx) / (self.sigma + 1e-12))
            wy = 1.0 + self.mu * torch.exp(-torch.abs(dny) / (self.sigma + 1e-12))

            thresh_x = wx / (2.0 * self.gamma)
            thresh_y = wy / (2.0 * self.gamma)
            xx = _shrink(dnx + bx, thresh_x)
            xy = _shrink(dny + by, thresh_y)

            fx = torch.fft.fft2((xx - bx).squeeze(1))
            fy = torch.fft.fft2((xy - by).squeeze(1))
            div_term = torch.conj(otf_dx) * fx + torch.conj(otf_dy) * fy

            numer = f_nhat + (self.alpha * self.gamma) * div_term
            f_n = numer / denom
            n = torch.fft.ifft2(f_n).real.unsqueeze(1)

            dnx_new, dny_new = _forward_diff_circular(n)
            bx = bx + dnx_new - xx
            by = by + dny_new - xy

            num = torch.sum(torch.abs(n - n_prev), dim=(1, 2, 3))
            den = torch.sum(torch.abs(n_prev), dim=(1, 2, 3)) + 1e-8
            eps_n = (num / den).max().item()
            if eps_n < self.eps_stop:
                break

        n = torch.clamp(n, 0.0, 1.0)
        n = _minmax_norm_per_sample(n)
        return n