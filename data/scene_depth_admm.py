"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: scene_depth_admm.py
    @Time: 2026/3/10 22:26
    @Email: None
"""


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
