"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: MIPTVDepthEstimator.py
    @Time: 2026/1/15 23:44
    @Email: None
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        mip_kernel: int = 31,          # local max pooling window for MIP
        p_low: float = 0.01,           # robust normalization low percentile
        p_high: float = 0.99,          # robust normalization high percentile
        alpha: float = 0.15,           # TV weight (smoothness strength)
        mu: float = 10.0,              # W = 1 + mu * exp(-|grad|/sigma)
        sigma_mode: str = "median",    # "median" or "mean" or "fixed"
        sigma_fixed: float = 0.02,     # used if sigma_mode == "fixed"
        pd_iters: int = 30,            # primal-dual iterations
        tau: float = 0.125,            # primal step
        sigma_pd: float = 0.125,       # dual step
        theta: float = 1.0,            # extrapolation
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
        # u: (B,1,H,W)
        dx = u[..., :, 1:] - u[..., :, :-1]
        dy = u[..., 1:, :] - u[..., :-1, :]

        dx = F.pad(dx, (0, 1, 0, 0))  # pad last col
        dy = F.pad(dy, (0, 0, 0, 1))  # pad last row
        return dx, dy

    @staticmethod
    def _divergence(px: torch.Tensor, py: torch.Tensor):
        """Divergence of dual field (px, py) with backward differences."""
        # px, py: (B,1,H,W)
        # backward diff for x
        div_x = px[..., :, :] - F.pad(px[..., :, :-1], (1, 0, 0, 0))
        # backward diff for y
        div_y = py[..., :, :] - F.pad(py[..., :-1, :], (0, 0, 1, 0))
        return div_x + div_y

    def _robust_normalize(self, x: torch.Tensor):
        """Per-image robust normalization to [0,1] using percentiles."""
        # x: (B,1,H,W)
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
        # grad_mag: (B,1,H,W)
        if self.sigma_mode == "fixed":
            return torch.full_like(grad_mag[:, :, :1, :1], float(self.sigma_fixed))

        B = grad_mag.shape[0]
        flat = grad_mag.view(B, -1)
        if self.sigma_mode == "mean":
            s = flat.mean(dim=1, keepdim=True)
        else:  # "median" default
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
        B, _, H, W = Iy.shape

        R = Iy[:, 0:1, :, :]
        G = Iy[:, 1:2, :, :]
        Bc = Iy[:, 2:3, :, :]

        # ---- (1) Compute D_mip via local max pooling ----
        k = self.mip_kernel
        pad = k // 2
        Rmax = F.max_pool2d(R, kernel_size=k, stride=1, padding=pad)
        Gmax = F.max_pool2d(G, kernel_size=k, stride=1, padding=pad)
        Bmax = F.max_pool2d(Bc, kernel_size=k, stride=1, padding=pad)
        GBmax = torch.maximum(Gmax, Bmax)

        Dmip = Rmax - GBmax
        # paper uses |D_mip|
        Dmip_abs = Dmip.abs()

        # ---- (2) Coarse depth N_hat = Fs(|D_mip| + R) ----
        N_hat = self._robust_normalize(Dmip_abs + R)

        # ---- (3) Scheme A: Fix W from grad(N_hat) ----
        dxh, dyh = self._forward_grad(N_hat)
        grad_mag = torch.sqrt(dxh * dxh + dyh * dyh + self.eps)

        sigma = self._estimate_sigma(grad_mag)
        # direction-specific weights (more precise than one shared W)
        Wx = 1.0 + self.mu * torch.exp(-dxh.abs() / sigma)
        Wy = 1.0 + self.mu * torch.exp(-dyh.abs() / sigma)

        # ---- (4) Weighted TV-L2 via Chambolle–Pock primal-dual ----
        # minimize 0.5||u - N_hat||^2 + alpha * sum (Wx*|dx u| + Wy*|dy u|)
        u = N_hat.clone()
        u_bar = u.clone()
        px = torch.zeros_like(u)
        py = torch.zeros_like(u)

        tau = self.tau
        sig = self.sigma_pd
        theta = self.theta
        alpha = self.alpha

        for _ in range(self.pd_iters):
            # dual ascent + projection (anisotropic TV, weighted bounds)
            dx, dy = self._forward_grad(u_bar)
            px = px + sig * dx
            py = py + sig * dy

            # project onto |p_x| <= alpha*Wx, |p_y| <= alpha*Wy
            px = torch.clamp(px, -alpha * Wx, alpha * Wx)
            py = torch.clamp(py, -alpha * Wy, alpha * Wy)

            # primal update with proximal of L2 fidelity
            div_p = self._divergence(px, py)
            v = u + tau * div_p
            u_new = (v + tau * N_hat) / (1.0 + tau)

            # extrapolation
            u_bar = u_new + theta * (u_new - u)
            u = u_new

        D = u.clamp(0.0, 1.0)
        return D

