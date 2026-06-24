"""
DGRM-A4: Depth-Guided Dual-Frequency Reconstruction Modulation.

Decoder-side feature modulation block guided by an estimated depth map.
The module decomposes a decoder feature into low/high-frequency parts,
uses depth and optional depth edges as conditions, and injects the learned
corrections in a residual manner.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_group_num(channels, max_groups=8):
    """Pick a valid GroupNorm group count for the given channel count."""
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0:
            return g
    return 1


class GaussianBlur2d(nn.Module):
    """Fixed depthwise Gaussian low-pass filter."""

    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.pad = kernel_size // 2

        ax = torch.arange(kernel_size).float() - kernel_size // 2
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        self.register_buffer("kernel", kernel.view(1, 1, kernel_size, kernel_size))

    def forward(self, x):
        _, c, _, _ = x.shape
        weight = self.kernel.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        x = F.pad(
            x,
            pad=(self.pad, self.pad, self.pad, self.pad),
            mode="reflect"
        )
        return F.conv2d(x, weight, bias=None, stride=1, padding=0, groups=c)


class ConvGNAct(nn.Module):
    """Conv + GroupNorm + SiLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2
        groups = _get_group_num(out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ZeroConv2d(nn.Module):
    """
    Zero-initialized convolution used for diffusion-friendly residual injection.
    """

    def __init__(self, channels, kernel_size=1):
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class DGRM_A4(nn.Module):
    """
    Depth-Guided Dual-Frequency Reconstruction Modulation.

    Args:
        feat: decoder feature, shape (B, C, H, W)
        depth: estimated depth map, shape (B, 1, H0, W0) or (B, H0, W0)

    Returns:
        Modulated decoder feature with shape (B, C, H, W).
    """

    def __init__(
        self,
        channels,
        cond_channels=32,
        gaussian_kernel=5,
        gaussian_sigma=1.0,
        gamma_scale_low=0.1,
        beta_scale_low=0.05,
        gamma_scale_high=0.1,
        use_depth_edge=True,
        alpha_init=1.0
    ):
        super().__init__()

        self.channels = channels
        self.use_depth_edge = use_depth_edge
        self.alpha_init = float(alpha_init)

        self.gamma_scale_low = gamma_scale_low
        self.beta_scale_low = beta_scale_low
        self.gamma_scale_high = gamma_scale_high

        self.blur_feat = GaussianBlur2d(
            kernel_size=gaussian_kernel,
            sigma=gaussian_sigma
        )
        self.blur_depth = GaussianBlur2d(
            kernel_size=gaussian_kernel,
            sigma=gaussian_sigma
        )

        lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("lap_kernel", lap_kernel)

        depth_cond_in = 2 if use_depth_edge else 1
        self.depth_encoder = nn.Sequential(
            ConvGNAct(depth_cond_in, cond_channels, kernel_size=3),
            ConvGNAct(cond_channels, cond_channels, kernel_size=3)
        )

        self.norm_low = nn.GroupNorm(_get_group_num(channels), channels)
        self.norm_high = nn.GroupNorm(_get_group_num(channels), channels)

        self.low_gate = nn.Sequential(
            nn.Conv2d(cond_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.low_gamma = nn.Conv2d(cond_channels, channels, kernel_size=1)
        self.low_beta = nn.Conv2d(cond_channels, channels, kernel_size=1)

        self.high_gate = nn.Sequential(
            nn.Conv2d(cond_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.high_gamma = nn.Conv2d(cond_channels, channels, kernel_size=1)

        self.detail_mapper = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(_get_group_num(channels), channels),
            nn.SiLU(inplace=True)
        )

        self.low_out = ZeroConv2d(channels, kernel_size=1)
        self.high_out = ZeroConv2d(channels, kernel_size=1)

        self.alpha_low = nn.Parameter(torch.tensor(self.alpha_init))
        self.alpha_high = nn.Parameter(torch.tensor(self.alpha_init))

        self.reset_identity_parameters()

    def reset_identity_parameters(self):
        """
        Restore identity-friendly initialization after global network init.

        The project applies a global initializer after module construction, so
        the zero output projections must be reset again to keep F' ~= F at the
        beginning of training.
        """
        nn.init.zeros_(self.low_gamma.weight)
        nn.init.zeros_(self.low_gamma.bias)
        nn.init.zeros_(self.low_beta.weight)
        nn.init.zeros_(self.low_beta.bias)
        nn.init.zeros_(self.high_gamma.weight)
        nn.init.zeros_(self.high_gamma.bias)
        self.low_out.reset_parameters()
        self.high_out.reset_parameters()
        with torch.no_grad():
            self.alpha_low.fill_(self.alpha_init)
            self.alpha_high.fill_(self.alpha_init)

    @staticmethod
    def _normalize_depth(depth, eps=1e-6):
        d_min = depth.amin(dim=(-2, -1), keepdim=True)
        d_max = depth.amax(dim=(-2, -1), keepdim=True)
        return (depth - d_min) / (d_max - d_min + eps)

    def _depth_laplacian_edge(self, depth):
        weight = self.lap_kernel.to(dtype=depth.dtype, device=depth.device)
        depth_pad = F.pad(depth, pad=(1, 1, 1, 1), mode="reflect")
        edge = F.conv2d(depth_pad, weight, bias=None, stride=1, padding=0)
        edge = torch.abs(edge)
        edge_max = edge.amax(dim=(-2, -1), keepdim=True)
        return edge / (edge_max + 1e-6)

    def forward(self, feat, depth, return_intermediate=False):
        if depth is None:
            if return_intermediate:
                return feat, {}
            return feat

        assert feat.dim() == 4, "feat must be [B, C, H, W]"

        b, c, h, w = feat.shape
        assert c == self.channels, (
            f"Input channels {c} do not match DGRM_A4 channels={self.channels}"
        )

        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        assert depth.dim() == 4, "depth must be [B, 1, H0, W0] or [B, H0, W0]"
        assert depth.shape[1] == 1, "depth must have one channel"
        assert depth.shape[0] == b, "feat and depth batch size must match"

        depth = depth.to(device=feat.device, dtype=feat.dtype)
        depth = F.interpolate(
            depth,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
        depth = torch.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
        depth = self._normalize_depth(depth)

        depth_g = self.blur_depth(depth)

        if self.use_depth_edge:
            depth_edge = self._depth_laplacian_edge(depth_g)
            depth_cond = torch.cat([depth_g, depth_edge], dim=1)
        else:
            depth_edge = None
            depth_cond = depth_g

        z_d = self.depth_encoder(depth_cond)

        feat_low = self.blur_feat(feat)
        feat_low_norm = self.norm_low(feat_low)

        m_low = self.low_gate(z_d)
        gamma_low = 1.0 + self.gamma_scale_low * torch.tanh(self.low_gamma(z_d))
        beta_low = self.beta_scale_low * torch.tanh(self.low_beta(z_d))
        low_mod = m_low * (gamma_low * feat_low_norm + beta_low)
        delta_low = self.low_out(low_mod)

        feat_high = feat - feat_low
        feat_high_norm = self.norm_high(feat_high)

        r_high = self.detail_mapper(feat_high_norm)
        m_high = self.high_gate(z_d)
        gamma_high = 1.0 + self.gamma_scale_high * torch.tanh(self.high_gamma(z_d))
        high_mod = m_high * (gamma_high * r_high)
        delta_high = self.high_out(high_mod)

        out = feat + self.alpha_low * delta_low + self.alpha_high * delta_high

        if return_intermediate:
            info = {
                "feat_low": feat_low,
                "feat_high": feat_high,
                "depth_g": depth_g,
                "depth_edge": depth_edge,
                "m_low": m_low,
                "m_high": m_high,
                "gamma_low": gamma_low,
                "beta_low": beta_low,
                "gamma_high": gamma_high,
                "delta_low": delta_low,
                "delta_high": delta_high,
                "alpha_low": self.alpha_low.detach(),
                "alpha_high": self.alpha_high.detach(),
            }
            return out, info

        return out
