"""
    @Project: DiffusionProject_CodeX
    @Author: paxton
    @FileName： DGRM_PLUS.py
    @Date：2026/6/16 22:17
    @OS：
    @Email: None
"""
# -*- coding: utf-8 -*-
"""
DGRM+
Depth-Guided Reconstruction Modulation Plus

Designed for decoder-side depth-guided modulation in DiffWater-like networks.

Core idea:
    1. Use Gaussian-smoothed depth to represent stable depth-related degradation.
    2. Use Laplacian response of depth to extract geometry-aware depth boundary.
    3. Generate spatial gate and affine modulation parameters from depth condition.
    4. Modulate decoder local-detail residual features in a residual manner.

Input:
    feat  : decoder feature, shape (B, C, H, W)
    depth : estimated depth map, shape (B, 1, H0, W0)

Output:
    feat_out : modulated decoder feature, shape (B, C, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-sample min-max normalization.

    Args:
        x: Tensor with shape (B, C, H, W)

    Returns:
        Normalized tensor in [0, 1] per sample.
    """
    b = x.shape[0]
    x_flat = x.reshape(b, -1)
    x_min = x_flat.min(dim=1).values.reshape(b, 1, 1, 1)
    x_max = x_flat.max(dim=1).values.reshape(b, 1, 1, 1)
    return (x - x_min) / (x_max - x_min + eps)


def get_valid_group_num(channels: int, max_groups: int = 8) -> int:
    """
    Select a valid GroupNorm group number.
    """
    for g in [max_groups, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class FixedDepthwiseFilter2d(nn.Module):
    """
    Fixed depthwise convolution for Gaussian / Laplacian filtering.

    This layer has no learnable parameters.
    It applies the same fixed kernel to each channel independently.

    Args:
        kernel: 2D tensor, shape (K, K)
        padding_mode: padding mode before convolution.
    """

    def __init__(self, kernel: torch.Tensor, padding_mode: str = "replicate"):
        super().__init__()
        assert kernel.dim() == 2, "Kernel must be a 2D tensor."

        k_h, k_w = kernel.shape
        assert k_h == k_w, "Only square kernels are supported."

        self.kernel_size = k_h
        self.pad = k_h // 2
        self.padding_mode = padding_mode

        kernel = kernel.float().view(1, 1, k_h, k_w)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape (B, C, H, W)

        Returns:
            Filtered tensor with shape (B, C, H, W)
        """
        b, c, h, w = x.shape

        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        weight = kernel.repeat(c, 1, 1, 1)

        if self.pad > 0:
            x = F.pad(
                x,
                pad=(self.pad, self.pad, self.pad, self.pad),
                mode=self.padding_mode
            )

        out = F.conv2d(
            x,
            weight=weight,
            bias=None,
            stride=1,
            padding=0,
            groups=c
        )

        return out


class DGRMPlus(nn.Module):
    """
    Depth-Guided Reconstruction Modulation Plus.

    This module is designed to be inserted into decoder stages.

    Args:
        channels:
            Number of channels of the decoder feature.

        hidden_channels:
            Hidden channels of the depth condition encoder.
            If None, it will be automatically set according to channels.

        alpha_init:
            Initial value of the learnable residual scaling parameter.
            Recommended: 0.01 or 0.05.

        use_detail_residual:
            If True, use Gaussian high-pass residual:
                F_detail = F - Gaussian(F)
            If False, directly use F.

        use_edge_guard:
            If True, suppress over-modulation around depth-discontinuous regions.

        edge_strength:
            Strength of edge protection.
            Larger value means stronger suppression near depth boundaries.

        modulation_scale:
            Bound gamma and beta by tanh and scale them for training stability.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = None,
        alpha_init: float = 0.01,
        use_detail_residual: bool = True,
        use_edge_guard: bool = True,
        edge_strength: float = 1.0,
        modulation_scale: float = 0.5,
    ):
        super().__init__()

        self.channels = channels
        self.use_detail_residual = use_detail_residual
        self.use_edge_guard = use_edge_guard
        self.edge_strength = edge_strength
        self.modulation_scale = modulation_scale

        if hidden_channels is None:
            hidden_channels = max(16, min(64, channels // 4))

        # ------------------------------------------------------------
        # Fixed Gaussian kernel
        # ------------------------------------------------------------
        gaussian_kernel = torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [2.0, 4.0, 2.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=torch.float32
        ) / 16.0

        # ------------------------------------------------------------
        # Fixed Laplacian kernel, 4-neighborhood version
        # ------------------------------------------------------------
        laplacian_kernel = torch.tensor(
            [
                [0.0,  1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0,  1.0, 0.0],
            ],
            dtype=torch.float32
        )

        self.gaussian_filter = FixedDepthwiseFilter2d(gaussian_kernel)
        self.laplacian_filter = FixedDepthwiseFilter2d(laplacian_kernel)

        # ------------------------------------------------------------
        # Depth condition encoder
        # Input channels:
        #   1 channel: Gaussian-smoothed depth
        #   1 channel: Laplacian depth boundary
        # ------------------------------------------------------------
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # ------------------------------------------------------------
        # Generate channel-wise spatial affine parameters:
        #   gamma: (B, C, H, W)
        #   beta : (B, C, H, W)
        # ------------------------------------------------------------
        self.to_gamma_beta = nn.Conv2d(
            hidden_channels,
            2 * channels,
            kernel_size=3,
            padding=1
        )

        # ------------------------------------------------------------
        # Generate spatial modulation gate:
        #   gate: (B, 1, H, W)
        # It controls where the depth modulation should be activated.
        # ------------------------------------------------------------
        self.to_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # ------------------------------------------------------------
        # Detail residual branch
        # This branch transforms local detail residual features.
        # ------------------------------------------------------------
        group_num = get_valid_group_num(channels)

        self.detail_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_num, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        # Learnable residual scaling parameter
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat:
                Decoder feature, shape (B, C, H, W)

            depth:
                Estimated depth map, shape (B, 1, H0, W0)

        Returns:
            Modulated decoder feature, shape (B, C, H, W)
        """
        if depth is None:
            return feat

        assert feat.dim() == 4, "feat should have shape (B, C, H, W)."
        assert depth.dim() == 4, "depth should have shape (B, 1, H, W)."
        assert depth.shape[1] == 1, "depth should be a single-channel map."

        b, c, h, w = feat.shape
        assert c == self.channels, (
            f"Channel mismatch: DGRMPlus expects {self.channels} channels, "
            f"but got {c}."
        )
        depth = depth.to(device=feat.device, dtype=feat.dtype)

        # ------------------------------------------------------------
        # 1. Resize depth map to current decoder feature scale
        # ------------------------------------------------------------
        depth_l = F.interpolate(
            depth,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
        depth_l = normalize_01(depth_l)

        # ------------------------------------------------------------
        # 2. Gaussian-smoothed depth prior
        #    This suppresses pseudo-depth noise and captures stable
        #    depth-related degradation distribution.
        # ------------------------------------------------------------
        depth_g = self.gaussian_filter(depth_l)
        depth_g = normalize_01(depth_g)

        # ------------------------------------------------------------
        # 3. Laplacian depth boundary response
        #    This extracts geometry-aware depth discontinuity.
        # ------------------------------------------------------------
        depth_edge = torch.abs(self.laplacian_filter(depth_g))
        depth_edge = normalize_01(depth_edge)

        # ------------------------------------------------------------
        # 4. Depth condition construction
        # ------------------------------------------------------------
        depth_cond = torch.cat([depth_g, depth_edge], dim=1)
        cond_feat = self.depth_encoder(depth_cond)

        # ------------------------------------------------------------
        # 5. Generate modulation parameters
        # ------------------------------------------------------------
        gamma_beta = self.to_gamma_beta(cond_feat)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)

        # Bound gamma and beta for stable residual modulation
        gamma = self.modulation_scale * torch.tanh(gamma)
        beta = self.modulation_scale * torch.tanh(beta)

        # ------------------------------------------------------------
        # 6. Generate spatial modulation gate
        # ------------------------------------------------------------
        gate = self.to_gate(cond_feat)

        # Optional edge guard:
        # reduce over-modulation near depth-discontinuous boundaries.
        if self.use_edge_guard:
            edge_guard = torch.exp(-self.edge_strength * depth_edge)
            gate = gate * edge_guard

        # ------------------------------------------------------------
        # 7. Local detail residual extraction
        #    F_detail = F - Gaussian(F)
        # ------------------------------------------------------------
        if self.use_detail_residual:
            feat_low = self.gaussian_filter(feat)
            feat_detail = feat - feat_low
        else:
            feat_detail = feat

        detail_residual = self.detail_branch(feat_detail)

        # ------------------------------------------------------------
        # 8. Depth-guided residual modulation
        # ------------------------------------------------------------
        delta_feat = gate * (gamma * detail_residual + beta)

        # ------------------------------------------------------------
        # 9. Residual injection
        # ------------------------------------------------------------
        out = feat + self.alpha * delta_feat

        return out
