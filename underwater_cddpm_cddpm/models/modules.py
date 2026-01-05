from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    timesteps: (B,) int64 or float
    returns: (B, dim)
    """
    if timesteps.dtype != torch.float32 and timesteps.dtype != torch.float64:
        timesteps = timesteps.float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, device=timesteps.device, dtype=timesteps.dtype) / half
    )
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def conv_nd(dims: int, in_ch: int, out_ch: int, kernel: int, stride: int = 1, padding: int = 0):
    if dims == 1:
        return nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding)
    if dims == 2:
        return nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
    if dims == 3:
        return nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=padding)
    raise ValueError(f"unsupported dims: {dims}")


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that always uses float32 for stability."""
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int):
    # 32 groups is standard; fall back if channels < 32
    groups = 32
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return GroupNorm32(groups, channels)


class TimestepBlock(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """A sequential module that passes timestep embeddings to the children that need it."""

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = conv_nd(2, channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.op = conv_nd(2, channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = True,
        up: bool = False,
        down: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.up = up
        self.down = down
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(in_channels),
            SiLU(),
            conv_nd(2, in_channels, self.out_channels, 3, padding=1),
        )

        if up:
            self.h_upd = Upsample(in_channels, use_conv=False)
            self.x_upd = Upsample(in_channels, use_conv=False)
        elif down:
            self.h_upd = Downsample(in_channels, use_conv=False)
            self.x_upd = Downsample(in_channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(2, self.out_channels, self.out_channels, 3, padding=1),
        )

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            if use_conv:
                self.skip_connection = conv_nd(2, in_channels, self.out_channels, 3, padding=1)
            else:
                self.skip_connection = conv_nd(2, in_channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # Standard DDPM/Guided-Diffusion style up/down ResBlock.
        h = x
        h = self.in_layers[0](h)
        h = self.in_layers[1](h)
        h = self.h_upd(h)
        x = self.x_upd(x)
        h = self.in_layers[2](h)
        emb_out = self.emb_layers(emb).type(h.dtype)

        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h_norm = self.out_layers[0](h)
            h = h_norm * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            h = self.out_layers[1:](h)
        else:
            h = h + emb_out[:, :, None, None]
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""

    def __init__(self, channels: int, num_heads: int = 1, use_checkpoint: bool = False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(2, channels, channels * 3, 1)
        self.proj_out = conv_nd(2, channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # (B, 3C, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # reshape to (B, heads, head_dim, HW)
        head_dim = C // self.num_heads
        assert C % self.num_heads == 0, "channels must be divisible by num_heads"

        q = q.view(B, self.num_heads, head_dim, H * W)
        k = k.view(B, self.num_heads, head_dim, H * W)
        v = v.view(B, self.num_heads, head_dim, H * W)

        # scaled dot-product attention
        scale = 1.0 / math.sqrt(head_dim)
        attn = torch.einsum("bhcn,bhcm->bhnm", q * scale, k)  # (B, heads, HW, HW)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)  # (B, heads, head_dim, HW)
        out = out.reshape(B, C, H, W)
        out = self.proj_out(out)
        return x + out
