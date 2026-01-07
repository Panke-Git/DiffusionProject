# coding=utf-8
"""
    @Project: 
    @Author: PyCharm
    @FileName： unet_ddpm.py
    @Date：2025/12/25 16:15
    @Email: None
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else d

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        half = self.dim // 2
        device = t.device
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class ResBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_scale_shift_norm = use_scale_shift_norm

        self.norm1 = nn.GroupNorm(32, in_ch)
        self.act1 = SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(time_emb_dim, out_ch * (2 if use_scale_shift_norm else 1))
        )

        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act2 = SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))

        temb = self.time_mlp(t_emb)  # (B, out_ch or 2*out_ch)
        if self.use_scale_shift_norm:
            scale, shift = temb.chunk(2, dim=1)
            scale = scale[:, :, None, None]
            shift = shift[:, :, None, None]
            h = self.norm2(h) * (1 + scale) + shift
            h = self.conv2(self.dropout(self.act2(h)))
        else:
            h = h + temb[:, :, None, None]
            h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        return h + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        head = self.num_heads
        ch = c // head
        q = q.reshape(b, head, ch, h * w)
        k = k.reshape(b, head, ch, h * w)
        v = v.reshape(b, head, ch, h * w)

        q = q * (ch ** -0.5)
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k)  # (b, head, hw, hw)
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return x + out

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)

@dataclass
class UNetConfig:
    in_channels: int
    out_channels: int
    base_channels: int = 128
    channel_mult: List[int] = None
    num_res_blocks: int = 2
    attn_resolutions: List[int] = None
    dropout: float = 0.0
    num_heads: int = 4
    use_scale_shift_norm: bool = True
    image_size: int = 256

class UNetModel(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        assert cfg.channel_mult is not None
        assert cfg.attn_resolutions is not None

        self.cfg = cfg
        time_dim = cfg.base_channels * 4

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(cfg.base_channels),
            nn.Linear(cfg.base_channels, time_dim),
            SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(cfg.in_channels, cfg.base_channels, 3, padding=1)

        # Down path
        self.downs = nn.ModuleList()
        self.down_channels = [cfg.base_channels]
        cur_ch = cfg.base_channels
        cur_res = cfg.image_size

        for i, mult in enumerate(cfg.channel_mult):
            out_ch = cfg.base_channels * mult
            for _ in range(cfg.num_res_blocks):
                self.downs.append(ResBlock(cur_ch, out_ch, time_dim, cfg.dropout, cfg.use_scale_shift_norm))
                cur_ch = out_ch
                if cur_res in cfg.attn_resolutions:
                    self.downs.append(AttentionBlock(cur_ch, cfg.num_heads))
                self.down_channels.append(cur_ch)
            if i != len(cfg.channel_mult) - 1:
                self.downs.append(Downsample(cur_ch))
                cur_res //= 2
                self.down_channels.append(cur_ch)

        # Middle
        self.mid = nn.ModuleList([
            ResBlock(cur_ch, cur_ch, time_dim, cfg.dropout, cfg.use_scale_shift_norm),
            AttentionBlock(cur_ch, cfg.num_heads),
            ResBlock(cur_ch, cur_ch, time_dim, cfg.dropout, cfg.use_scale_shift_norm),
        ])

        # Up path
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(cfg.channel_mult))):
            out_ch = cfg.base_channels * mult
            for _ in range(cfg.num_res_blocks + 1):
                skip_ch = self.down_channels.pop()
                self.ups.append(ResBlock(cur_ch + skip_ch, out_ch, time_dim, cfg.dropout, cfg.use_scale_shift_norm))
                cur_ch = out_ch
                if cur_res in cfg.attn_resolutions:
                    self.ups.append(AttentionBlock(cur_ch, cfg.num_heads))
            if i != 0:
                self.ups.append(Upsample(cur_ch))
                cur_res *= 2

        self.out_norm = nn.GroupNorm(32, cur_ch)
        self.out_act = SiLU()
        self.out_conv = nn.Conv2d(cur_ch, cfg.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        if t.device != x.device:
            t = t.to(x.device)
        if t.dtype != torch.long:
            t = t.long()

        # x: (B, in_channels, H, W), t: (B,)
        t_emb = self.time_embed(t)

        h = self.in_conv(x)
        hs = [h]

        for m in self.downs:
            if isinstance(m, ResBlock):
                h = m(h, t_emb)
                hs.append(h)
            elif isinstance(m, AttentionBlock):
                h = m(h)
                hs.append(h)
            else:
                h = m(h)
                hs.append(h)

        for m in self.mid:
            if isinstance(m, ResBlock):
                h = m(h, t_emb)
            else:
                h = m(h)

        for m in self.ups:
            if isinstance(m, ResBlock):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = m(h, t_emb)
            elif isinstance(m, AttentionBlock):
                h = m(h)
            else:
                h = m(h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h