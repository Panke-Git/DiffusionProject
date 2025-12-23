"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: eps_unet.py.py
    @Time: 2025/12/23 23:21
    @Email: None
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        t = t.float()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch, num_heads=4, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.num_heads = num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        head = self.num_heads
        q = q.view(b, head, c // head, h * w)
        k = k.view(b, head, c // head, h * w)
        v = v.view(b, head, c // head, h * w)

        q = q * (c // head) ** (-0.5)
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.contiguous().view(b, c, h, w)
        out = self.proj(out)
        return out + x_in


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class EpsUNet(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch=3,
        base_ch=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        img_size=256
    ):
        super().__init__()
        self.img_size = int(img_size)
        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        downs = []
        chs = [base_ch]
        now_ch = base_ch
        now_res = self.img_size

        for mult in channel_mults:
            outc = base_ch * mult
            for _ in range(num_res_blocks):
                downs.append(ResBlock(now_ch, outc, time_dim, dropout=dropout))
                now_ch = outc
                if now_res in attn_resolutions:
                    downs.append(AttentionBlock(now_ch))
                chs.append(now_ch)
            if mult != channel_mults[-1]:
                downs.append(Downsample(now_ch))
                now_res //= 2
                chs.append(now_ch)

        self.downs = nn.ModuleList(downs)

        self.mid1 = ResBlock(now_ch, now_ch, time_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(now_ch)
        self.mid2 = ResBlock(now_ch, now_ch, time_dim, dropout=dropout)

        ups = []
        for mult in reversed(channel_mults):
            outc = base_ch * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                ups.append(ResBlock(now_ch + skip_ch, outc, time_dim, dropout=dropout))
                now_ch = outc
                if now_res in attn_resolutions:
                    ups.append(AttentionBlock(now_ch))
            if mult != channel_mults[0]:
                ups.append(Upsample(now_ch))
                now_res *= 2

        self.ups = nn.ModuleList(ups)
        self.out_norm = nn.GroupNorm(8, now_ch)
        self.out_conv = nn.Conv2d(now_ch, out_ch, 3, padding=1)

    def forward(self, x, t, extra_cond=None):
        t_emb = self.time_mlp(t)

        h = self.in_conv(x)
        hs = [h]

        for m in self.downs:
            if isinstance(m, ResBlock):
                h = m(h, t_emb)
                hs.append(h)
            elif isinstance(m, AttentionBlock):
                h = m(h)
            else:
                h = m(h)
                hs.append(h)

        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        for m in self.ups:
            if isinstance(m, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = m(h, t_emb)
            elif isinstance(m, AttentionBlock):
                h = m(h)
            else:
                h = m(h)

        return self.out_conv(F.silu(self.out_norm(h)))
