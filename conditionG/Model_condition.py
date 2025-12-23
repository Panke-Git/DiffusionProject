# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
# [MOD] Completed missing parts + changed channels for underwater RGB conditional DDPM:
#       Input = concat([noisy_gt(3), cond(3)]) => 6 channels
#       Output = predicted noise for gt => 3 channels

import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    Sinusoidal embedding -> MLP
    """
    def __init__(self, T: int, d_model: int, dim: int):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.time_embedding = nn.Embedding.from_pretrained(emb, freeze=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

        self.initialize()

    def initialize(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, t):
        return self.mlp(self.time_embedding(t))


class DownSample(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        return self.main(x)


class UpSample(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.main(x)


class AttnBlock(nn.Module):
    """
    Simple self-attention over spatial positions.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.initialize()

    def initialize(self):
        for m in [self.q, self.k, self.v, self.proj]:
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.q(h).reshape(B, C, H * W).permute(0, 2, 1)      # (B, HW, C)
        k = self.k(h).reshape(B, C, H * W)                       # (B, C, HW)
        v = self.v(h).reshape(B, C, H * W).permute(0, 2, 1)      # (B, HW, C)

        attn = torch.bmm(q, k) * (C ** -0.5)                     # (B, HW, HW)
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        out = self.proj(out)
        return x + out


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, tdim: int, dropout: float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(tdim, out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

        self.act = Swish()
        self.initialize()

    def initialize(self):
        for m in [self.conv1, self.conv2]:
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)
        if isinstance(self.shortcut, nn.Conv2d):
            init.xavier_uniform_(self.shortcut.weight)
            init.zeros_(self.shortcut.bias)
        init.xavier_uniform_(self.temb_proj.weight)
        init.zeros_(self.temb_proj.bias)

    def forward(self, x, temb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.temb_proj(self.act(temb))[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return self.shortcut(x) + h


class UNet(nn.Module):
    """
    [MOD] Conditional UNet for RGB enhancement diffusion:
      - in_channels = 6  (noisy_gt(3) + cond(3))
      - out_channels = 3 (predicted noise for gt)
    """
    def __init__(
        self,
        T: int,
        ch: int = 128,
        ch_mult=(1, 2, 2, 2),
        attn=(3,),  # resolutions indices to apply attention (0 is highest res)
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        in_channels: int = 6,   # [MOD]
        out_channels: int = 3,  # [MOD]
    ):
        super().__init__()
        self.T = T
        self.ch = ch
        tdim = ch * 4
        self.time_embed = TimeEmbedding(T, d_model=ch, dim=tdim)

        self.head = nn.Conv2d(in_channels, ch, 3, padding=1)  # [MOD]
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)

        # down
        downblocks = []
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch_i = ch * mult
            for _ in range(num_res_blocks):
                downblocks.append(ResBlock(now_ch, out_ch_i, tdim, dropout))
                now_ch = out_ch_i
                chs.append(now_ch)
                if i in attn:
                    downblocks.append(AttnBlock(now_ch))
                    # chs.append(now_ch)
            if i != len(ch_mult) - 1:
                downblocks.append(DownSample(now_ch))
                chs.append(now_ch)
        self.downblocks = nn.ModuleList(downblocks)

        # middle
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout),
            AttnBlock(now_ch),
            ResBlock(now_ch, now_ch, tdim, dropout),
        ])

        # up
        upblocks = []
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch_i = ch * mult
            for _ in range(num_res_blocks + 1):
                in_ch_i = now_ch + chs.pop()
                upblocks.append(ResBlock(in_ch_i, out_ch_i, tdim, dropout))
                now_ch = out_ch_i
                if i in attn:
                    upblocks.append(AttnBlock(now_ch))
            if i != 0:
                upblocks.append(UpSample(now_ch))
        self.upblocks = nn.ModuleList(upblocks)

        self.tail = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=now_ch, eps=1e-6, affine=True),
            Swish(),
            nn.Conv2d(now_ch, out_channels, 3, padding=1),  # [MOD]
        )
        init.xavier_uniform_(self.tail[-1].weight)
        init.zeros_(self.tail[-1].bias)

        assert len(chs) == 0

    def forward(self, x, t):
        temb = self.time_embed(t)
        h = self.head(x)
        hs = [h]

        for layer in self.downblocks:
            if isinstance(layer, AttnBlock):
                h = layer(h)
            else:
                h = layer(h, temb)
                hs.append(h)

        for layer in self.middleblocks:
            if isinstance(layer, AttnBlock):
                h = layer(h)
            else:
                h = layer(h, temb)

        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, temb)
            elif isinstance(layer, AttnBlock):
                h = layer(h)
            else:
                h = layer(h, temb)

        h = self.tail(h)
        return h
