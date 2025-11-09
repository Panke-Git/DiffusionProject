"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT
    @FileName: diffusion_unet.py
    @Time: 2025/11/09
    @Email: None
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= 时间编码（sin/cos + MLP） =========

class SinusoidalPosEmb(nn.Module):
    """t(B,) -> (B,dim) 的正弦/余弦时间编码；dim 必须为偶数。"""
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        inv_freq = torch.exp(-math.log(10000) * torch.arange(0, half).float() / half)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        t = t.float().unsqueeze(1) # (B,1)
        sinus_inp = t * self.inv_freq[None, :]  # (B,half)
        emb = torch.cat([torch.sin(sinus_inp), torch.cos(sinus_inp)], dim=1)
        return emb  # (B,dim)

class TimeEmbedding(nn.Module):
    """Sinusoidal + 2-layer MLP -> e_t (B,time_dim)"""
    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.pos = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pos(t))

# ========= U-Net 基础模块 =========

class ResidualBlock(nn.Module):
    """
    ResBlock：GN -> SiLU -> Conv3x3；第二个Norm处做时间调制（FiLM: scale/shift）。
    输入:  x (B,C_in,H,W), t_emb (B,time_dim)
    输出:  y (B,C_out,H,W)
    """
    def __init__(self, C_in: int, C_out: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, C_in)
        self.conv1 = nn.Conv2d(C_in, C_out, 3, padding=1)

        self.norm2 = nn.GroupNorm(32, C_out)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(C_out, C_out, 3, padding=1)

        self.time_mlp = nn.Linear(time_dim, C_out * 2)  # (scale, shift)
        nn.init.zeros_(self.time_mlp.weight); nn.init.zeros_(self.time_mlp.bias)

        self.skip = nn.Identity() if C_in == C_out else nn.Conv2d(C_in, C_out, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        scale_shift = self.time_mlp(F.silu(t_emb))  # (B,2*C_out)
        scale, shift = scale_shift.chunk(2, dim=1)  # (B,C_out)
        scale = scale[..., None, None]
        shift = shift[..., None, None]

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(F.silu(self.dropout(h)))
        return h + self.skip(x)

class SelfAttention2d(nn.Module):
    """多头自注意力（空间维 H*W 上的 MHSA），不改变形状。"""
    def __init__(self, C: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(32, C)
        self.qkv = nn.Conv2d(C, C*3, 1)
        self.proj = nn.Conv2d(C, C, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_n = self.norm(x)
        qkv = self.qkv(x_n)
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = c // self.heads
        # 形状整理为 (B,heads,N,dim)
        q = q.view(b, self.heads, head_dim, h*w).transpose(2,3)
        k = k.view(b, self.heads, head_dim, h*w)  # (B,heads,dim,N)
        v = v.view(b, self.heads, head_dim, h*w).transpose(2,3)
        attn = torch.softmax((q @ k) / math.sqrt(head_dim), dim=-1)  # (B,heads,N,N)
        out = attn @ v   # (B,heads,N,dim)
        out = out.transpose(2,3).contiguous().view(b, c, h, w)
        return x + self.proj(out)

class DownBlock(nn.Module):
    def __init__(self, C_in: int, C_out: int, time_dim: int, num_res: int, attn: bool, dropout: float):
        super().__init__()
        self.res = nn.ModuleList([ResidualBlock(C_in if i==0 else C_out, C_out, time_dim, dropout) for i in range(num_res)])
        self.attn = SelfAttention2d(C_out) if attn else nn.Identity()
        self.down = nn.Conv2d(C_out, C_out, 3, stride=2, padding=1)
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        for blk in self.res:
            x = blk(x, t_emb)
        x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, C_in: int, C_out: int, time_dim: int, num_res: int, attn: bool, dropout: float):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(C_in, C_out, 3, padding=1)
        self.res = nn.ModuleList([ResidualBlock(C_out*2 if i==0 else C_out, C_out, time_dim, dropout) for i in range(num_res)])
        self.attn = SelfAttention2d(C_out) if attn else nn.Identity()
    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor):
        x = self.conv(self.up(x))
        x = torch.cat([x, skip], dim=1)
        for blk in self.res:
            x = blk(x, t_emb)
        x = self.attn(x)
        return x

class MiddleBlock(nn.Module):
    def __init__(self, C: int, time_dim: int, dropout: float):
        super().__init__()
        self.res1 = ResidualBlock(C, C, time_dim, dropout)
        self.attn = SelfAttention2d(C)
        self.res2 = ResidualBlock(C, C, time_dim, dropout)
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        x = self.res1(x, t_emb)
        x = self.attn(x)
        x = self.res2(x, t_emb)
        return x

class UNet(nn.Module):
    """
    条件扩散 U-Net（通道拼接版）：
      输入通道 = 3(x_t) + 3(y) = 6；输出通道 = 3（预测噪声）。
    """
    def __init__(self, image_size: int = 256, in_channels: int = 6, out_channels: int = 3,
                 base_channels: int = 64, channel_mults=(1,2,4,8,8),
                 num_res_blocks: int = 2, attn_resolutions=(32,16),
                 time_pos_dim: int = 128, time_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.image_size = image_size
        self.time_emb = TimeEmbedding(time_pos_dim, time_dim)

        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Down
        in_ch = base_channels
        self.downs = nn.ModuleList()
        feats = []
        reso = image_size
        for mult in channel_mults:
            out_ch = base_channels * mult
            attn = (reso in attn_resolutions)
            self.downs.append(DownBlock(in_ch, out_ch, time_dim, num_res_blocks, attn, dropout))
            feats.append(out_ch)
            in_ch = out_ch
            reso //= 2
        mid_ch = in_ch

        # Middle
        self.middle = MiddleBlock(mid_ch, time_dim, dropout)

        # Up
        self.ups = nn.ModuleList()
        for mult, skip_ch in zip(reversed(channel_mults), reversed(feats)):
            out_ch = base_channels * mult
            attn = (reso in attn_resolutions)
            self.ups.append(UpBlock(in_ch, out_ch, time_dim, num_res_blocks, attn, dropout))
            in_ch = out_ch
            reso *= 2

        self.head = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_channels, 3, padding=1)
        )
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t, y: (B,3,H,W) in [-1,1]；t: (B,)
        x = torch.cat([x_t, y], dim=1)  # (B,6,H,W)
        t_emb = self.time_emb(t)

        x = self.stem(x)
        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)
        x = self.middle(x, t_emb)
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, t_emb)
        out = self.head(x)
        return out  # (B,3,H,W)
