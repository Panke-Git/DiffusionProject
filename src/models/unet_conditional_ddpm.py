"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT
    @FileName: unet_conditional_ddpm.py
    @Time: 2025/11/09
    @Email: None
"""
# models/unet_conditional_ddpm.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """标准 DDPM 用的时间步嵌入."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B] (long)
        返回: [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x, t_emb):
        # x: [B, C, H, W], t_emb: [B, time_emb_dim]
        h = self.block1(x)
        t = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        h = self.block2(h)
        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNetConditional(nn.Module):
    """
    条件 U-Net：
      输入: concat([y_t, x_input]) -> in_channels=6
      输出: 噪声预测 eps_pred，shape [B,3,H,W]
    """

    def __init__(self,
                 in_channels=6,
                 base_channels=64,
                 channel_mults=(1, 2, 4),
                 time_emb_dim=256,
                 out_channels=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 时间步 embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # 输入卷积
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # 下采样阶段
        self.downs = nn.ModuleList()
        in_ch = base_channels
        channels = []
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            block = ResidualBlock(in_ch, out_ch, time_emb_dim)
            down = Downsample(out_ch) if i != len(channel_mults) - 1 else nn.Identity()
            self.downs.append(nn.ModuleList([block, down]))
            channels.append(out_ch)
            in_ch = out_ch
        self.channels = channels

        # 中间层
        self.mid_block1 = ResidualBlock(in_ch, in_ch, time_emb_dim)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, time_emb_dim)

        # 上采样阶段
        self.ups = nn.ModuleList()
        in_ch = channels[-1]
        skip_channels = list(reversed(channels))
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            skip_ch = skip_channels[i]
            block = ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim)
            up = Upsample(out_ch) if i != len(channel_mults) - 1 else nn.Identity()
            self.ups.append(nn.ModuleList([block, up]))
            in_ch = out_ch

        # 输出卷积
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        """
        x: [B, in_channels, H, W]   (concat[y_t, x_input])
        t: [B] (long)               时间步
        """
        t_emb = self.time_mlp(t)
        x = self.conv_in(x)

        skips = []
        for block, down in self.downs:
            x = block(x, t_emb)
            skips.append(x)
            x = down(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)

        for (block, up) in self.ups:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)
            x = up(x)

        x = self.conv_out(x)
        return x
