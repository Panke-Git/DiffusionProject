"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: BaseUnet.py
    @Time: 2025/12/4 22:30
    @Email: None
"""

"""U-Net backbone for DDPM.

该文件实现了一个简化版的 U-Net，用于预测扩散过程中的噪声。结构包含：
- 残差块（ResBlock）
- 时间步嵌入（Sinusoidal positional embedding + MLP）
- 下采样与上采样模块
"""

from typing import List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """将时间步编码为固定的正弦/余弦位置嵌入。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        # 公式来自 Transformer 的位置编码，使用 log space
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


def _group_norm(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """Construct a GroupNorm that safely divides channels.

    GroupNorm 要求 num_channels 可以被 num_groups 整除，输入通道数为 3 时直接使用 8 会报错。
    这里自动选择一个不超过 num_groups 的最大因子，若没有可用因子则退化为 LayerNorm 等价的单组。
    """

    # 选择不超过 num_groups 的最大因子，至少为 1
    candidate = min(num_groups, num_channels)
    while num_channels % candidate != 0 and candidate > 1:
        candidate -= 1
    return nn.GroupNorm(candidate, num_channels)


class ResidualBlock(nn.Module):
    """带时间步调制的残差块。"""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        self.block1 = nn.Sequential(
            _group_norm(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            _group_norm(out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        # time embedding broadcast to spatial dims
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)


class Downsample(nn.Module):
    """简单的下采样卷积。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """转置卷积上采样。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Unet(nn.Module):
    """适用于 DDPM 的 U-Net 模型。"""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        channel_mults: List[int],
        num_res_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.num_res_blocks = num_res_blocks
        self.channel_mults = channel_mults

        time_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ---------------- 下采样路径 ----------------
        self.downs = nn.ModuleList()
        ch = in_channels
        skip_channels: List[int] = []

        for mult in channel_mults:
            out_ch = base_channels * mult
            # 当前分辨率上的若干残差块
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
                skip_channels.append(ch)   # 记录每个 skip 的通道数
            # 下采样到下一层分辨率
            self.downs.append(Downsample(ch))

        # ---------------- 中间层 ----------------
        self.mid = nn.ModuleList(
            [
                ResidualBlock(ch, ch, time_dim, dropout),
                ResidualBlock(ch, ch, time_dim, dropout),
            ]
        )

        # ---------------- 上采样路径 ----------------
        self.ups = nn.ModuleList()
        # 下采样时 skip_channels 是按从高分辨率到低分辨率记录的
        # 上采样时我们希望先用“最低分辨率”的 skip，所以直接从末尾 pop
        skip_stack = skip_channels.copy()

        # up path 设计：每个 level 的顺序是 [Upsample, ResBlock, ResBlock, ...]
        #   1) 先 Upsample, 通道数保持为当前 ch
        #   2) 再做 num_res_blocks 个 ResidualBlock，每个 block 前 concat 一个 skip
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult

            # 1) 先上采样：输入/输出通道 = 当前 ch
            self.ups.append(Upsample(ch))

            # 2) 再做 num_res_blocks 个残差块（每个都会 concat 一个 skip）
            for _ in range(num_res_blocks):
                skip_ch = skip_stack.pop()      # 与 forward 中 skips.pop() 对应
                self.ups.append(
                    ResidualBlock(ch + skip_ch, out_ch, time_dim, dropout)
                )
                ch = out_ch                     # 更新当前通道数为 out_ch

        self.final_norm = _group_norm(ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # 计算时间嵌入
        t = self.time_embedding(timesteps)

        # ---------- 下采样：保存每个 ResidualBlock 的输出作为 skip ----------
        skips = []
        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
                skips.append(x)      # 这里的顺序与 skip_channels 一一对应
            else:
                # Downsample
                x = layer(x)

        # ---------- 中间层 ----------
        for layer in self.mid:
            x = layer(x, t)

        # ---------- 上采样 ----------
        # self.ups 的模式是：
        #   [Up(level3), Res, Res,
        #    Up(level2), Res, Res,
        #    Up(level1), Res, Res,
        #    ...]
        # 线性扫描即可：遇到 Up 就上采样，遇到 ResidualBlock 就 pop 一个 skip, concat 后通过 block
        for layer in self.ups:
            if isinstance(layer, Upsample):
                x = layer(x)
            else:
                skip = skips.pop()   # 先从最低分辨率的 skip 开始用

                # 理论上尺寸应该完全一致；若有 1 像素差，兜底插值到 skip 的大小
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="nearest")

                x = torch.cat([x, skip], dim=1)
                x = layer(x, t)

        x = self.final_norm(x)
        x = self.final_act(x)
        return self.final_conv(x)

