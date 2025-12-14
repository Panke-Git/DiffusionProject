"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: cond_unet_ddpm.py
    @Time: 2025/12/11 23:13
    @Email: None
"""

import math
from typing import Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 1. 时间步 sinus 位置编码
# -------------------------

class SinusoidalPosEmb(nn.Module):
    """
    标准 DDPM / Transformer 风格的时间步位置编码:
    输入:  t (B,)  或 (B,1)
    输出:  (B, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        :param t: (B,) 或 (B, 1)，int 或 float 都可以
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B,1)
        # sin用一半dim, cos用一半dim；
        half_dim = self.dim // 2
        # 生成频率
        emb_scale = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)  # (half_dim,)

        # t * freqs  -> (B, half_dim)
        args = t.float() * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)  dim=2*half_dim
        if self.dim % 2 == 1:  # 如果 dim 是奇数，补一维 0
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


# -------------------------
# 2. 基础 ResBlock (带时间步注入)
# -------------------------

class ResBlock(nn.Module):
    """
    ResNet 风格的残差块 + 时间步注入（FiLM 方式）:
    y = F(x, t_emb) + shortcut(x)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_channels), num_channels=in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # 时间嵌入 -> 通道维度
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(num_groups=min(groups, out_channels), num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 如果 in/out 通道不一致，用 1x1 卷积调整 shortcut 通道
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, C_in, H, W)
        :param t_emb: (B, time_emb_dim)
        """
        h = self.conv1(self.act(self.norm1(x)))  # (B, C_out, H, W)

        # 时间嵌入映射到通道维度，并 broadcast 到 (B, C_out, H, W)
        t = self.time_mlp(t_emb)  # (B, C_out)
        h = h + t[:, :, None, None]

        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.shortcut(x)


# -------------------------
# 3. 自注意力模块 (空间 Self-Attn)
# -------------------------

class AttentionBlock(nn.Module):
    """
    空间自注意力:
    - 先 GroupNorm
    - 再 QKV 1x1 卷积
    - 在空间维度 (H*W) 上做多头注意力
    """
    def __init__(self, channels: int, num_heads: int = 4, groups: int = 32):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert (
            channels % num_heads == 0
        ), f"channels={channels} 必须能被 num_heads={num_heads} 整除"

        self.norm = nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)

        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)

        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, C, H, W)
        """
        b, c, h, w = x.shape
        h_in = x

        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # 变形为 multi-head: (B, heads, head_dim, H*W)
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        # 计算注意力: (B, heads, H*W, H*W)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(b, c, h, w)

        out = self.proj(out)
        return out + h_in


# -------------------------
# 4. 上下采样模块
# -------------------------

class Downsample(nn.Module):
    """
    下采样卷积: stride=2, 保持通道不变
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    上采样: 最近邻 + 3x3 卷积
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


# -------------------------
# 5. 条件 UNet 核心网络
# -------------------------

class UNetConditional(nn.Module):
    """
    条件 DDPM 用的 UNet:
    - 输入:
        x_t:   带噪图像 (B, in_channels,  H, W)
        cond:  条件图像 (B, cond_channels, H, W)  ← 你的原始水下图
        t:     时间步    (B,) 或 (B,1)
    - 输出:
        eps:   噪声预测 (B, out_channels, H, W)
    """

    def __init__(
        self,
        in_channels: int = 3,          # x_t 的通道数（例如 RGB=3）
        cond_channels: int = 3,        # 条件图像通道数（例如 RGB=3）
        out_channels: int = 3,         # 输出 eps 通道数（一般=3）
        base_channels: int = 64,       # UNet 最底层通道数
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        dropout: float = 0.0,
        attn_resolutions: Tuple[int, ...] = (16,),
        num_heads: int = 4,
        image_size: int = 256,         # 用于决定在哪些分辨率启用 attention
    ):
        super().__init__()

        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.image_size = image_size

        # ---- 时间嵌入 MLP ----
        self.time_pos_emb = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )
        time_emb_out_dim = time_emb_dim * 4

        # ---- 输入 conv (拼接条件图像) ----
        # 输入通道 = x_t 通道 + cond 通道
        input_channels = in_channels + cond_channels
        self.init_conv = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)

        # ---- 编码路径 (Down) ----
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        # 记录每个阶段的通道数，用于 up path 的 skip connection
        self.down_channels = [base_channels]

        in_ch = base_channels
        current_res = image_size
        # 下采样stage建立；
        for i, mult in enumerate(channel_mults):
            # 确定输出的通道数
            out_ch = base_channels * mult

            # 一层 stage 内部的多个 ResBlock，构建多个ResBlock
            for _ in range(num_res_blocks):
                block = ResBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    time_emb_dim=time_emb_out_dim,
                    dropout=dropout,
                )
                self.down_blocks.append(block)

                # 是否在该分辨率使用 self-attention
                if current_res in attn_resolutions:
                    attn = AttentionBlock(out_ch, num_heads=num_heads)
                else:
                    attn = nn.Identity()
                self.down_attns.append(attn)

                in_ch = out_ch
                self.down_channels.append(out_ch)

            # 该 stage 的下采样（最后一层不下采样）
            if i != len(channel_mults) - 1:
                self.down_samples.append(Downsample(in_ch))
                current_res //= 2
            else:
                self.down_samples.append(nn.Identity())

        # ---- bottleneck 中间层 ----
        self.mid_block1 = ResBlock(
            in_channels=in_ch,
            out_channels=in_ch,
            time_emb_dim=time_emb_out_dim,
            dropout=dropout,
        )
        self.mid_attn = AttentionBlock(in_ch, num_heads=num_heads)
        self.mid_block2 = ResBlock(
            in_channels=in_ch,
            out_channels=in_ch,
            time_emb_dim=time_emb_out_dim,
            dropout=dropout,
        )

        # ---- 解码路径 (Up) ----
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        # 逆序使用 channel_mults 构建上采样
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                # 上采样阶段，每个 ResBlock 输入:
                #   当前特征 x  与 一个 skip 特征 concat
                # skip 的通道数 = down_channels.pop()
                skip_ch = self.down_channels.pop()
                block = ResBlock(
                    in_channels=in_ch + skip_ch,
                    out_channels=out_ch,
                    time_emb_dim=time_emb_out_dim,
                    dropout=dropout,
                )
                self.up_blocks.append(block)

                if current_res in attn_resolutions:
                    attn = AttentionBlock(out_ch, num_heads=num_heads)
                else:
                    attn = nn.Identity()
                self.up_attns.append(attn)

                in_ch = out_ch

            # 上采样（最上层不再上采样）
            if i != 0:
                self.up_samples.append(Upsample(in_ch))
                current_res *= 2
            else:
                self.up_samples.append(nn.Identity())

        # ---- 输出 Head ----
        self.out_norm = nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x:     带噪图像 x_t, (B, in_channels,  H, W)
        :param cond:  条件图像, (B, cond_channels, H, W)  ← 原始水下图像
        :param t:     时间步, (B,) or (B,1)
        :return:      噪声预测 eps, (B, out_channels, H, W)
        """
        # ------ 1) time embedding ------
        t_emb = self.time_pos_emb(t)        # (B, time_emb_dim)
        t_emb = self.time_mlp(t_emb)        # (B, time_emb_out_dim)

        # ------ 2) concat 条件图像 ------
        # 拼接: [x_t, cond] -> (B, in+cond, H, W)
        h = torch.cat([x, cond], dim=1)
        h = self.init_conv(h)


        # 存放 skip connection
        skips = [h]

        # ------ 3) 编码路径 ------
        down_idx = 0
        for i in range(len(self.channel_mults)):
            # 每个 stage 里 num_res_blocks 个 ResBlock + Attn
            for _ in range(self.num_res_blocks):
                block = self.down_blocks[down_idx]
                attn = self.down_attns[down_idx]
                h = block(h, t_emb)
                h = attn(h)
                skips.append(h)
                down_idx += 1

            # 下采样
            h = self.down_samples[i](h)

        # ------ 4) 中间层 bottleneck ------
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # ------ 5) 解码路径 ------
        up_idx = 0
        for i in range(len(self.channel_mults) - 1, -1, -1):
            for _ in range(self.num_res_blocks):
                # 从 skip 中取一个特征，与当前特征 concat
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)

                block = self.up_blocks[up_idx]
                attn = self.up_attns[up_idx]
                h = block(h, t_emb)
                h = attn(h)
                up_idx += 1

            # 上采样
            h = self.up_samples[len(self.channel_mults) - 1 - i](h)

        # ------ 6) 输出 head ------
        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h


# 小测试 (你可以在本地单独跑一下)
if __name__ == "__main__":
    model = UNetConditional(
        in_channels=3,
        cond_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        dropout=0.0,
        attn_resolutions=(16,),  # 在 16x16 分辨率启用 self-attn
        num_heads=4,
        image_size=256,
    )

    x_t = torch.randn(2, 3, 256, 256)     # 带噪图像
    cond = torch.randn(2, 3, 256, 256)    # 原始水下图像
    t = torch.randint(low=0, high=1000, size=(2,))  # 时间步

    eps = model(x_t, cond, t)
    print("输出 eps shape:", eps.shape)  # 期望: (2, 3, 256, 256)


