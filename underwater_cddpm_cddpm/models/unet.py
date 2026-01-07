from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn

from .modules import (
    TimestepEmbedSequential,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    conv_nd,
    normalization,
    SiLU,
    timestep_embedding,
)


class UNetModel(nn.Module):
    """
    A full DDPM-style U-Net with residual blocks + multi-head self-attention.
    Conditional setting: we concatenate (x_t, cond) along channel dim at the input.
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        cond_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: List[int],
        dropout: float = 0.0,
        channel_mult: List[int] = (1, 2, 4, 8),
        num_heads: int = 4,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.cond_channels = int(cond_channels)
        self.model_channels = int(model_channels)
        self.out_channels = int(out_channels)
        self.num_res_blocks = int(num_res_blocks)
        self.attention_resolutions = set(int(r) for r in attention_resolutions)
        self.dropout = float(dropout)
        self.channel_mult = list(channel_mult)
        self.num_heads = int(num_heads)
        self.use_checkpoint = bool(use_checkpoint)
        self.use_scale_shift_norm = bool(use_scale_shift_norm)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # input blocks
        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(
            TimestepEmbedSequential(
                conv_nd(2, self.in_channels + self.cond_channels, model_channels, 3, padding=1)
            )
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1  # downsample factor

        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                resolution = self.image_size // ds
                if resolution in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(self.channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            down=True,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    )
                )
                input_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # output blocks
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                resolution = self.image_size // ds
                if resolution in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint))

                if level != 0 and i == self.num_res_blocks:
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            up=True,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    )
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            conv_nd(2, ch, out_channels, 3, padding=1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x_t: (B,3,H,W), cond: (B,3,H,W), t: (B,)"""
        # concat condition
        x = torch.cat([x_t, cond], dim=1)

        emb = timestep_embedding(t, self.model_channels)
        emb = self.time_embed(emb)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        return self.out(h)
