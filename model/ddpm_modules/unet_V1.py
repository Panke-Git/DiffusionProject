from inspect import isfunction

import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0, norm_groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        # mid_ch=1024
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid_uas = UAS4StatsAttnBlock(
            channels=pre_channel,
            time_embed_dim=time_dim,  # ✅ 你有 time embedding 就传；如果不想用就改 None
            reduced_channels=max(8, pre_channel // 4),  # 可选
        )


        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=norm_groups, 
                                dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, time_emb_dim=time_dim, dropout=dropout, norm_groups=norm_groups, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)


        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x.to(torch.float32))
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            # print(f'mid layer {i} output shape:', x.shape)
        x = self.mid_uas(x, t)
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)



def _choose_gn_groups(ch: int, max_groups: int = 32) -> int:
    """Pick a GroupNorm group count that divides channels."""
    gmax = min(max_groups, ch)
    for g in range(gmax, 0, -1):
        if ch % g == 0:
            return g
    return 1


def _zero_module(m: nn.Module) -> nn.Module:
    """Zero-initialize parameters (diffusion-friendly residual branches)."""
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class UAS4StatsAttnBlock(nn.Module):
    """
    UAS-4 Statistical Attention Block for diffusion U-Net.

    Input:
        x:    (B, C, H, W)
        temb: (B, T) time embedding used in diffusion UNet (optional)

    Tokens (K=4):
        1) mean      (mu)        : per-channel spatial mean
        2) std       (sigma)     : per-channel spatial std
        3) dcp_like  (softmin)   : per-channel spatial soft-min (DCP proxy)
        4) grad_energy           : per-channel mean gradient magnitude

    Attention:
        Query  = spatial features (B, HW, Cr)
        Key/Val= stats tokens     (B, K,  Cr)
        Output = token-weighted injection back to spatial map + residual

    Notes:
      - stats + attention are computed in float32 for stability under AMP.
      - GroupNorm is used (batch-size=1 friendly).
    """

    def __init__(
            self,
            channels: int,
            time_embed_dim: Optional[int] = None,  # ✅ 改：默认 None
            reduced_channels: Optional[int] = None,
            eps: float = 1e-6,
            softmin_beta: float = 10.0,
            token_film: bool = True,
            token_film_mode: Literal["shared", "per_token"] = "shared",
            gate_residual: bool = True,
            gate_type: Literal["scalar", "channel"] = "scalar",
            attn_dropout: float = 0.0,
            max_gn_groups: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        self.reduced_channels = reduced_channels or max(8, channels // 4)
        self.reduced_channels = min(self.reduced_channels, channels)
        self.eps = eps
        self.softmin_beta = softmin_beta

        self.K = 4  # UAS-4

        # ✅ 如果没给 time_embed_dim，就彻底关掉 token_film / gate_residual
        self.token_film = bool(token_film and (time_embed_dim is not None))
        self.token_film_mode = token_film_mode
        self.gate_residual = bool(gate_residual and (time_embed_dim is not None))
        self.gate_type = gate_type

        self.proj_in = nn.Conv2d(channels, self.reduced_channels, kernel_size=1)
        self.gn_in = nn.GroupNorm(
            _choose_gn_groups(self.reduced_channels, max_gn_groups),
            self.reduced_channels,
        )

        # 2) token FiLM (optional)
        if self.token_film:
            if self.token_film_mode == "shared":
                self.token_film_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, 2 * self.reduced_channels),
                )
                _zero_module(self.token_film_mlp[-1])
            elif self.token_film_mode == "per_token":
                self.token_film_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, 2 * self.K * self.reduced_channels),
                )
                _zero_module(self.token_film_mlp[-1])
            else:
                raise ValueError(f"Unknown token_film_mode={token_film_mode}")

        self.drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

        self.proj_out = _zero_module(nn.Conv2d(self.reduced_channels, channels, kernel_size=1))
        self.gn_out = nn.GroupNorm(
            _choose_gn_groups(channels, max_gn_groups),
            channels,
        )

        # residual gate with temb (optional)
        if self.gate_residual:
            out_dim = 1 if gate_type == "scalar" else channels
            self.gate_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, out_dim))
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.constant_(self.gate_mlp[-1].bias, -2.0)

    @staticmethod
    def _softmin_spatial(x_flat: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Differentiable approximation of per-channel spatial min.
        x_flat: (B, Cr, N)
        returns: (B, Cr, 1)
        """
        # softmin = -1/beta * logsumexp(-beta*x)
        return -(1.0 / beta) * torch.logsumexp(-beta * x_flat, dim=-1, keepdim=True)

    def _grad_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Cr, H, W) float32
        returns: (B, Cr, 1) mean gradient magnitude per channel
        """
        # finite differences
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]  # (B,Cr,H-1,W)
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]  # (B,Cr,H,W-1)

        # pad back to (H,W)
        gx = F.pad(dx, (0, 0, 0, 1))  # pad bottom
        gy = F.pad(dy, (0, 1, 0, 0))  # pad right

        mag = torch.sqrt(gx * gx + gy * gy + self.eps)  # (B,Cr,H,W)
        ge = mag.mean(dim=(2, 3), keepdim=True)         # (B,Cr,1)
        return ge

    def _compute_tokens(self, x_r: torch.Tensor) -> torch.Tensor:
        """
        x_r: (B, Cr, H, W)
        returns tokens S: (B, K, Cr) float32
        """
        B, Cr, H, W = x_r.shape
        N = H * W

        xr = x_r.float()
        x_flat = xr.view(B, Cr, N)  # (B,Cr,N)

        mu = x_flat.mean(dim=-1, keepdim=True)                        # (B,Cr,1)
        xc = x_flat - mu
        var = (xc * xc).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)                              # (B,Cr,1)

        # DCP-like: per-channel spatial soft-min (stable & differentiable)
        dcp = self._softmin_spatial(x_flat, beta=self.softmin_beta)   # (B,Cr,1)

        # GradEnergy: per-channel mean gradient magnitude
        ge = self._grad_energy(xr).view(B, Cr, 1)                     # (B,Cr,1)

        # stack to (B,Cr,K) then transpose to (B,K,Cr)
        S = torch.cat([mu, std, dcp, ge], dim=-1)                     # (B,Cr,4)
        S = S.transpose(1, 2).contiguous()                            # (B,4,Cr)
        return S

    def _film_tokens(self, S: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        S: (B, K, Cr) float32
        temb: (B, T)
        """
        B, K, Cr = S.shape
        film = self.token_film_mlp(temb).float()

        if self.token_film_mode == "shared":
            gamma, beta = film.chunk(2, dim=-1)  # (B,Cr),(B,Cr)
            gamma = gamma[:, None, :]            # (B,1,Cr)
            beta = beta[:, None, :]              # (B,1,Cr)
        else:
            film = film.view(B, 2, K, Cr)
            gamma = film[:, 0]                   # (B,K,Cr)
            beta = film[:, 1]                    # (B,K,Cr)

        # identity-friendly FiLM
        return (1.0 + gamma) * S + beta

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        dtype = x.dtype

        # reduce + GN
        x_r = self.gn_in(self.proj_in(x))                 # (B,Cr,H,W)

        # tokens
        S = self._compute_tokens(x_r)                     # (B,4,Cr) float32
        if self.token_film and temb is not None:
            S = self._film_tokens(S, temb)

        # attention distribute tokens back to spatial
        Cr = x_r.shape[1]
        N = H * W
        Q = x_r.float().view(B, Cr, N).transpose(1, 2).contiguous()  # (B,N,Cr)

        scores = (Q @ S.transpose(1, 2)) / math.sqrt(Cr)             # (B,N,4)
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        O = attn @ S                                                 # (B,N,Cr)
        O = O.transpose(1, 2).contiguous().view(B, Cr, H, W)          # (B,Cr,H,W)

        out = self.proj_out(O.to(dtype))
        out = self.gn_out(out)

        # residual gate
        if self.gate_residual and temb is not None:
            g = self.gate_mlp(temb)
            if self.gate_type == "scalar":
                g = torch.sigmoid(g).view(B, 1, 1, 1)
            else:
                g = torch.sigmoid(g).view(B, C, 1, 1)
            out = out * g

        return x + out