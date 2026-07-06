import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_xyz(rgb):
    mask = rgb > 0.04045
    rgb_linear = torch.where(
        mask,
        ((rgb + 0.055) / 1.055).clamp(min=0) ** 2.4,
        rgb / 12.92,
    )

    r = rgb_linear[:, 0:1]
    g = rgb_linear[:, 1:2]
    b = rgb_linear[:, 2:3]

    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return torch.cat([x, y, z], dim=1)


def xyz_to_rgb(xyz):
    x = xyz[:, 0:1]
    y = xyz[:, 1:2]
    z = xyz[:, 2:3]

    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    rgb_linear = torch.cat([r, g, b], dim=1).clamp(min=0.0)
    mask = rgb_linear > 0.0031308
    rgb = torch.where(
        mask,
        1.055 * rgb_linear.clamp(min=0) ** (1 / 2.4) - 0.055,
        12.92 * rgb_linear,
    )
    return rgb.clamp(0.0, 1.0)


def xyz_to_lab(xyz, eps=1e-6):
    x = xyz[:, 0:1] / 0.95047
    y = xyz[:, 1:2] / 1.00000
    z = xyz[:, 2:3] / 1.08883
    delta = 6 / 29

    def f(t):
        return torch.where(
            t > delta ** 3,
            t.clamp(min=eps) ** (1 / 3),
            t / (3 * delta ** 2) + 4 / 29,
        )

    fx = f(x)
    fy = f(y)
    fz = f(z)

    l_channel = (116 * fy - 16) / 100.0
    a_channel = (500 * (fx - fy) + 128.0) / 255.0
    b_channel = (200 * (fy - fz) + 128.0) / 255.0
    return (
        l_channel.clamp(0.0, 1.0),
        a_channel.clamp(0.0, 1.0),
        b_channel.clamp(0.0, 1.0),
    )


def lab_to_xyz(l_channel, a_channel, b_channel):
    l_channel = l_channel * 100.0
    a_channel = a_channel * 255.0 - 128.0
    b_channel = b_channel * 255.0 - 128.0

    fy = (l_channel + 16) / 116
    fx = fy + a_channel / 500
    fz = fy - b_channel / 200
    delta = 6 / 29

    def f_inv(t):
        return torch.where(
            t > delta,
            t ** 3,
            3 * delta ** 2 * (t - 4 / 29),
        )

    x = 0.95047 * f_inv(fx)
    y = 1.00000 * f_inv(fy)
    z = 1.08883 * f_inv(fz)
    return torch.cat([x, y, z], dim=1).clamp(min=0.0)


def rgb_to_lab(rgb):
    return xyz_to_lab(rgb_to_xyz(rgb))


def lab_to_rgb(l_channel, a_channel, b_channel):
    return xyz_to_rgb(lab_to_xyz(l_channel, a_channel, b_channel))


class DepthGuidedLuminanceGate(nn.Module):
    """Depth-guided luminance-only correction in Lab space."""

    def __init__(
        self,
        hidden_channels=32,
        alpha_init=0.05,
        input_range="0_1",
        output_range="0_1",
    ):
        super().__init__()

        assert input_range in ["0_1", "-1_1"]
        assert output_range in ["0_1", "-1_1"]

        self.input_range = input_range
        self.output_range = output_range

        in_channels = 4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.residual_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 3, 1, 1),
            nn.Tanh(),
        )

        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def _to_01(self, x):
        if self.input_range == "-1_1":
            return (x + 1.0) / 2.0
        return x

    def _from_01(self, x):
        if self.output_range == "-1_1":
            return x * 2.0 - 1.0
        return x

    @staticmethod
    def normalize_depth(depth):
        batch = depth.shape[0]
        depth_flat = depth.view(batch, -1)
        depth_min = depth_flat.min(dim=1)[0].view(batch, 1, 1, 1)
        depth_max = depth_flat.max(dim=1)[0].view(batch, 1, 1, 1)
        depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        return depth_norm.clamp(0.0, 1.0)

    def forward(self, x_raw, i_main, depth):
        module_dtype = self.alpha.dtype
        module_device = self.alpha.device
        x_raw = x_raw.to(device=module_device, dtype=module_dtype)
        i_main = i_main.to(device=module_device, dtype=module_dtype)
        depth = depth.to(device=module_device, dtype=module_dtype)

        x_raw = self._to_01(x_raw).clamp(0.0, 1.0)
        i_main = self._to_01(i_main).clamp(0.0, 1.0)
        depth = self.normalize_depth(depth)

        if depth.shape[-2:] != i_main.shape[-2:]:
            depth = F.interpolate(
                depth,
                size=i_main.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        raw_l, _, _ = rgb_to_lab(x_raw)
        main_l, main_a, main_b = rgb_to_lab(i_main)
        diff_l = torch.abs(raw_l - main_l)
        feat_in = torch.cat([raw_l, main_l, depth, diff_l], dim=1)

        feat = self.encoder(feat_in)
        lum_residual = self.residual_head(feat)
        depth_gate = self.gate_head(feat)
        alpha = torch.clamp(self.alpha, 0.0, 1.0)

        out_l = (main_l + alpha * depth_gate * lum_residual).clamp(0.0, 1.0)
        out_rgb = lab_to_rgb(out_l, main_a, main_b).clamp(0.0, 1.0)
        return self._from_01(out_rgb)
