"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: depth_guided_adaptive_regularization.py
    @Time: 2026/3/24 22:52
    @Email: None
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    b = x.shape[0]
    x_flat = x.view(b, -1)
    x_min = x_flat.min(dim=1).values.view(b, 1, 1, 1)
    x_max = x_flat.max(dim=1).values.view(b, 1, 1, 1)
    return (x - x_min) / (x_max - x_min + eps)


def image_to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def sobel_grad(x: torch.Tensor):
    """
    x: (B,1,H,W)
    """
    device = x.device
    dtype = x.dtype

    kx = torch.tensor(
        [[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]],
        dtype=dtype, device=device
    ).unsqueeze(0)  # (1,1,3,3)

    ky = torch.tensor(
        [[[-1, -2, -1],
          [ 0,  0,  0],
          [ 1,  2,  1]]],
        dtype=dtype, device=device
    ).unsqueeze(0)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    gm = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return gx, gy, gm


def forward_diff(x: torch.Tensor):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


class DepthGuidedAdaptiveRegularizer(nn.Module):
    """
    用于 DiffWater 训练阶段的附加正则项。

    输入:
        pred_x0 : 网络恢复出的清晰图像, (B,3,H,W)
        depth   : 深度图, (B,1,H,W), [0,1]
        guide   : 引导图, 建议使用 degraded image 或 3C 条件图, (B,3,H,W)

    输出:
        total_reg_loss, loss_dict
    """
    def __init__(
        self,
        lambda_tv: float = 1.0,
        lambda_edge: float = 0.3,
        depth_gain: float = 1.5,
        edge_suppress_tau: float = 0.15,
    ):
        super().__init__()
        self.lambda_tv = float(lambda_tv)
        self.lambda_edge = float(lambda_edge)
        self.depth_gain = float(depth_gain)
        self.edge_suppress_tau = float(edge_suppress_tau)

    def forward(
        self,
        pred_x0: torch.Tensor,
        depth: torch.Tensor,
        guide: torch.Tensor,
    ):
        pred_x0 = torch.clamp(pred_x0, 0.0, 1.0)
        depth = normalize_01(depth)
        guide_gray = image_to_gray(torch.clamp(guide, 0.0, 1.0))
        pred_gray = image_to_gray(pred_x0)

        # ----------------------------------------
        # 1) 深度引导的空间变权 TV
        # 深度越大 => 正则越强
        # 边缘越强 => 正则越弱
        # ----------------------------------------
        _, _, guide_gm = sobel_grad(guide_gray)
        edge_suppress = torch.exp(-guide_gm / self.edge_suppress_tau)

        depth_weight = 1.0 + self.depth_gain * depth
        spatial_weight = depth_weight * edge_suppress

        dx, dy = forward_diff(pred_gray)
        wx = spatial_weight[:, :, :, :-1]
        wy = spatial_weight[:, :, :-1, :]

        tv_x = (wx * dx.abs()).mean()
        tv_y = (wy * dy.abs()).mean()
        adaptive_tv = tv_x + tv_y

        # ----------------------------------------
        # 2) 深度边缘一致性
        # 希望图像结构边缘与深度边缘更一致
        # ----------------------------------------
        _, _, pred_gm = sobel_grad(pred_gray)
        _, _, depth_gm = sobel_grad(depth)

        pred_gm_n = normalize_01(pred_gm)
        depth_gm_n = normalize_01(depth_gm)

        # 只在深度边缘更明显的位置强化
        depth_edge_mask = (depth_gm_n > 0.1).float()
        edge_align = (depth_edge_mask * (pred_gm_n - depth_gm_n).abs()).mean()

        total = self.lambda_tv * adaptive_tv + self.lambda_edge * edge_align

        loss_dict = {
            "reg_total": total.detach(),
            "reg_adaptive_tv": adaptive_tv.detach(),
            "reg_edge_align": edge_align.detach(),
        }
        return total, loss_dict