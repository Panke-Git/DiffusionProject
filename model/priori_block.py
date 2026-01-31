"""
    @Project: MIPTVDepthEstimator.py
    @Author: Panke
    @FileName: priori_block.py
    @Time: 2026/1/31 00:03
    @Email: None
"""

from MIPTVDepthEstimator import *

class Block1_MIPTV(nn.Module):
    """
    第一个基础的物理先验模块，将SeaDiff中的dpth用MIPTVDepthEstimator替换
    """
    def __init__(
        self,
        input_channels,
        output_channels,
        n_channels,
        ch_mults,
        n_blocks,
        eps=1e-6,
    ):
        super(Block1_MIPTV, self).__init__()
        self.beta_predictor = Beta_UNet(3, 3, n_channels, ch_mults, n_blocks)
        self.depth_estimator = MIPTVDepthEstimator()
        self.eps = float(eps)
        # self.block1 = Block1_MIPTV(3, 3, 32, [1, 2, 3, 4], 1)

    def forward(self, condition):
        pred_beta = self.beta_predictor(condition)
        depth = self.depth_estimator(condition)  # (B,1,H,W)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + self.eps)
        T_direct = torch.clamp((torch.exp(-pred_beta * depth)), 0, 1)
        T_scatter = torch.clamp((1 - torch.exp(-pred_beta * depth)), 0, 1)
        atm_light = [get_A(item) for item in condition]
        atm_light = torch.stack(atm_light).to(condition.device)
        J = torch.clamp(((condition - T_scatter * atm_light) / T_direct), 0, 1)
        return J


class Block2_MIPTV(nn.Module):
    """
    第二个基础的物理先验模块，将SeaDiff中的dpth用MIPTVDepthEstimator替换，带有x和condition
    """

    def __init__(self, n_channels, ch_mults, n_blocks,
                 force_size=256, eps=1e-6,
                 depth_kwargs=None):
        super().__init__()
        self.beta_predictor = Beta_UNet(3, 3, n_channels, ch_mults, n_blocks)
        self.depth_estimator = MIPTVDepthEstimator(**(depth_kwargs or {}))
        self.force_size = force_size
        self.eps = eps

    def forward(self, x, condition, t=None):
        # 1) 强制输入变成 256×256（如果你上游保证就是 256，可改成 assert 更干净）
        # if self.force_size is not None:
            # target = (self.force_size, self.force_size)
            # if x.shape[-2:] != target:
            #     x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
            # if condition.shape[-2:] != target:
            #     condition = F.interpolate(condition, size=target, mode="bilinear", align_corners=False)

        # 2) beta（每通道一个标量）：[B,3,1,1]
        pred_beta = self.beta_predictor(condition)

        # 3) 用你给的 depth estimator 生成 depth：[B,1,H,W] in [0,1]
        #    这里用 condition 生成 depth 最符合“从观测图提先验”的思路；
        #    如果你希望从 x 生成，把 condition 改成 x 即可。
        depth = self.depth_estimator(condition)

        # 4) 计算 T_direct / T_scatter（广播到 [B,3,H,W]）
        T_direct = torch.clamp(torch.exp(-pred_beta * depth), 0.0, 1.0)
        T_scatter = torch.clamp(1.0 - torch.exp(-pred_beta * depth), 0.0, 1.0)

        # 5) 大气光 A（保持你原来的 get_A 引入方式）
        atm_light = torch.stack([get_A(item) for item in condition]).to(condition.device)

        # 6) 反演得到 J（加 eps 避免除 0）
        J = (condition - T_scatter * atm_light) / (T_direct + self.eps)
        J = torch.clamp(J, 0.0, 1.0)

        # 输出保证 [B,3,256,256]
        return J