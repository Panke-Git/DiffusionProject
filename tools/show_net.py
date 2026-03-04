
"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: show_net.py
    @Time: 2026/1/20 22:34
    @Email: None
"""

from model.ddpm_modules.unet import UNet
import torch


if __name__ == "__main__":
    # 创建模型
    model = UNet(
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        image_size=256
    )

    # 打印网络结构
    print(model)

    # 构造输入
    x = torch.randn(1, 6, 128, 128)   # batch=1, 6通道, 128x128
    t = torch.randint(0, 1000, (1,))  # diffusion timestep

    # 前向传播
    out = model(x, t)

    # 打印输入输出shape
    print("input shape:", x.shape)
    print("output shape:", out.shape)