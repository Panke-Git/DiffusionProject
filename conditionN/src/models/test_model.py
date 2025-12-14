"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: test_model.py
    @Time: 2025/12/11 23:46
    @Email: None
"""
import torch
from cond_unet_ddpm import SinusoidalPosEmb

def test_SinusoidalPosEmb():
    t1 = torch.tensor([[1.0], [10.0], [50.0]], dtype=torch.float32)  # B = 3
    pos_emb_layer = SinusoidalPosEmb(dim=8)  # 输出维度设成 8，你可以改成你想要的

    emb1 = pos_emb_layer(t1)
    print("t1 shape:", t1.shape)
    print("emb1 shape:", emb1.shape)
    print("emb1:", emb1)


def test_ResBlock():
    t1 = torch.tensor([[1.0], [10.0], [50.0]], dtype=torch.float32)


if __name__ == '__main__':
    test_SinusoidalPosEmb()
