"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: demo_time_emd_show.py
    @Time: 2025/12/13 14:47
    @Email: None
"""

import math
import torch
import torch.nn as nn

# ====== 1. 你之前的时间编码（原样搬过来） ======
class SinusoidalPosEmb(nn.Module):
    """
    标准 DDPM / Transformer 风格的时间步位置编码:
    输入:  t (B,)  或 (B,1)
    输出:  (B, dim)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        :param t: (B,) 或 (B, 1)，int 或 float 都可以
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B,1)
        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)  # (half_dim,)
        args = t.float() * freqs                     # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        if self.dim % 2 == 1:  # 如果 dim 是奇数，补一维 0
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


def main():
    torch.set_printoptions(precision=4, sci_mode=False)

    # ====== 2. 假设一个很小的场景 ======
    B, C_out, H, W = 2, 3, 2, 2
    time_emb_dim = 4

    # 构造一个很简单的特征图 h，数值好看一点
    # h[0] 和 h[1] 不同，方便你区分
    h = torch.arange(B * C_out * H * W, dtype=torch.float32).view(B, C_out, H, W) / 10.0

    # 两个样本各自的时间步 t[0] = 1, t[1] = 10
    t = torch.tensor([1, 10], dtype=torch.long)

    # ====== 3. 时间编码 + MLP ======
    time_emb_layer = SinusoidalPosEmb(time_emb_dim)
    # 一个很简单的 MLP：Linear -> SiLU -> Linear
    time_mlp = nn.Sequential(
        nn.Linear(time_emb_dim, 8),
        nn.SiLU(),
        nn.Linear(8, C_out)  # 输出维度 = 通道数
    )

    print("=== 原始特征 h ===")
    print("h.shape:", h.shape)
    print(h)

    print("\n=== 时间步 t ===")
    print("t.shape:", t.shape)
    print(t)

    # 1) t -> 正余弦时间编码
    t_emb = time_emb_layer(t)
    print("\n=== 时间编码 t_emb = SinusoidalPosEmb(t) ===")
    print("t_emb.shape:", t_emb.shape)  # (B, time_emb_dim)
    print(t_emb)

    # 2) 时间编码送入 MLP，得到每个样本的一组通道偏置
    t_vec = time_mlp(t_emb)  # (B, C_out)
    print("\n=== MLP 输出的时间向量 t_vec ===")
    print("t_vec.shape:", t_vec.shape)  # (B, C_out)
    print(t_vec)

    # 3) 扩展成 (B, C_out, 1, 1) 用于广播
    t_vec_expanded = t_vec[:, :, None, None]
    print("\n=== 扩展后的时间向量 t_vec_expanded ===")
    print("t_vec_expanded.shape:", t_vec_expanded.shape)  # (B, C_out, 1, 1)
    print(t_vec_expanded)

    # 4) 加到特征图上
    h_new = h + t_vec_expanded

    print("\n=== 加时间偏置后的特征 h_new = h + t_vec_expanded ===")
    print("h_new.shape:", h_new.shape)
    print(h_new)

    # 为了更直观地看每个样本、每个通道分别发生了什么：
    for b in range(B):
        print(f"\n------ 样本 b = {b}, 对应时间步 t = {t[b].item()} ------")
        print("对应的 t_vec[b]:", t_vec[b])  # 每个通道一个偏置

        for c in range(C_out):
            print(f"\n  通道 c = {c}")
            print("  原始 h[b, c, :, :]:")
            print(h[b, c])
            print("  偏置 t_vec[b, c]:", float(t_vec[b, c]))
            print("  新的 h_new[b, c, :, :]:")
            print(h_new[b, c])


if __name__ == "__main__":
    main()




