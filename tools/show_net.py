
"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: show_net.py
    @Time: 2026/1/20 22:34
    @Email: None
"""


import torch
from model.MIPTVDepthEstimator import Block1_MIPTV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Block1_MIPTV =", Block1_MIPTV)
print("type(Block1_MIPTV) =", type(Block1_MIPTV))
net = Block1_MIPTV(3, 3, 32, [1,2,3,4], 1)
net = net.to(device).eval()

x = torch.randn(32, 3, 256, 256, device=device)
with torch.no_grad():
    y = net(x, x)

print("output shape:", y.shape)
