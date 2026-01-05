"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: plot.py
    @Time: 2026/1/5 23:27
    @Email: None
"""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_train_val_loss(history: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    epochs = [int(h["epoch"]) for h in history]
    train_losses = [float(h["train_loss"]) for h in history]
    val_losses = [float(h["val_loss"]) for h in history]

    plt.figure()
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
