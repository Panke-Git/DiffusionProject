# coding=utf-8
"""
    @Project: 
    @Author: PyCharm
    @FileName： ema.py
    @Date：2025/12/25 16:15
    @Email: None
"""

from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class EMAConfig:
    enable: bool = True
    decay: float = 0.9999
    update_every: int = 1
    warmup_steps: int = 500

class EMA:
    def __init__(self, model: nn.Module, cfg: EMAConfig):
        self.cfg = cfg
        self.step = 0
        self.ema_model = self._clone_model(model)
        self.ema_model.eval()

        # 确保 EMA 模型在与原模型相同的 device
        dev = next(model.parameters()).device
        self.ema_model.to(dev)

    def _clone_model(self, model: nn.Module) -> nn.Module:
        ema = type(model)(model.cfg) if hasattr(model, "cfg") else None
        if ema is None:
            raise RuntimeError("Model must be re-creatable via type(model)(model.cfg).")
        ema.load_state_dict(model.state_dict(), strict=True)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model: nn.Module):
        if not self.cfg.enable:
            return
        self.step += 1
        if self.step < self.cfg.warmup_steps:
            self.ema_model.load_state_dict(model.state_dict(), strict=True)
            return
        if (self.step % self.cfg.update_every) != 0:
            return

        decay = self.cfg.decay
        msd = model.state_dict()
        esd = self.ema_model.state_dict()

        for k in esd.keys():
            if esd[k].dtype.is_floating_point:
                esd[k].mul_(decay).add_(msd[k].detach(), alpha=1 - decay)
            else:
                esd[k].copy_(msd[k])

    def state_dict(self):
        return {
            "step": self.step,
            "ema": self.ema_model.state_dict()
        }

    def load_state_dict(self, state):
        self.step = int(state.get("step", 0))
        self.ema_model.load_state_dict(state["ema"], strict=True)
