from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn


@dataclass
class EMAConfig:
    enabled: bool = True
    decay: float = 0.9999
    update_after_step: int = 0
    update_every: int = 1


class EMA:
    """Exponential Moving Average of model weights (common in diffusion training)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.register(model)

    @torch.no_grad()
    def register(self, model: nn.Module):
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.detach() + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}
