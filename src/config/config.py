"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT (adapted to user's style)
    @FileName: config.py
    @Time: 2025/11/09
    @Email: None
"""
from __future__ import annotations
import yaml
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict

def _dict_to_ns(d: Dict[str, Any]) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_ns(v))
        else:
            setattr(ns, k, v)
    return ns

class Config:
    """YAML -> 对象（点访问），保持与用户风格一致：Config.load(path)."""
    @staticmethod
    def load(path: str | Path) -> SimpleNamespace:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return _dict_to_ns(cfg)
