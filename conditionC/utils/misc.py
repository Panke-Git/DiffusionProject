# coding=utf-8
"""
    @Project: 
    @Author: PyCharm
    @FileName： misc.py
    @Date：2025/12/25 16:15
    @Email: None
"""

import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())