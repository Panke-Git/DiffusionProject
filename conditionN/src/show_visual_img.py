"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: show_visual_img.py
    @Time: 2025/12/20 15:54
    @Email: None
"""
import sys
from pathlib import Path
import argparse
import time
import math

from datetime import datetime

import copy

from .utils.generate_best_grids import auto_generate_best_grids

# 保证能找到 src 目录下的包 (models, data, diffusion, utils)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from .utils.train_utils import *

from .utils.config import load_config
from .utils.record_utils import *

from utils.generate_best_grids import auto_generate_best_grids
from utils.config import load_config
from typing import List
import argparse
import torch
from .utils.validation_random_cond import *


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional DDPM for underwater enhancement")
    parser.add_argument(
        "--config",
        type=str,
        default="/public/home/hnust15874739861/pro/DiffusionProject/conditionN/src/config/underwater_ddpm.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use, e.g. 'cuda', 'cuda:0', or 'cpu'",
    )
    return parser.parse_args()




def show_random_conda():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    base_dir= "/public/home/hnust15874739861/pro/DiffusionProject/conditionN/record_data/Base_DDPM/20251223_125543/"
    best_path = base_dir + 'best_result'
    output_path = base_dir + "view_out"
    auto_generate_best_grids(
        cfg=cfg,
        device=device,
        ckpt_dir=best_path,
        out_dir=output_path,
        t_start=300,
        step=50,
        n_rows=10,
        seed=42,
        use_ema=True,
        strict=True
    )
show_random_conda()

# show_visual_img([800, 1000])
