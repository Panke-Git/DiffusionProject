"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT
    @FileName: record_utils.py
    @Time: 2025/11/09
    @Email: None
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

def make_train_path(root: str | Path, model_name: str, start_time: str) -> tuple[Path, Path]:
    """
    创建训练记录路径：
      root/{model_name}/{start_time}/
                 ├── ckpts/
                 ├── samples/
                 └── logs/
    返回：record_path, best_ckpt_dir
    """
    root = Path(root) / 'exp_record' / model_name / start_time
    (root / "ckpts").mkdir(parents=True, exist_ok=True)
    (root / "samples").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    return root, root / "ckpts"

def save_train_config(record_path: Path, **kwargs) -> Path:
    """保存一次训练的配置到 JSON 文件，便于复现。"""
    path = record_path / "train_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(kwargs, f, indent=2, ensure_ascii=False)
    return path
