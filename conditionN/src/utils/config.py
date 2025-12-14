"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: config.py
    @Time: 2025/12/13 11:06
    @Email: None
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml


def _dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def load_config(path: str) -> SimpleNamespace:
    """
    从 YAML 文件加载配置, 返回一个可以点号访问的对象:
        cfg.DATA.train_input_dir
        cfg.MODEL.base_channels
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")
    return _dict_to_namespace(raw)


if __name__ == "__main__":
    # 简单自测: 你可以先创建 configs/underwater_ddpm.yaml 再跑
    try:
        cfg = load_config("../config/underwater_ddpm.yaml")
        print("train_input_dir:", cfg.DATA.train_input_dir)
        print("image_size:", cfg.DATA.image_size)
        print("lr", cfg.TRAIN.lr)
    except FileNotFoundError:
        print("请先创建 configs/underwater_ddpm.yaml 再运行自测.")
