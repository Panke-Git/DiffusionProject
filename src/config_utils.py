"""
    @Project: DiffusionProject
    @Author: paxton
    @FileName： config_utils.py
    @Date：2025/11/17 23:10
    @OS：
    @Email: None
"""
# config_utils.py
import yaml
from types import SimpleNamespace


def dict_to_namespace(d):
    """把嵌套 dict 递归转换成 SimpleNamespace，支持点号访问."""
    if isinstance(d, dict):
        ns = SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, dict_to_namespace(v))
        return ns
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


def load_config(cfg_path: str):
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return dict_to_namespace(cfg_dict)
