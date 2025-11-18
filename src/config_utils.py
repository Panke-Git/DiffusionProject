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
    print(cfg_dict)  # 打印配置文件的内容

    if 'TRAIN' in cfg_dict:
        for key in ['LR', 'BATCH_SIZE', 'IMG_H', 'IMG_W', 'EPOCHS', 'TIMESTEPS', 'NUM_WORKERS', 'RECON_LAMBDA']:  # 按需添加字段
            if key in cfg_dict['TRAIN']:
                # 检查是否是字符串类型，如果是，则转换为浮动数值
                if isinstance(cfg_dict['TRAIN'][key], str):
                    cfg_dict['TRAIN'][key] = float(cfg_dict['TRAIN'][key])

    return dict_to_namespace(cfg_dict)
