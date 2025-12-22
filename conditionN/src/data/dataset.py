"""
    @Project: UnderwaterImageEnhanced
    @Author: Panke
    @FileName: dataset.py
    @Time: 2025/5/20 00:20
    @Email: None
"""
import os
import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
import random

from .split_data import is_image_file
import torch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def _A_RandomResizedCrop(h, w, **kwargs):
    # albumentations 1.x: height, width
    # albumentations 2.x: size=(h, w)
    try:
        return A.RandomResizedCrop(height=h, width=w, **kwargs)
    except Exception:
        return A.RandomResizedCrop(size=(h, w), **kwargs)

def _A_Resize(h, w, **kwargs):
    # 大多数版本 Resize 仍是 height/width，但也做个兜底更安全
    try:
        return A.Resize(height=h, width=w, **kwargs)
    except Exception:
        return A.Resize(size=(h, w), **kwargs)



class DataReader(Dataset):
    def __init__(self, img_dir, input='input', target='GT', mode='train', ori=False, img_options=None):
        super().__init__()
        assert img_options is not None and 'h' in img_options and 'w' in img_options

        self.img_dir = img_dir
        self.input_sub = input
        self.target_sub = target
        self.mode = mode
        self.img_options = img_options

        input_dir = os.path.join(img_dir, input)
        target_dir = os.path.join(img_dir, target)

        input_files = sorted([x for x in os.listdir(input_dir) if is_image_file(x)])
        target_files = sorted([x for x in os.listdir(target_dir) if is_image_file(x)])

        target_map = {os.path.basename(x): os.path.join(target_dir, x) for x in target_files}

        self.samples = []
        for x in input_files:
            name = os.path.basename(x)
            inp_path = os.path.join(input_dir, x)
            gt_path = target_map.get(name, None)
            if gt_path is None:
                raise FileNotFoundError(f"GT file not found for {inp_path} -> {os.path.join(target_dir, name)}")
            self.samples.append((inp_path, gt_path))

        if len(self.samples) == 0:
            raise RuntimeError("No paired samples found!")

        if self.mode == 'train':
            self.transform = A.Compose([
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                ], p=0.3),
                A.RandomRotate90(p=0.3),
                A.Rotate(p=0.3),
                A.Transpose(p=0.3),
                _A_RandomResizedCrop(img_options['h'], img_options['w']),
            ], additional_targets={'target': 'image'})
            self.degrade = A.Compose([A.NoOp()])
        else:
            if ori:
                self.transform = A.Compose([A.NoOp()], additional_targets={'target': 'image'})
            else:
                self.transform = A.Compose([
                    _A_Resize(img_options['h'], img_options['w'])
                ], additional_targets={'target': 'image'})
            self.degrade = A.Compose([A.NoOp()])

    def __len__(self):
        return len(self.samples)

    def _to_tensor_minus1_1(self, img_np):
        # 关键：albumentations 可能返回负stride/非连续视图，这里必须转成连续内存
        img_np = np.ascontiguousarray(img_np)

        # 关键：copy 一份，避免 from_numpy 绑定到不可 resize 的 numpy storage
        t = torch.from_numpy(img_np.copy()).permute(2, 0, 1).contiguous().float() / 255.0
        t = t * 2.0 - 1.0
        return t

    def __getitem__(self, index):
        inp_path, gt_path = self.samples[index]

        inp_img = Image.open(inp_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        inp_np = np.array(inp_img)
        gt_np = np.array(gt_img)

        transformed = self.transform(image=inp_np, target=gt_np)

        if self.mode == 'train':
            inp_np2 = self.degrade(image=transformed['image'])['image']
        else:
            inp_np2 = transformed['image']

        gt_np2 = transformed['target']

        inp_t = self._to_tensor_minus1_1(inp_np2)
        gt_t = self._to_tensor_minus1_1(gt_np2)

        return {
            "input": inp_t,
            "gt": gt_t,
            "input_path": str(inp_path),
            "gt_path": str(gt_path),
        }
