"""
    @Project: DiffusionProject
    @Author: paxton
    @FileName： gen_val_samples.py
    @Date：2025/12/9 21:09
    @OS：
    @Email: None
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从验证集中随机选若干张图像，用训练好的 pt 模型生成增强结果并保存到本地。

使用示例：
    python gen_val_samples.py \
        --ckpt ./checkpoints/best_psnr.pt \
        --val-input-dir ./data/val/input \
        --val-gt-dir ./data/val/gt \
        --out-dir ./vis_samples \
        --num-samples 5
"""

import os
import argparse
import random
from pathlib import Path
from . import model as modellib
import torch
from torchvision import transforms
from PIL import Image


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ===================== 1. 构建 model + diffusion =====================

def build_model_and_diffusion(device):
    """
    TODO: 这里完全照抄你训练脚本里创建 model 和 diffusion 的那几行。

    例如如果你训练代码里是这样的（示意）：
        from myproj.script_util import create_model_and_diffusion, create_config
        cfg = create_config(...)
        model, diffusion = create_model_and_diffusion(cfg)
        model.to(device)

    那就把那几行搬过来，最后 return model, diffusion
    """
    raise modellib.Unet(
        in_channels=3,
        base_channels=64,
        channel_mults=[1, 2, 4, 8],
        num_res_blocks=2,
        dropout=0.1,
        out_channels=3,
    )


def load_checkpoint_into_model(model, ckpt_path, device):
    """
    从 pt 文件里把模型权重加载进来。
    尝试兼容几种常见 key：'ema_model', 'model_ema', 'model', 'state_dict' 等。
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = None
    if isinstance(ckpt, dict):
        if "ema_model" in ckpt:
            print(">> 使用 ckpt['ema_model'] 作为采样权重")
            state_dict = ckpt["ema_model"]
        elif "model_ema" in ckpt:
            print(">> 使用 ckpt['model_ema'] 作为采样权重")
            state_dict = ckpt["model_ema"]
        elif "model" in ckpt:
            print(">> 使用 ckpt['model'] 作为采样权重")
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            print(">> 使用 ckpt['state_dict'] 作为采样权重")
            state_dict = ckpt["state_dict"]
        else:
            # 可能本身就是一个 state_dict
            print(">> ckpt 是一个 dict，直接当作 state_dict 使用")
            state_dict = ckpt
    else:
        # 直接是 state_dict
        print(">> ckpt 直接是 state_dict")
        state_dict = ckpt

    # 去掉可能的 'module.' 前缀
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    print(">> 权重加载完成:", ckpt_path)
    return model


# ===================== 2. 图像预处理 / 后处理 =====================

def list_images(dir_path):
    return [
        f for f in os.listdir(dir_path)
        if f.lower().endswith(IMG_EXTS)
    ]


def build_preprocess():
    """
    条件图像的预处理。
    一般 input 在 [0,1]，如果你的扩散模型里还有别的 Normalize，可以在这里加。
    """
    return transforms.ToTensor()


def tensor_m11_to_pil(img_tensor):
    """
    扩散采样结果通常是 [-1,1]，这里转回 [0,1] 再变成 PIL 图像。
    """
    img = img_tensor.detach().cpu()
    img = (img + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)
    return transforms.ToPILImage()(img)


# ===================== 3. 一个小 wrapper，让 diffusion 能吃条件 =====================

class ModelWrapper(torch.nn.Module):
    """
    把你原来的 model(x_t, t, cond) 包装成
    model_wrap(x_t, t, **model_kwargs) 的形式，方便 diffusion.p_sample_loop 调用。

    !!! TODO !!!
    如果你原本的模型函数签名不是 (x, t, cond)，
    在这里按你的实际情况改一下 forward。
    """
    def __init__(self, inner_model):
        super().__init__()
        self.inner_model = inner_model

    def forward(self, x, t, **model_kwargs):
        # 假设我们把条件图像放在 model_kwargs["cond"] 里
        cond = model_kwargs.get("cond", None)
        # 如果你训练时是 model(x, t, cond)，那下面这一行就对了，
        # 否则按你的模型实际参数顺序改：
        if cond is not None:
            return self.inner_model(x, t, cond)
        else:
            # 无条件情况（比如你做无条件采样时）
            return self.inner_model(x, t)


# ===================== 4. 对单张 cond 图像做一次完整采样 =====================

@torch.no_grad()
def sample_one_with_cond(diffusion, model, cond, num_steps=None):
    """
    cond: (1, C, H, W) 的条件图像 tensor，范围 [0,1] 或做过你训练时的归一化。
    diffusion: 你的 GaussianDiffusion 对象
    model: 已经包了 ModelWrapper 的模型

    返回：采样得到的增强图像 (1, 3, H, W)，范围通常在 [-1, 1]
    """
    device = cond.device
    B, C, H, W = cond.shape
    assert B == 1, "这里先只实现了 batch_size=1 的采样，够你可视化用了。"

    # 如果你的 diffusion 有 num_timesteps，可以按它来
    if num_steps is None and hasattr(diffusion, "num_timesteps"):
        num_steps = diffusion.num_timesteps
    print(f">> 使用 {num_steps} 步扩散采样")

    # 使用你 diffusion 里的标准接口：p_sample_loop 或 ddim_sample_loop
    # 官方 improved-diffusion 风格示例：
    samples = diffusion.p_sample_loop(
        model,
        (B, 3, H, W),           # 采样图像的 shape
        clip_denoised=True,
        model_kwargs={"cond": cond},   # 把条件传进去，ModelWrapper 里会取出来
    )
    # 如果你用的是 ddim_sample_loop，把上面那行换掉就行

    return samples  # (B,3,H,W), [-1,1]


# ===================== 5. 主流程 =====================

def main():
    parser = argparse.ArgumentParser(
        description="从验证集随机选若干图像，用 pt 模型生成增强结果"
    )
    parser.add_argument("--ckpt", type=str,
                        default='/public/home/hnust15874739861/pro/DiffusionProject/condition/runs/UNet/20251209-0342/checkpoints',
                        help="模型 ckpt 路径，如 ./checkpoints/best_psnr.pt")
    parser.add_argument("--val-input-dir",
                        default="/public/home/hnust15874739861/pro/publicdata/LSUI19/Val/input",
                        help="验证集 input 图像所在目录")
    parser.add_argument("--val-gt-dir",
                        default='/public/home/hnust15874739861/pro/publicdata/LSUI19/Val/GT',
                        type=str,
                        help="验证集 GT 图像所在目录（可选）")
    parser.add_argument("--out-dir", type=str,
                        default='//public/home/hnust15874739861/pro/DiffusionProject/condition/output',
                        help="输出保存目录")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="随机生成的图片数量")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机数种子（方便复现）")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 构建 model & diffusion
    base_model, diffusion = build_model_and_diffusion(device)
    base_model = load_checkpoint_into_model(base_model, args.ckpt, device)
    base_model.to(device)
    base_model.eval()

    model = ModelWrapper(base_model).to(device)
    model.eval()

    # 2) 列出验证集 input 文件
    input_files = list_images(args.val_input_dir)
    if not input_files:
        raise RuntimeError(f"在验证集输入目录 {args.val_input_dir} 中没有找到图片")

    if args.num_samples > len(input_files):
        print("[Warn] num_samples 大于验证集数量，只会使用全部验证集")
        num = len(input_files)
    else:
        num = args.num_samples

    sampled_files = random.sample(input_files, num)
    preprocess = build_preprocess()

    print(f">> 将从验证集中随机选择 {num} 张图像进行扩散采样，可视化输出...")

    for idx, filename in enumerate(sampled_files, start=1):
        in_path = os.path.join(args.val_input_dir, filename)
        basename, _ = os.path.splitext(filename)

        # 读入条件图像
        img_cond = Image.open(in_path).convert("RGB")
        cond_tensor = preprocess(img_cond).unsqueeze(0).to(device)  # (1,3,H,W)

        # 3) 采样
        samples = sample_one_with_cond(diffusion, model, cond_tensor, num_steps=args.steps)
        out_tensor = samples[0]  # (3,H,W)

        # 4) 保存：input / output / gt (如果有)
        out_img = tensor_m11_to_pil(out_tensor)

        in_save_path = os.path.join(args.out_dir, f"{idx:02d}_{basename}_input.png")
        out_save_path = os.path.join(args.out_dir, f"{idx:02d}_{basename}_ddpm.png")

        img_cond.save(in_save_path)
        out_img.save(out_save_path)

        # 保存 GT（可选）
        if args.val_gt_dir:
            gt_path = os.path.join(args.val_gt_dir, filename)
            if os.path.exists(gt_path):
                img_gt = Image.open(gt_path).convert("RGB")
                gt_save_path = os.path.join(args.out_dir, f"{idx:02d}_{basename}_gt.png")
                img_gt.save(gt_save_path)

        print(f"[{idx}/{num}] 已完成 {filename}")

    print("全部完成，结果保存在：", args.out_dir)


if __name__ == "__main__":
    main()
