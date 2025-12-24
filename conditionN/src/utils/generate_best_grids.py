"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: generate_best_grids.py.py
    @Time: 2025/12/20 00:16
    @Email: None
"""
# src/generate_best_grids.py
import random
from pathlib import Path
from PIL import Image
import torch

# 你项目里应该已有 build_model_and_diffusion / build_dataloaders
from .train_utils import build_model_and_diffusion, build_dataloaders, load_checkpoint


def _to_uint8_chw(x: torch.Tensor) -> torch.Tensor:
    """
    x: (3,H,W) in [-1,1] float
    return: (3,H,W) uint8
    """
    x = ((x + 1) / 2).clamp(0, 1)
    x = (x * 255.0).round().to(torch.uint8)
    return x


def _save_grid_10x3(
    gt_list, inp_list, out_list,
    save_path: Path,
    pad: int = 0
):
    """
    gt_list/inp_list/out_list: list of (3,H,W) uint8 tensors, length=10
    输出：10行，每行 [GT | input | output]
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    assert len(gt_list) == len(inp_list) == len(out_list) == 10
    c, h, w = gt_list[0].shape

    W = 3 * w + 2 * pad * 4  # 左右padding简单一点
    H = 10 * h + 9 * pad * 2
    canvas = Image.new("RGB", (W, H))

    def paste(img_chw_u8, x, y):
        arr = img_chw_u8.permute(1, 2, 0).cpu().numpy()
        canvas.paste(Image.fromarray(arr), (x, y))

    y = 0
    for r in range(10):
        x = 0
        paste(gt_list[r], x, y)
        x += w + pad * 2
        paste(inp_list[r], x, y)
        x += w + pad * 2
        paste(out_list[r], x, y)
        y += h + pad * 2

    canvas.save(save_path)


@torch.no_grad()
def auto_generate_best_grids(
    cfg,
    device,
    ckpt_dir,
    out_dir,
    t_start: int = 200,
    step: int = 50,
    n_rows: int = 10,
    seed: int = 42,
    use_ema: bool = True,
    strict: bool = True,
):
    """
    训练结束后自动生成 3 张拼图：
      grid_best_loss.png / grid_best_psnr.png / grid_best_ssim.png
    每张是 10 行 * 3 列：GT | input | output

    参数：
      - ckpt_dir: 保存 best_loss.pt/best_psnr.pt/best_ssim.pt 的目录（就是你 best_path）
      - out_dir : 输出目录
      - t_start : sample_from_input 的强度（越大改变越大、也越慢）
    """
    ckpt_dir = Path(ckpt_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 固定随机性：保证每次生成的10张是同一批（便于对比）
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 1) 构建 val dataset（为了随机抽取 10 张）
    _, val_loader = build_dataloaders(cfg)
    dataset = val_loader.dataset
    n = len(dataset)
    if n < n_rows:
        raise RuntimeError(f"Val dataset size={n} < n_rows={n_rows}")

    idxs = random.sample(range(n), k=n_rows)

    # 2) 取出 10 张，组成 batch（更快）
    gts = []
    inps = []
    for idx in idxs:
        item = dataset[idx]
        if isinstance(item, dict):
            gt = item["gt"]
            inp = item["input"]
        else:
            # 兼容 tuple/list: (input, gt) 或 (gt, input) ——你按实际情况改一下
            inp, gt = item

        gts.append(gt)
        inps.append(inp)

    gt_b = torch.stack(gts, dim=0)    # (10,3,H,W) [-1,1]
    inp_b = torch.stack(inps, dim=0)  # (10,3,H,W) [-1,1]

    # 3) 针对每个 best ckpt：加载权重 -> 生成 -> 拼图保存
    tag_map = {
        "loss": ckpt_dir / "best_loss.pt",
        "psnr": ckpt_dir / "best_psnr.pt",
        "ssim": ckpt_dir / "best_ssim.pt",
    }

    for tag, ckpt_path in tag_map.items():
        if not ckpt_path.exists():
            print(f"[Preview] skip, not found: {ckpt_path}")
            continue

        # 每个 ckpt 建一个新模型，避免加载覆盖你训练中的对象（更干净）
        model, diffusion = build_model_and_diffusion(cfg, device)
        model.to(device).eval()

        # 加载：优先 EMA 权重（如果有）
        load_checkpoint(model, ckpt_path, device=device, optimizer=None, use_ema=use_ema, strict=strict)

        cond = inp_b.to(device)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        out_b = diffusion.ddim_sample(
            model,
            cond=cond,
            steps=50,
            eta=0.0,
            clip_x0=True,
        ).clamp(-1, 1).cpu()
        # 转 uint8 并保存拼图
        gt_list = [_to_uint8_chw(gt_b[i]) for i in range(n_rows)]
        inp_list = [_to_uint8_chw(inp_b[i]) for i in range(n_rows)]
        out_list = [_to_uint8_chw(out_b[i]) for i in range(n_rows)]

        save_path = out_dir / f"grid_best_{tag}_t{t_start}.png"
        _save_grid_10x3(gt_list, inp_list, out_list, save_path, pad=2)
        print(f"[Preview] saved: {save_path}")
