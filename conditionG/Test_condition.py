# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
# [MOD] Standalone inference script for underwater RGB conditional DDPM.
# [MOD] Removed torchvision dependency.

# Usage:
#   python Test_condition.py --data_root /dataset --ckpt runs/cond_ddpm_uie/checkpoints/best_psnr.pt --out_dir ./out --num_images 10

import sys  # [MOD] make local imports work when running from elsewhere
from pathlib import Path as _Path  # [MOD]
_THIS_DIR = _Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import argparse
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import UnderwaterPairDataset
from Model_condition import UNet
from Diffusion_condition import GaussianDiffusionSampler_cond


def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5


def tensor_to_pil(x_01: torch.Tensor) -> Image.Image:
    x = (x_01.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def save_triplet(cond, pred, gt, path: Path):
    c = tensor_to_pil(denorm_to_01(cond))
    p = tensor_to_pil(denorm_to_01(pred))
    g = tensor_to_pil(denorm_to_01(gt))

    W, H = c.size
    canvas = Image.new("RGB", (W * 3, H))
    canvas.paste(c, (0, 0))
    canvas.paste(p, (W, 0))
    canvas.paste(g, (W * 2, 0))
    canvas.save(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/public/home/hnust15874739861/pro/publicdata/LSUI19")
    parser.add_argument("--ckpt", type=str, default="./expt_record")
    parser.add_argument("--out_dir", type=str, default="./test_out")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_images", type=int, default=10)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.data_root)
    val_ds = UnderwaterPairDataset(
        input_dir=str(root / "Val" / "input"),
        gt_dir=str(root / "Val" / "GT"),
        image_size=args.image_size,
        is_train=False,
    )
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = UNet(
        T=args.timesteps,
        ch=args.ch,
        ch_mult=(1, 2, 2, 2),
        attn=(1,),
        num_res_blocks=2,
        dropout=args.dropout,
        in_channels=6,
        out_channels=3,
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    sampler = GaussianDiffusionSampler_cond(model, args.beta_1, args.beta_T, args.timesteps).to(device)

    saved = 0
    for batch in loader:
        cond = batch["cond"].to(device)
        gt = batch["gt"].to(device)
        names = batch["name"]

        pred = sampler(cond)

        for i in range(cond.shape[0]):
            if saved >= args.num_images:
                print(f"[DONE] saved {saved} images to {out_dir}")
                return
            save_triplet(cond[i], pred[i], gt[i], out_dir / f"{names[i]}_input_pred_gt.png")
            saved += 1

    print(f"[DONE] saved {saved} images to {out_dir}")


if __name__ == "__main__":
    main()