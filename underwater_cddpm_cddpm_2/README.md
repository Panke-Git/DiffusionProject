# Conditional DDPM for Underwater Image Enhancement (PyTorch)

This project trains a **conditional diffusion model (DDPM)** for underwater image enhancement.

- **Condition**: the underwater input image
- **Diffusion target**: the enhanced GT image
- **Model**: full U-Net DDPM backbone with multi-head self-attention blocks (not a toy/simplified model)
- **Training config**: YAML
- **Checkpoints saved**: only **best SSIM**, **best PSNR**, **best Loss**
- After training, the script can generate enhanced images using each of the 3 best checkpoints.

## Dataset structure (as you described)

```
/dataset/Train/input
/dataset/Train/GT
/dataset/Val/input
/dataset/Val/GT
```

Filenames between `input` and `GT` should match (e.g. `0001.png` in both folders).

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --config configs/train.yaml
```

Outputs (logs, checkpoints, samples) go under `paths.out_dir` in the YAML.

## Inference (use a saved .pt checkpoint)

Single image:
```bash
python infer.py \
  --config configs/train.yaml \
  --ckpt outputs/underwater_cddpm/checkpoints/best_ssim.pt \
  --input /path/to/underwater.png \
  --output outputs/infer_out
```

Folder:
```bash
python infer.py \
  --config configs/train.yaml \
  --ckpt outputs/underwater_cddpm/checkpoints/best_psnr.pt \
  --input /path/to/folder_of_underwater_images \
  --output outputs/infer_out
```

## Notes

- The model is trained on a fixed `data.image_size` (default 256).  
  For inference, inputs are resized to this size (configurable).
- Validation PSNR/SSIM uses diffusion sampling (DDIM by default for speed).  
  You can switch to full DDPM sampling in the YAML.
