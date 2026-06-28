import argparse
import json
import logging
import os
import random
import re

import core.logger as Logger


# Edit these defaults, then run:
#   python validate_checkpoint.py
DEFAULT_CONFIG = "config/config3.yaml"
DEFAULT_VERSION = "v3"

# Set this to a checkpoint prefix or a *_gen.pth file.
# Example:
#   "experiments/UNet_UAS4_ADMM_V3_xxxxxx/checkpoint/I860000_E607_gen.pth"
# If left empty, the script auto-picks the newest experiments/**/checkpoint/*_gen.pth.
DEFAULT_CHECKPOINT = ""

# -1 means full validation. Positive N means validate N samples.
DEFAULT_NUM_SAMPLES = -1

# False: first N samples. True: random N samples. Only used when DEFAULT_NUM_SAMPLES > 0.
DEFAULT_RANDOM_SAMPLE = False
DEFAULT_SEED = 42

# None means use config val schedule. Common quick check: 100. Full check: 2000.
DEFAULT_N_TIMESTEP = None
DEFAULT_TAG = None
DEFAULT_MAX_SAVE_IMAGES = 16
DEFAULT_SAVE_IMAGES = True

# "auto": use CUDA when available, otherwise CPU.
# "cpu": force CPU.
# "0" / "0,1": force specific CUDA devices.
DEFAULT_DEVICE = "auto"
DEFAULT_GPU_IDS = None


def _find_latest_checkpoint():
    candidates = []
    root = "experiments"
    if not os.path.isdir(root):
        return None
    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(dirpath) != "checkpoint":
            continue
        for filename in filenames:
            if filename.endswith("_gen.pth"):
                path = os.path.join(dirpath, filename)
                candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _configure_device(opt, args, torch_module):
    device_setting = str(DEFAULT_DEVICE).lower()
    force_cpu = device_setting == "cpu"
    auto_cpu = device_setting == "auto" and not torch_module.cuda.is_available()

    if force_cpu or auto_cpu:
        opt["gpu_ids"] = None
        opt["distributed"] = False
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if auto_cpu:
            print("CUDA is not available. Using CPU for validation.")
        else:
            print("DEFAULT_DEVICE='cpu'. Using CPU for validation.")
        return opt

    if device_setting not in ["auto", "cuda", "gpu"] and args.gpu_ids is None:
        args.gpu_ids = DEFAULT_DEVICE
        opt["gpu_ids"] = [int(item) for item in str(DEFAULT_DEVICE).split(",")]
        opt["distributed"] = len(opt["gpu_ids"]) > 1
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEFAULT_DEVICE)
    return opt


def _checkpoint_to_resume_prefix(checkpoint):
    if checkpoint is None:
        return None, None
    checkpoint = os.path.expanduser(checkpoint)
    if checkpoint.endswith("_gen.pth"):
        return checkpoint[:-len("_gen.pth")], checkpoint
    if checkpoint.endswith(".pth"):
        return None, checkpoint
    return checkpoint, f"{checkpoint}_gen.pth"


def _infer_epoch_step(path):
    if not path:
        return 0, 0
    name = os.path.basename(path)
    match = re.search(r"I(\d+)_E(\d+)", name)
    if match:
        return int(match.group(2)), int(match.group(1))
    return 0, 0


def _make_model(opt, version):
    import model as Model

    version = version.lower()
    if version in ["base", "v0"]:
        return Model.create_model(opt)
    if version == "v1":
        return Model.create_modelV1(opt)
    if version == "v3":
        return Model.create_modelV3(opt)
    if version == "v4":
        return Model.create_modelV4(opt)
    if version == "v5":
        return Model.create_modelV5(opt)
    if version == "v6":
        return Model.create_modelV6(opt)
    if version == "v7":
        return Model.create_modelV7(opt)
    raise ValueError(f"Unsupported version: {version}")


def _load_direct_pth(diffusion, pth_path, strict=True):
    import torch

    network = diffusion.netG
    if isinstance(network, torch.nn.DataParallel):
        network = network.module
    state = torch.load(pth_path, map_location=diffusion.device)
    network.load_state_dict(state, strict=strict)


def _build_val_loader(opt, num_samples, random_sample, seed):
    from torch.utils.data import DataLoader, Subset

    import data as Data

    val_opt = opt["datasets"]["val"]
    val_set = Data.create_datasetV1(val_opt, "val")

    selected_indices = None
    if random_sample:
        if num_samples <= 0:
            raise ValueError("--random_sample requires --num_samples > 0")
        n = min(int(num_samples), len(val_set))
        selected_indices = random.Random(seed).sample(range(len(val_set)), n)
        val_set = Subset(val_set, selected_indices)

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=int(val_opt.get("num_workers", 1) or 1),
        pin_memory=bool(opt.get("gpu_ids") is not None),
    )
    return val_loader, selected_indices


def main():
    parser = argparse.ArgumentParser(
        description="Validate a saved DiffWater checkpoint on full val set or a custom subset."
    )
    parser.add_argument("-c", "--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION,
                        choices=["base", "v0", "v1", "v3", "v4", "v5", "v6", "v7"])
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Checkpoint prefix or *_gen.pth file.")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help="-1 means full validation; positive N means validate N samples.")
    parser.add_argument("--random_sample", action="store_true", default=DEFAULT_RANDOM_SAMPLE,
                        help="Randomly choose num_samples images instead of taking the first N.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_timestep", type=int, default=DEFAULT_N_TIMESTEP,
                        help="Override validation diffusion steps. Default uses config val schedule.")
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument("--max_save_images", type=int, default=DEFAULT_MAX_SAVE_IMAGES)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--no_save_images", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Use strict=True when directly loading a .pth that is not named *_gen.pth.")

    # Arguments consumed by Logger.parse.
    parser.add_argument("-p", "--phase", type=str, default="val")
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=DEFAULT_GPU_IDS)
    parser.add_argument("-debug", "-d", action="store_true")
    parser.add_argument("-enable_wandb", action="store_true")
    parser.add_argument("-log_wandb_ckpt", action="store_true")
    parser.add_argument("-log_eval", action="store_true")

    args = parser.parse_args()
    args.phase = "val"
    args.enable_wandb = False

    if not args.checkpoint:
        args.checkpoint = _find_latest_checkpoint()
        if args.checkpoint:
            print(f"Auto-selected latest checkpoint: {args.checkpoint}")
    if not args.checkpoint:
        raise FileNotFoundError(
            "No checkpoint was provided and no experiments/**/checkpoint/*_gen.pth file was found. "
            "Set DEFAULT_CHECKPOINT near the top of validate_checkpoint.py."
        )

    resume_prefix, direct_pth = _checkpoint_to_resume_prefix(args.checkpoint)
    if direct_pth is None or not os.path.isfile(direct_pth):
        raise FileNotFoundError(f"Cannot find checkpoint file: {direct_pth}")

    import torch
    from core.validation import run_validation

    opt = Logger.parse(args)
    opt = _configure_device(opt, args, torch)
    opt["path"]["resume_state"] = resume_prefix
    opt["validation"]["seed"] = args.seed
    opt = Logger.dict_to_nonedict(opt)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    Logger.setup_logger(None, opt["path"]["log"], "validate_checkpoint", level=logging.INFO, screen=True)
    Logger.setup_logger("val", opt["path"]["log"], "val", level=logging.INFO)
    logger = logging.getLogger("base")
    logger.info(Logger.dict2str(opt))

    val_loader, selected_indices = _build_val_loader(
        opt,
        num_samples=args.num_samples,
        random_sample=args.random_sample,
        seed=args.seed,
    )
    logger.info("Initial validation dataset finished. selected_indices=%s", selected_indices)

    diffusion = _make_model(opt, args.version)
    if resume_prefix is None:
        logger.info("Directly loading checkpoint [%s] ...", direct_pth)
        _load_direct_pth(diffusion, direct_pth, strict=args.strict)
    logger.info("Initial model finished.")

    epoch, step = _infer_epoch_step(direct_pth)
    tag = args.tag
    if tag is None:
        if args.random_sample:
            tag = f"random{args.num_samples}"
        elif args.num_samples > 0:
            tag = f"first{args.num_samples}"
        else:
            tag = "full"

    cfg = {
        "enabled": True,
        "num_samples": -1 if args.random_sample else int(args.num_samples),
        "save_images": bool((DEFAULT_SAVE_IMAGES or args.save_images) and not args.no_save_images),
        "max_save_images": int(args.max_save_images),
        "metrics": opt["validation"].get("report_metrics", ["psnr", "ssim", "uiqm", "uciqe"]),
        "seed": int(args.seed),
    }
    if args.n_timestep is not None:
        cfg["n_timestep"] = int(args.n_timestep)

    info = run_validation(
        diffusion=diffusion,
        val_loader=val_loader,
        opt=opt,
        current_epoch=epoch,
        current_step=step,
        tag=tag,
        cfg=cfg,
        tb_logger=None,
        wandb_logger=None,
    )

    result_dir = os.path.join(opt["path"]["results"], tag, f"E{epoch}_I{step}")
    info["checkpoint"] = direct_pth
    info["version"] = args.version
    info["config"] = args.config
    info["random_sample"] = bool(args.random_sample)
    info["selected_indices"] = selected_indices
    with open(os.path.join(result_dir, "validation_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    logger.info("Validation info saved to %s", os.path.join(result_dir, "validation_info.json"))
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
