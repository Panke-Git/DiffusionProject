"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: train.py
    @Time: 2025/12/4 22:31
    @Email: None
"""
"""训练入口：从 YAML 读取配置，构建数据集、模型与扩散过程并启动训练。

使用方式：
    python train.py --config configs/default.yaml

训练过程中会：
- 在日志目录下保存配置备份、采样图、模型检查点
- 支持混合精度与梯度裁剪
- 每个 epoch 结束后可生成若干采样结果，方便观察收敛
"""
import argparse
import json
from typing import Any, Dict, List

import torch
import torch.amp as amp
import torch.nn.functional as F
import torch.optim as optim
import yaml

from . import model as modellib
from .model.diffusion import GaussianDiffusion
from .utils.train_utils import prepare_dataloader, set_seed, create_output_dirs, load_config, save_comparison_grid, \
    plot_loss_curve, tensor_to_01, save_samples, ssim, psnr, evaluate_sampling
from pathlib import Path


IMG_SIZE = 256

def build_model(cfg: Dict[str, Any]) -> modellib.Unet:
    # 条件扩散：输入为 [x_t, cond]，通道数翻倍
    in_channels = cfg["data"]["channels"] * 2
    out_channels = cfg["data"]["channels"]
    return modellib.Unet(
        in_channels=in_channels,
        base_channels=64,
        channel_mults=[1, 2, 4, 8],
        num_res_blocks=2,
        dropout=0.1,
        out_channels=out_channels,
    )


def train(cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["experiment"].get("seed", 42))

    # 日志/输出目录：根目录/模型名/时间戳
    dirs = create_output_dirs(cfg)
    output_dir = dirs["output_dir"]
    samples_dir = dirs["samples_dir"]
    ckpt_dir = dirs["ckpt_dir"]
    metrics_path = dirs["metrics_path"]

    num_epochs = cfg["train"]["num_epochs"]

    # 保存配置备份，方便溯源
    with open(output_dir / "config_saved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    train_loader = prepare_dataloader(cfg, split="train", shuffle=True)
    val_loader = None
    if cfg["data"].get("val"):
        val_loader = prepare_dataloader(cfg, split="val", shuffle=False)
    model = build_model(cfg).to(device)
    diffusion = GaussianDiffusion(
        model=model,
        image_size=cfg["data"]["image_size"],
        channels=cfg["data"]["channels"],
        timesteps=cfg["diffusion"]["num_steps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        recon_weight=cfg["train"].get("recon_weight", 0.0),
    ).to(device)

    print("Training...")
    print("output_dir: {}".format(output_dir))
    print("num_epochs: {}".format(num_epochs))
    print("database: {}".format(cfg["data"]["train"]["input_dir"].split("/")[-3]))

    optimizer = optim.AdamW(
        diffusion.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    use_amp = cfg["train"].get("mixed_precision", True) and (device.type == "cuda")

    scaler = amp.GradScaler(
        "cuda",  # 注意：这里是位置参数，不是 device_type=...
        enabled=use_amp,
    )

    ema_decay = cfg["train"].get("ema_decay", 1.0)
    ema_model = None
    if ema_decay < 1.0:
        ema_model = build_model(cfg).to(device)
        ema_model.load_state_dict(model.state_dict())


    log_interval = cfg["train"].get("log_interval", 50)

    global_step = 0
    metrics_history: List[Dict[str, float]] = []
    best_records = {
        "loss": {"value": float("inf"), "path": None},
        "ssim": {"value": float("-inf"), "path": None},
        "psnr": {"value": float("-inf"), "path": None},
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for batch in train_loader:
            optimizer.zero_grad()

            gt = batch["image"].to(device)  # GT 图像
            cond = batch.get("input")
            if cond is None:
                raise RuntimeError(
                    "当前为条件扩散训练，但训练集 batch 中没有 'input'。"
                    "请在 config.data.train.input_dir 中配置原始水下图像目录。"
                )
            cond = cond.to(device)

            noise = torch.randn_like(gt)
            t = torch.randint(
                0, cfg["diffusion"]["num_steps"], (gt.shape[0],), device=device
            ).long()

            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = diffusion.p_losses(gt, t, noise, cond=cond)

            scaler.scale(loss).backward()
            if cfg["train"].get("grad_clip", 0.0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            train_loss_sum += loss.item() * gt.size(0)
            train_samples += gt.size(0)

            global_step += 1
            if global_step % log_interval == 0:
                print(f"Epoch {epoch} Step {global_step}: loss={loss.item():.6f}")

        # 验证集评估：使用 EMA 模型（若存在），计算噪声预测 MSE 与图像质量指标
        val_loss = None
        val_ssim = None
        val_psnr = None
        eval_ssim = None
        eval_psnr = None
        model_to_eval = None
        if val_loader is not None:
            model_to_eval = ema_model if ema_model is not None else model
            original_model = diffusion.model
            diffusion.model = model_to_eval
            model_to_eval.eval()
            total_loss = 0.0
            total_ssim = 0.0
            total_psnr = 0.0
            total_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    gt = batch["image"].to(device)
                    cond = batch.get("input")
                    if cond is None:
                        raise RuntimeError(
                            "验证集也需要提供 'input' 条件图像，请在 config.data.val.input_dir 中配置。"
                        )
                    cond = cond.to(device)

                    noise = torch.randn_like(gt)
                    t = torch.randint(
                        0, cfg["diffusion"]["num_steps"], (gt.shape[0],), device=device
                    ).long()
                    x_noisy = diffusion.q_sample(gt, t, noise)

                    # 条件尺寸对齐
                    if cond.shape[2:] != x_noisy.shape[2:]:
                        cond_resized = F.interpolate(
                            cond,
                            size=x_noisy.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    else:
                        cond_resized = cond

                    model_in = torch.cat([x_noisy, cond_resized], dim=1)
                    predicted_noise = model_to_eval(model_in, t)
                    loss_val = F.mse_loss(predicted_noise, noise)
                    recon = diffusion.predict_start_from_noise(x_noisy, t, predicted_noise)

                    gt_01 = tensor_to_01(gt)
                    recon_01 = tensor_to_01(recon)
                    total_ssim += ssim(recon_01, gt_01).sum().item()
                    total_psnr += psnr(recon_01, gt_01).sum().item()
                    total_loss += loss_val.item() * gt.size(0)
                    total_samples += gt.size(0)

            val_loss = total_loss / max(total_samples, 1)
            val_ssim = total_ssim / max(total_samples, 1)
            val_psnr = total_psnr / max(total_samples, 1)
            diffusion.model = original_model
            print(f"Epoch {epoch}: validation loss={val_loss:.6f}")
            print(f"Epoch {epoch}: validation SSIM={val_ssim:.4f}, PSNR={val_psnr:.2f}dB")

            # 配置里可以加一个 evaluation.eval_interval 控制频率；没有就每个 epoch 都 eval
            eval_cfg = cfg.get("evaluation", {})
            eval_interval = int(eval_cfg.get("eval_interval", 1))
            max_eval_batches = eval_cfg.get("max_batches", None)

            # if (epoch % eval_interval) == 0:
            #     print("  -> 运行完整采样 eval_sampling（enhance(cond)）...")
            #     sampling_metrics = evaluate_sampling(
            #         model=model_to_eval,
            #         diffusion=diffusion,
            #         dataloader=val_loader,
            #         device=device,
            #         use_ddim=cfg["sampling"].get("ddim_steps", 0) > 0,
            #         num_steps=cfg["sampling"].get("ddim_steps", 0),
            #         max_batches=max_eval_batches,
            #     )
            #     eval_ssim = sampling_metrics["eval_ssim"]
            #     eval_psnr = sampling_metrics["eval_psnr"]
            #     print(
            #         f"  [EvalSampling] SSIM={eval_ssim:.4f}, "
            #         f"PSNR={eval_psnr:.2f}dB"
            #     )

        # 记录指标到 JSON
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_samples, 1),
        }
        if val_loss is not None:
            epoch_record.update(
                {
                    "val_loss": val_loss,
                    "val_ssim": val_ssim,
                    "val_psnr": val_psnr,
                }
            )
            # 如果本 epoch 做了完整采样 eval，则一起记录
            # if eval_ssim is not None and eval_psnr is not None:
            #     epoch_record.update(
            #         {
            #             "eval_ssim": eval_ssim,
            #             "eval_psnr": eval_psnr,
            #         }
            #     )

        metrics_history.append(epoch_record)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_history, f, ensure_ascii=False, indent=2)

        # 最优检查点：分别按最小 val_loss、最大 SSIM/PSNR 保存
        def maybe_save_best(metric: str, value: float) -> None:
            improved = False
            if metric == "loss" and value < best_records[metric]["value"]:
                improved = True
            elif metric != "loss" and value > best_records[metric]["value"]:
                improved = True

            if improved:
                best_records[metric]["value"] = value
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                if ema_model is not None:
                    state["ema_model"] = ema_model.state_dict()
                ckpt_path = ckpt_dir / f"best_{metric}.pt"
                torch.save(state, ckpt_path)
                best_records[metric]["path"] = str(ckpt_path)

                # 保存对比图（输入、GT、输出），覆盖之前的最佳文件
                if model_to_eval is not None:
                    comparison_path = ckpt_dir / f"best_{metric}_comparison.png"
                    save_comparison_grid(
                        model=model_to_eval,
                        diffusion=diffusion,
                        dataloader=val_loader,
                        device=device,
                        save_path=comparison_path,
                        num_samples=5,
                    )

        if val_loss is not None:
            maybe_save_best("loss", val_loss)
            maybe_save_best("ssim", val_ssim)
            maybe_save_best("psnr", val_psnr)

        # 采样与保存（条件扩散：使用输入图像作为条件做增强）
        if epoch % cfg["train"].get("sample_interval", 1) == 0:
            diffusion.eval()
            sampler = ema_model if ema_model is not None else model
            diffusion.model = sampler  # 暂时替换，便于采样

            with torch.no_grad():
                # 从验证集（若存在）或训练集取一批输入图像
                source_loader = val_loader if val_loader is not None else train_loader
                batch = next(iter(source_loader))

                cond = batch.get("input")
                if cond is None:
                    raise RuntimeError(
                        "采样阶段需要 batch['input'] 作为条件，请检查数据集与配置。"
                    )
                cond = cond.to(device)

                use_ddim = cfg["sampling"].get("ddim_steps", 0) > 0
                if use_ddim:
                    samples = diffusion.enhance(
                        cond,
                        use_ddim=True,
                        num_steps=cfg["sampling"]["ddim_steps"],
                    )
                else:
                    samples = diffusion.enhance(cond, use_ddim=False)

            save_samples(samples, samples_dir, epoch, cfg["sampling"].get("save_grid", True))
            diffusion.model = model  # 采样后换回训练模型

    # =========================
    # 训练循环结束：对 best_loss / best_ssim / best_psnr 三个 ckpt 做完整采样 eval
    # =========================
    if val_loader is not None:
        print("开始对 best_loss / best_ssim / best_psnr 三个检查点做完整采样评估...")

        # 用于存储最终 eval 结果
        best_eval_results = {}

        # 方便：从 config/sampling 里读采样设置
        use_ddim = cfg["sampling"].get("ddim_steps", 0) > 0
        num_steps = cfg["sampling"].get("ddim_steps", 0)
        eval_cfg = cfg.get("evaluation", {})
        max_eval_batches = eval_cfg.get("max_batches", None)  # 可选：只在部分 batch 上评估

        for metric_name in ["loss", "ssim", "psnr"]:
            record = best_records.get(metric_name)
            ckpt_path = record.get("path") if record is not None else None
            best_value = record.get("value") if record is not None else None

            if not ckpt_path:
                print(f"  [Eval] 跳过 {metric_name}，因为还没有对应的 best_{metric_name}.pt")
                continue

            ckpt_path = Path(ckpt_path)
            if not ckpt_path.is_file():
                print(f"  [Eval] 找不到检查点文件: {ckpt_path}")
                continue

            print(f"  [Eval] 加载 {metric_name} 最优检查点: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)

            # 优先使用保存的 EMA 模型；如果没有，就用普通模型权重
            if "ema_model" in ckpt:
                model.load_state_dict(ckpt["ema_model"])
                eval_model_type = "ema_model"
            else:
                model.load_state_dict(ckpt["model"])
                eval_model_type = "model"

            # 使用完整采样链做评估
            sampling_metrics = evaluate_sampling(
                model=model,
                diffusion=diffusion,
                dataloader=val_loader,
                device=device,
                use_ddim=use_ddim,
                num_steps=num_steps,
                max_batches=max_eval_batches,
            )

            eval_ssim = sampling_metrics["eval_ssim"]
            eval_psnr = sampling_metrics["eval_psnr"]

            print(
                f"  [Eval-{metric_name}] ({eval_model_type}) "
                f"best_value={best_value:.4f}, "
                f"EvalSampling SSIM={eval_ssim:.4f}, PSNR={eval_psnr:.2f}dB"
            )

            best_eval_results[metric_name] = {
                "metric": metric_name,
                "best_value": float(best_value),
                "ckpt_path": str(ckpt_path),
                "epoch": int(ckpt.get("epoch", -1)),
                "model_type": eval_model_type,
                "eval_ssim": float(eval_ssim),
                "eval_psnr": float(eval_psnr),
            }

        # 把三组结果写到一个单独的 JSON 文件里
        eval_results_path = output_dir / "eval_best_checkpoints.json"
        with open(eval_results_path, "w", encoding="utf-8") as f:
            json.dump(best_eval_results, f, ensure_ascii=False, indent=2)

        print(f"完整采样评估结果已保存到: {eval_results_path}")
    # =========================


    print("训练完成！模型与采样结果已保存。")

    # 绘制损失曲线并保存到输出目录
    if metrics_history:
        plot_loss_curve(metrics_history, output_dir / "loss_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM for underwater images")
    parser.add_argument(
        "--config",
        type=str,
        default="/public/home/hnust15874739861/pro/DiffusionProject/condition/src/configs/config.yaml",
        help="路径指向 YAML 配置文件",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)
