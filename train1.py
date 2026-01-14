import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import json
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default=r'/public/home/hnust15874739861/pro/DiffusionProject/config/config.yaml',
                        help='yml file for configuration')
    parser.add_argument('-p', '--phase', type=str, help='Run train(training)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # 解析配置文件
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model 创建Model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    logger.info("current_step:%s\ncurrent_epoch:%s", current_step, current_epoch)
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        # ---- best tracking & logging ----
        best_dir = opt['path'].get('best', os.path.join(opt['path']['experiments_root'], 'best'))
        os.makedirs(best_dir, exist_ok=True)

        def _dump_json(path, obj):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

        def save_best_metrics(tag, info_dict):
            """
            Save metrics for the current best_{tag}.
            tag: 'loss' | 'psnr' | 'ssim'
            """
            # per-best file
            _dump_json(os.path.join(best_dir, f'best_{tag}_metrics.json'), info_dict)

            # summary file (always updated)
            summary_path = os.path.join(best_dir, 'best_summary.json')
            if os.path.exists(summary_path):
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
            else:
                summary = {}

            summary[f'best_{tag}'] = info_dict
            _dump_json(summary_path, summary)


        # Track best values + where they happen
        best_records = {
            'loss': {'value': float('inf'), 'epoch': -1, 'iter': -1},
            'psnr': {'value': float('-inf'), 'epoch': -1, 'iter': -1},
            'ssim': {'value': float('-inf'), 'epoch': -1, 'iter': -1},
        }
        # Track train loss between validations (so we can store train_loss with each best)
        train_loss_sum = 0.0
        train_loss_count = 0

        os.makedirs(opt['path'].get('best', os.path.join(opt['path']['experiments_root'], 'best')), exist_ok=True)
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                if wandb_logger:
                    wandb_logger.log_metrics(len(train_data))
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()

                # accumulate train loss for the upcoming validation
                cur_logs = diffusion.get_current_log() if hasattr(diffusion, 'get_current_log') else diffusion.log_dict
                if cur_logs is not None and 'l_pix' in cur_logs:
                    train_loss_sum += float(cur_logs['l_pix'])
                    train_loss_count += 1
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_train_loss = (train_loss_sum / max(1, train_loss_count))
                    train_loss_sum = 0.0
                    train_loss_count = 0
                    avg_loss = 0.0
                    avg_psnr = 0.0
                    avg_ssim = 0.0

                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')

                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.netG.eval()
                        with torch.no_grad():
                            l_pix = diffusion.netG(diffusion.data)
                            b, c, h, w = diffusion.data['target'].shape
                            l_pix = l_pix.sum()/int(b*c*h*w)
                        avg_loss += l_pix.item()
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        restore_img = Metrics.tensor2img(visuals['output'])  # uint8
                        target_img = Metrics.tensor2img(visuals['target'])  # uint8
                        input_img = Metrics.tensor2img(visuals['input'])  # uint8

                        # generation
                        Metrics.save_img(target_img, '{}/{}_{}_target.png'.format(result_path, current_step, idx))
                        Metrics.save_img(restore_img, '{}/{}_{}_output.png'.format(result_path, current_step, idx))
                        Metrics.save_img(input_img, '{}/{}_{}_input.png'.format(result_path, current_step, idx))
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate((input_img, restore_img, target_img), axis=1), [2, 0, 1]), idx)
                        avg_psnr += Metrics.calculate_psnr(restore_img, target_img)
                        avg_ssim += Metrics.calculate_ssim(restore_img, target_img)



                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((input_img, restore_img, target_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    avg_loss = avg_loss / idx

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # Loss:{:.4e} PSNR: {:.4e} SSIM:{:.4e}'.format(avg_loss, avg_psnr, avg_ssim))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> loss: {:.4e} ssim: {:.4e} psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_loss, avg_ssim, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('train_loss: ', avg_train_loss, current_step)
                    tb_logger.add_scalar('loss: ', avg_loss, current_step)
                    tb_logger.add_scalar('psnr: ', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim: ', avg_ssim, current_step)

                    info={
                        'epoch': current_epoch,
                        'iter': current_step,
                        'train_loss': avg_train_loss,
                        'val_loss': avg_loss,
                        'psnr': avg_psnr,
                        'ssim': avg_ssim,
                    }
                    if avg_loss < best_records['loss']['value']:
                        best_records['loss']={'value': float(avg_loss), 'epoch': current_epoch, 'iter': current_step,}
                        diffusion.save_best_network('loss', current_epoch, current_step)
                        save_best_metrics('loss', info)
                    if avg_psnr > best_records['psnr']['value']:
                        best_records['psnr'] = {'value': float(avg_psnr), 'epoch': (current_epoch), 'iter': current_step,}
                        diffusion.save_best_network('psnr', current_epoch, current_step)
                        save_best_metrics('psnr', info)
                    if avg_ssim > best_records['ssim']['value']:
                        best_records['ssim']={'value':float(avg_ssim), 'epoch': current_epoch, 'iter':current_step}
                        diffusion.save_best_network('ssim', current_epoch, current_step)
                        save_best_metrics('ssim', info)
                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation_loss': avg_loss,
                            'validation/val_psnr': avg_psnr,
                            'validation/val_loss': avg_loss,
                            'validation/val_step': val_step,
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch - 1})
        logger.info('======Training Finished: Best Summary======')
        logger.info('Best loss: {:.6e} @ epoch {} iter {}'.format(
            best_records['loss']['value'], best_records['loss']['epoch'], best_records['loss']['iter']))
        logger.info('Best psnr: {:.4f} @ epoch {} iter {}'.format(
            best_records['psnr']['value'], best_records['psnr']['epoch'], best_records['psnr']['iter']))
        logger.info('Best ssim: {:.4f} @ epoch {} iter {}'.format(
            best_records['ssim']['value'],best_records['ssim']['epoch'], best_records['ssim']['iter'] ))
        _dump_json(os.path.join(best_dir, 'best_records.json'), best_records)
        # save model
        logger.info('End of training.')
    else:
        raise NotImplementedError('phase should be the train phase')
