import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.validation import (
    ValidationScheduler,
    load_best_records,
    load_validation_history,
    run_validation,
    save_best_records,
    should_save_checkpoint,
    should_track_best,
    update_best_records,
)
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import json
import os
import random
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default=r'/public/home/hnust15874739861/pro/DiffusionProject/config/config5.yaml',
                        help='yml file for configuration')
    parser.add_argument('-p', '--phase', type=str, help='Run train(training)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # 解析配置文件
    args = parser.parse_args()

    SEED = 42
    # def seed_everything(seed=42):
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed_all(seed)
    #
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     torch.use_deterministic_algorithms(True)
    #
    #
    # seed_everything(SEED)

    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

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
            train_set = Data.create_datasetV1(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase, seed=SEED)
        elif phase == 'val':
            val_set = Data.create_datasetV1(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase, seed=SEED)
    logger.info('Initial Dataset Finished')

    # model 创建Model
    diffusion = Model.create_modelV5(opt)
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


        val_log_path = os.path.join(opt['path']['log'], 'val.log')
        validation_history = load_validation_history(val_log_path)
        best_records = load_best_records(opt, best_dir, validation_history)
        save_best_records(best_dir, best_records)
        logger.info('Loaded %d previous validation records from %s.', len(validation_history), val_log_path)
        logger.info('Resume best records: %s', best_records)

        # Track train loss between validations (so we can store train_loss with each best)
        train_loss_sum = 0.0
        train_loss_count = 0
        validation_scheduler = ValidationScheduler(
            opt, n_iter, start_step=current_step, validation_history=validation_history)

        os.makedirs(opt['path'].get('best', os.path.join(opt['path']['experiments_root'], 'best')), exist_ok=True)
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                checkpoint_saved_this_step = False
                if current_step > n_iter:
                    break
                if wandb_logger:
                    wandb_logger.log_metrics({'train/batch_size': train_data['target'].shape[0]}, commit=False)
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
                validation_jobs = validation_scheduler.get_jobs(current_step)
                if validation_jobs:
                    avg_train_loss = (train_loss_sum / max(1, train_loss_count))
                    train_loss_sum = 0.0
                    train_loss_count = 0
                    tb_logger.add_scalar('train_loss', avg_train_loss, current_step)
                    for job in validation_jobs:
                        info = run_validation(
                            diffusion, val_loader, opt, current_epoch, current_step,
                            job['tag'], job['cfg'], tb_logger=tb_logger,
                            wandb_logger=wandb_logger
                        )
                        info['train_loss'] = avg_train_loss
                        if should_track_best(opt, job['tag']):
                            best_records = update_best_records(diffusion, best_records, info, save_best_metrics)
                            save_best_records(best_dir, best_records)
                        if should_save_checkpoint(opt, job['tag']):
                            logger.info('Saving checkpoint after %s validation.', job['tag'])
                            diffusion.save_network(current_epoch, current_step)
                            checkpoint_saved_this_step = True
                        if wandb_logger:
                            wandb_metrics = {
                                'validation/{}/val_loss'.format(job['tag']): info['val_loss'],
                                'validation/val_step': val_step,
                            }
                            for metric_name in ['psnr', 'ssim', 'uiqm', 'uciqe']:
                                if metric_name in info:
                                    wandb_metrics['validation/{}/{}'.format(job['tag'], metric_name)] = info[metric_name]
                            wandb_logger.log_metrics(wandb_metrics)
                            val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0 and not checkpoint_saved_this_step:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch - 1})
        final_job = validation_scheduler.final_job()
        if final_job is not None:
            info = run_validation(
                diffusion, val_loader, opt, current_epoch, current_step,
                final_job['tag'], final_job['cfg'], tb_logger=tb_logger,
                wandb_logger=wandb_logger
            )
            info['train_loss'] = None
            if should_track_best(opt, final_job['tag']):
                best_records = update_best_records(diffusion, best_records, info, save_best_metrics)
                save_best_records(best_dir, best_records)
            if should_save_checkpoint(opt, final_job['tag']):
                logger.info('Saving checkpoint after %s validation.', final_job['tag'])
                diffusion.save_network(current_epoch, current_step)
        logger.info('======Training Finished: Best Summary======')
        logger.info('Best loss: {:.6e} @ epoch {} iter {}'.format(
            best_records['loss']['value'], best_records['loss']['epoch'], best_records['loss']['iter']))
        logger.info('Best psnr: {:.4f} @ epoch {} iter {}'.format(
            best_records['psnr']['value'], best_records['psnr']['epoch'], best_records['psnr']['iter']))
        logger.info('Best ssim: {:.4f} @ epoch {} iter {}'.format(
            best_records['ssim']['value'],best_records['ssim']['epoch'], best_records['ssim']['iter'] ))
        save_best_records(best_dir, best_records)
        # save model
        logger.info('End of training.')
    else:
        raise NotImplementedError('phase should be the train phase')
