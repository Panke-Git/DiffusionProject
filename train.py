import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.validation import ValidationScheduler, run_validation, should_save_checkpoint
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
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
    logger.info("current_step:{}".format(current_step), "\n", "current_epoch:{}".format(current_epoch))
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        validation_scheduler = ValidationScheduler(opt, n_iter)
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
                    for job in validation_jobs:
                        info = run_validation(
                            diffusion, val_loader, opt, current_epoch, current_step,
                            job['tag'], job['cfg'], tb_logger=tb_logger,
                            wandb_logger=wandb_logger
                        )
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
            run_validation(
                diffusion, val_loader, opt, current_epoch, current_step,
                final_job['tag'], final_job['cfg'], tb_logger=tb_logger,
                wandb_logger=wandb_logger
            )
            if should_save_checkpoint(opt, final_job['tag']):
                logger.info('Saving checkpoint after %s validation.', final_job['tag'])
                diffusion.save_network(current_epoch, current_step)
        # save model
        logger.info('End of training.')
    else:
        raise NotImplementedError('phase should be the train phase')
