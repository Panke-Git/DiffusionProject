import copy
import logging
import os
import random

import numpy as np
import torch

import core.metrics as Metrics


logger = logging.getLogger('base')


def _get(opt, key, default=None):
    value = opt.get(key) if isinstance(opt, dict) else None
    return default if value is None else value


def _validation_opt(opt):
    return _get(opt, 'validation', {})


def _mode_opt(opt, mode):
    return _get(_validation_opt(opt), mode, {})


def _enabled(cfg, default=True):
    return bool(_get(cfg, 'enabled', default))


def _result_dir(opt, tag, epoch, step):
    path = os.path.join(opt['path']['results'], tag, 'E{}_I{}'.format(epoch, step))
    os.makedirs(path, exist_ok=True)
    return path


def _make_schedule(base_schedule, cfg):
    schedule = copy.deepcopy(base_schedule)
    n_timestep = _get(cfg, 'n_timestep', None)
    if n_timestep is not None:
        schedule['n_timestep'] = int(n_timestep)
    return schedule


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FixedSeed:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        if self.seed is None:
            return
        self.random_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        self.cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        _set_seed(int(self.seed))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seed is None:
            return
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)
        if self.cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_state)


class ValidationScheduler:
    def __init__(self, opt, n_iter):
        self.opt = opt
        self.n_iter = int(n_iter)
        self.done = {
            'medium': set(),
            'full': set(),
        }

    def _due_by_ratios(self, mode, current_step):
        cfg = _mode_opt(self.opt, mode)
        if not _enabled(cfg, default=False):
            return []
        ratios = _get(cfg, 'progress_ratios', [])
        due = []
        for ratio in ratios:
            ratio = float(ratio)
            if ratio in self.done[mode]:
                continue
            if current_step >= max(1, int(round(self.n_iter * ratio))):
                self.done[mode].add(ratio)
                due.append({'tag': mode, 'cfg': cfg, 'ratio': ratio})
        return due

    def get_jobs(self, current_step):
        jobs = []

        full_jobs = self._due_by_ratios('full', current_step)
        if full_jobs:
            return full_jobs

        medium_jobs = self._due_by_ratios('medium', current_step)
        if medium_jobs:
            return medium_jobs

        fast_cfg = _mode_opt(self.opt, 'fast')
        fast_freq = int(_get(fast_cfg, 'freq', _get(self.opt['train'], 'val_freq', 1000)))
        if _enabled(fast_cfg, default=True) and fast_freq > 0 and current_step % fast_freq == 0:
            jobs.append({'tag': 'fast', 'cfg': fast_cfg, 'ratio': None})
        return jobs

    def final_job(self):
        cfg = _mode_opt(self.opt, 'final')
        if _enabled(cfg, default=True):
            return {'tag': 'final', 'cfg': cfg, 'ratio': 1.0}
        return None


def _net_core(diffusion):
    return diffusion.netG.module if isinstance(diffusion.netG, torch.nn.DataParallel) else diffusion.netG


def _loss_value(diffusion, raw_loss):
    net = _net_core(diffusion)
    if bool(getattr(net, 'loss_is_normalized', False)):
        return float(raw_loss.item())
    b, c, h, w = diffusion.data['target'].shape
    return float((raw_loss.sum() / int(b * c * h * w)).item())


def _metric_names(opt, cfg):
    names = _get(cfg, 'metrics', None)
    if names is None:
        names = _get(_validation_opt(opt), 'report_metrics', ['psnr', 'ssim'])
    return [str(name).lower() for name in names]


def _calculate_metrics(restore_img, target_img, metric_names):
    values = {}
    if 'psnr' in metric_names:
        values['psnr'] = Metrics.calculate_psnr(restore_img, target_img)
    if 'ssim' in metric_names:
        values['ssim'] = Metrics.calculate_ssim(restore_img, target_img)
    if 'uiqm' in metric_names:
        values['uiqm'] = Metrics.calculate_uiqm(restore_img)
    if 'uciqe' in metric_names:
        values['uciqe'] = Metrics.calculate_uciqe(restore_img)
    return values


def run_validation(diffusion, val_loader, opt, current_epoch, current_step, tag,
                   cfg, tb_logger=None, wandb_logger=None):
    max_samples = int(_get(cfg, 'num_samples', -1))
    save_images = bool(_get(cfg, 'save_images', tag in ['medium', 'full', 'final']))
    max_save_images = int(_get(cfg, 'max_save_images', 8))
    seed = _get(cfg, 'seed', _get(_validation_opt(opt), 'seed', None))
    metric_names = _metric_names(opt, cfg)
    result_path = _result_dir(opt, tag, current_epoch, current_step)

    schedule = _make_schedule(opt['model']['beta_schedule']['val'], cfg)
    diffusion.set_new_noise_schedule(schedule, schedule_phase='{}_val'.format(tag))

    totals = {'loss': 0.0}
    for name in metric_names:
        totals[name] = 0.0
    idx = 0

    logger.info('Begin %s validation: max_samples=%s, n_timestep=%s, seed=%s',
                tag, max_samples, schedule['n_timestep'], seed)

    with FixedSeed(seed):
        for _, val_data in enumerate(val_loader):
            if max_samples > 0 and idx >= max_samples:
                break
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.netG.eval()
            with torch.no_grad():
                raw_loss = diffusion.netG(diffusion.data)
            totals['loss'] += _loss_value(diffusion, raw_loss)

            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals()
            restore_img = Metrics.tensor2img(visuals['output'])
            target_img = Metrics.tensor2img(visuals['target'])
            input_img = Metrics.tensor2img(visuals['input'])

            metrics = _calculate_metrics(restore_img, target_img, metric_names)
            for name, value in metrics.items():
                totals[name] += value

            if save_images and idx <= max_save_images:
                prefix = '{}_{}_{}'.format(current_step, tag, idx)
                Metrics.save_img(target_img, os.path.join(result_path, '{}_target.png'.format(prefix)))
                Metrics.save_img(restore_img, os.path.join(result_path, '{}_output.png'.format(prefix)))
                Metrics.save_img(input_img, os.path.join(result_path, '{}_input.png'.format(prefix)))
                if tb_logger is not None:
                    tb_logger.add_image(
                        '{}/Iter_{}'.format(tag, current_step),
                        np.transpose(np.concatenate((input_img, restore_img, target_img), axis=1), [2, 0, 1]),
                        idx
                    )
                if wandb_logger is not None:
                    wandb_logger.log_image(
                        '{}_validation_{}'.format(tag, idx),
                        np.concatenate((input_img, restore_img, target_img), axis=1)
                    )

    if idx == 0:
        raise RuntimeError('Validation loader produced no samples.')

    info = {
        'tag': tag,
        'epoch': current_epoch,
        'iter': current_step,
        'num_samples': idx,
        'n_timestep': schedule['n_timestep'],
        'val_loss': float(totals['loss'] / idx),
    }
    for name in metric_names:
        info[name] = float(totals[name] / idx)

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')

    message = '# {} Validation # Loss:{:.4e}'.format(tag, info['val_loss'])
    for name in metric_names:
        message += ' {}:{:.4e}'.format(name.upper(), info[name])
    logger.info(message)

    logger_val = logging.getLogger('val')
    logger_val.info(str(info))

    if tb_logger is not None:
        tb_logger.add_scalar('{}/loss'.format(tag), info['val_loss'], current_step)
        for name in metric_names:
            tb_logger.add_scalar('{}/{}'.format(tag, name), info[name], current_step)

    return info


def should_track_best(opt, tag):
    track_modes = _get(_validation_opt(opt), 'track_best_modes', ['full', 'final'])
    return tag in track_modes


def should_save_checkpoint(opt, tag):
    checkpoint_modes = _get(_validation_opt(opt), 'checkpoint_modes', ['full', 'final'])
    return tag in checkpoint_modes


def update_best_records(diffusion, best_records, info, save_best_metrics):
    if info['val_loss'] < best_records['loss']['value']:
        best_records['loss'] = {
            'value': float(info['val_loss']),
            'epoch': info['epoch'],
            'iter': info['iter'],
            'tag': info['tag'],
        }
        diffusion.save_best_network('loss', info['epoch'], info['iter'])
        save_best_metrics('loss', info)

    if 'psnr' in info and info['psnr'] > best_records['psnr']['value']:
        best_records['psnr'] = {
            'value': float(info['psnr']),
            'epoch': info['epoch'],
            'iter': info['iter'],
            'tag': info['tag'],
        }
        diffusion.save_best_network('psnr', info['epoch'], info['iter'])
        save_best_metrics('psnr', info)

    if 'ssim' in info and info['ssim'] > best_records['ssim']['value']:
        best_records['ssim'] = {
            'value': float(info['ssim']),
            'epoch': info['epoch'],
            'iter': info['iter'],
            'tag': info['tag'],
        }
        diffusion.save_best_network('ssim', info['epoch'], info['iter'])
        save_best_metrics('ssim', info)

    return best_records
