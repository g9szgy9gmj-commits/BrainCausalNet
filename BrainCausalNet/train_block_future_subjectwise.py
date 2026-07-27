import argparse
import copy
import csv
import os
from pathlib import Path

import numpy as np
import torch

import data_loader.data_loaders_block_future_subjectwise as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device, read_json


SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def parse_subjects(raw_subjects, start_subject, end_subject, n_subjects):
    if raw_subjects:
        subjects = []
        for part in raw_subjects.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                start, end = [int(x) for x in part.split('-', 1)]
                subjects.extend(range(start, end + 1))
            else:
                subjects.append(int(part))
    else:
        start = 0 if start_subject is None else int(start_subject)
        end = n_subjects - 1 if end_subject is None else int(end_subject)
        subjects = list(range(start, end + 1))

    bad = [s for s in subjects if s < 0 or s >= n_subjects]
    if bad:
        raise ValueError(f"Subject indices out of range [0, {n_subjects - 1}]: {bad}")
    return subjects


def append_summary(summary_path, row):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_path.exists()
    with summary_path.open('a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'subject_idx', 'status', 'save_dir', 'log_dir',
                'n_windows', 'error'
            ])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_subject(base_cfg, subject_idx, run_id_prefix, args):
    cfg = copy.deepcopy(base_cfg)
    cfg['name'] = args.name
    cfg['data_loader']['args']['subject_idx'] = int(subject_idx)
    if args.epochs is not None:
        cfg['trainer']['epochs'] = int(args.epochs)
    if args.lr is not None:
        cfg['optimizer']['args']['lr'] = float(args.lr)
    if args.batch_size is not None:
        cfg['data_loader']['args']['batch_size'] = int(args.batch_size)

    run_id = f"{run_id_prefix}_subject_{subject_idx:03d}"
    config = ConfigParser(cfg, resume=None, modification=None, run_id=run_id)
    logger = config.get_logger('train')

    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    config['data_loader']['args']['series_num'] = data_loader.series_num
    config['data_loader']['args']['time_step'] = data_loader.time_step
    config['data_loader']['args']['output_window'] = data_loader.output_window

    model = config.init_obj('arch', module_arch, config)
    logger.info(model)

    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    lam = config['trainer']['lam']

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      lam=lam)

    trainer.train()
    torch.cuda.empty_cache()

    return {
        'subject_idx': int(subject_idx),
        'status': 'ok',
        'save_dir': str(config.save_dir),
        'log_dir': str(config.log_dir),
        'n_windows': len(data_loader.dataset),
        'error': '',
    }


def main():
    parser = argparse.ArgumentParser(description='Subject-wise block future prediction')
    parser.add_argument('-c', '--config', required=True, type=str)
    parser.add_argument('-d', '--device', default=None, type=str)
    parser.add_argument('--subjects', default='', type=str,
                        help='Comma list/ranges, e.g. 0,2,5-8. Defaults to all subjects.')
    parser.add_argument('--start-subject', default=None, type=int)
    parser.add_argument('--end-subject', default=None, type=int)
    parser.add_argument('--run-id-prefix', default='subjectwise_block32_pred32', type=str)
    parser.add_argument('--summary', default='saved/subjectwise_block32_pred32_summary.csv', type=str)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--batch-size', default=None, type=int)
    parser.add_argument('--name', default='BD_HC Subjectwise Block32 Predict32 Causality Learning', type=str)
    args = parser.parse_args()

    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    base_cfg = read_json(args.config)
    raw = np.load(base_cfg['data_loader']['args']['data_dir'], mmap_mode='r')
    subjects = parse_subjects(args.subjects, args.start_subject, args.end_subject, raw.shape[0])

    summary_path = Path(args.summary)
    print(f"Subject-wise block future training: {len(subjects)} subjects")
    print(f"Data shape: {tuple(raw.shape)}")
    print(f"Summary: {summary_path}")

    for subject_idx in subjects:
        run_id = f"{args.run_id_prefix}_subject_{subject_idx:03d}"
        save_dir = Path(base_cfg['trainer']['save_dir']) / 'models' / args.name / run_id
        if args.skip_existing and (save_dir / 'model_best.pth').exists():
            row = {
                'subject_idx': int(subject_idx),
                'status': 'skipped',
                'save_dir': str(save_dir),
                'log_dir': '',
                'n_windows': '',
                'error': 'model_best.pth exists',
            }
            append_summary(summary_path, row)
            print(f"[subject {subject_idx:03d}] skipped")
            continue

        try:
            print(f"[subject {subject_idx:03d}] training")
            row = train_one_subject(base_cfg, subject_idx, args.run_id_prefix, args)
            append_summary(summary_path, row)
            print(f"[subject {subject_idx:03d}] done")
        except Exception as exc:
            torch.cuda.empty_cache()
            row = {
                'subject_idx': int(subject_idx),
                'status': 'error',
                'save_dir': str(save_dir),
                'log_dir': '',
                'n_windows': '',
                'error': repr(exc),
            }
            append_summary(summary_path, row)
            print(f"[subject {subject_idx:03d}] error: {exc}")


if __name__ == '__main__':
    main()
