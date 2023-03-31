import os
import shutil

import torch
import random
import copy
import argparse
import logging
import time
import datetime
import numpy as np
from collections import defaultdict
import warnings

warnings.filterwarnings(action='ignore')

from tensorboardX import SummaryWriter
from models.loss import Criterion
from pathlib import Path
from engine import train_one_epoch_null_nohead, server_evaluate
from util.misc import get_rank

from data import build_different_dataloader, build_server_dataloader
from config import build_config
# from models.model_config import get_cfg
from util.adam_svd import AdamSVD

from models.vit_models import Swin


def average_model(server_model, client_models, sampled_client_indices, coefficients):
    """Average the updated and transmitted parameters from each selected client."""
    averaged_weights = {}

    for k, v in client_models[0].state_dict().items():
        if 'prompter' in k or 'running' in k:
            averaged_weights[k] = torch.zeros_like(v.data)
    for it, idx in enumerate(sampled_client_indices):
        for k, v in client_models[idx].state_dict().items():
            if k in averaged_weights.keys():
                averaged_weights[k] += coefficients[it] * v.data

    for k, v in server_model.state_dict().items():
        if k in averaged_weights.keys():
            v.data.copy_(averaged_weights[k].data.clone())

    for client_idx in np.arange(len(client_models)):
        for key, param in averaged_weights.items():
            if 'prompter' in key:
                client_models[client_idx].state_dict()[key].data.copy_(param)
    return server_model, client_models


def create_all_model(cfg):
    device = torch.device(cfg.SOLVER.DEVICE)

    server_model = Swin(cfg).to(device)

    checkpoint = torch.load(cfg.PRETRAINED_FASTMRI_CKPT, map_location='cpu')
    state_dict = checkpoint['server_model']
    server_model.load_state_dict(state_dict, strict=False)
    for k, v in server_model.head.named_parameters():
        v.requires_grad = False

    models = [copy.deepcopy(server_model) for idx in range(cfg.FL.CLIENTS_NUM)]

    return server_model, models


def make_logger(dirname):
    logger = logging.getLogger('FedMRI_log')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt='%(asctime)s %(filename)s [lineno: %(lineno)d] %(message)s')
    filename = '{}/log.txt'.format(dirname)
    fh = logging.FileHandler(filename=filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt=fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt=fmt)
    logger.addHandler(hdlr=fh)
    logger.addHandler(hdlr=sh)
    return logger


def main(cfg):

    outputdir = os.path.join(cfg.OUTPUTDIR, cfg.FL.MODEL_NAME, cfg.DISTRIBUTION_TYPE)
    experiments_num = max([int(k.split('_')[0]) + 1 for k in os.listdir(outputdir)]) if os.path.exists(outputdir) and not len(os.listdir(outputdir)) == 0 else 0
    outputdir = os.path.join(outputdir, f'{experiments_num:02d}_' + time.strftime('%y-%m-%d_%H-%M') + f'local{cfg.TRAIN.LOCAL_EPOCHS}')

    if outputdir:
        os.makedirs(outputdir, exist_ok=True)
    ckpt_root = Path(outputdir) / 'ckpt'
    ckpt_root.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(os.path.join(outputdir, 'tensorboard'))
    logger = make_logger(outputdir)
    logger.info(logger.handlers[0].baseFilename)
    logger.info('New job assigned {}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')))
    logger.info('\nconfig:\n{}\n'.format(cfg))
    logger.info('=======' * 5 + '\n')


    server_model, models = create_all_model(cfg)
    criterion = Criterion()

    start_epoch = 0
    seed = cfg.SEED + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(cfg.SOLVER.DEVICE)

    n_parameters = sum(p.numel() for p in server_model.parameters() if p.requires_grad)
    logger.info('TOTAL Trainable Params:  {:.2f} M'.format(n_parameters / 1000 / 1000))

    dataloader_train, lens_train = build_different_dataloader(cfg, mode='train')
    if cfg.DISTRIBUTION_TYPE == 'in-distribution':
        dataloader_val, lens_val = build_different_dataloader(cfg, mode='val')
    elif cfg.DISTRIBUTION_TYPE == 'out-of-distribution':
        dataloader_val, lens_val = build_server_dataloader(cfg, mode='val')
    else:
        raise ValueError("cfg.DISTRIBUTION_TYPE should be in ['in-distribution', 'out-of-distribution']")

    logger.info(f'train dataset:{lens_train}')
    logger.info(f'val   dataset:{lens_val}')

    # build optimizer
    trainable_prompt = []
    for idx in range(len(models)):
        m_param = [v for k, v in models[idx].enc.prompter.named_parameters() if v.requires_grad]
        trainable_prompt.append(m_param)

    optimizers = [AdamSVD(trainable_prompt[idx], lr=cfg.SOLVER.LR[idx], weight_decay=cfg.SOLVER.WEIGHT_DECAY, ratio=cfg.SOLVER.RATIO) for idx in range(cfg.FL.CLIENTS_NUM)]

    # milestone = [30, ]
    # lr_schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizers[idx], milestones=milestone, gamma=cfg.SOLVER.LR_GAMMA) for idx in range(cfg.FL.CLIENTS_NUM)]

    cfg.RESUME = ''

    if cfg.RESUME != '':
        checkpoint = torch.load(cfg.RESUME, device)
        server_model.load_state_dict(checkpoint['server_model'], strict=True)
        for idx, client_name in enumerate(cfg.DATASET.CLIENTS):
            models[idx].load_state_dict(checkpoint['server_model'])

    start_time = time.time()
    server_best_status = {'NMSE': 10000000, 'PSNR': 0, 'SSIM': 0, 'bestround': 0}

    for com_round in range(start_epoch, cfg.TRAIN.EPOCHS):
        logger.info('---------------- com_round {:<3d}/{:<3d}----------------'.format(com_round, cfg.TRAIN.EPOCHS))
        sampled_client_indices = np.random.choice(a=range(cfg.FL.CLIENTS_NUM), size=cfg.meta_client_num, replace=False).tolist()
        logger.info(f"sampled clients: {sampled_client_indices}")
        for idx, client_idx in enumerate(sampled_client_indices):
            for _ in range(cfg.TRAIN.LOCAL_EPOCHS):
                train_one_epoch_null_nohead(model=models[client_idx], criterion=criterion, data_loader=dataloader_train[client_idx],
                                            optimizer=optimizers[client_idx], device=device)
            # #lr_schedulers[client_idx].step()

        logger.info(f"[Round: {str(com_round).zfill(4)}] Aggregate updated weights ...!")
        # calculate averaging coefficient of weights
        selected_total_size = sum([lens_train[idx] for idx in sampled_client_indices])
        mixing_coefficients = [lens_train[idx] / selected_total_size for idx in sampled_client_indices]

        # Aggregation
        server_model, models = average_model(server_model, models, sampled_client_indices, mixing_coefficients)

        fea_in = defaultdict(dict)
        for idx, (k, p) in enumerate(server_model.enc.prompter.named_parameters()):
            fea_in[idx] = torch.bmm(p.transpose(1, 2), p)
        for idx in sampled_client_indices:
            optimizers[idx].get_eigens(fea_in=fea_in)
            optimizers[idx].get_transforms()

        # server evaluate
        eval_status = server_evaluate(server_model, criterion, dataloader_val, device)

        logger.info(f'**** Current_round: {com_round:03d}  server PSNR: {eval_status["PSNR"]:.3f}  SSIM: {eval_status["SSIM"]:.3f} '
                    f'NMSE: {eval_status["NMSE"]:.3f} val_loss: {eval_status["loss"]:.3f}')

        writer.add_scalar(tag='server psnr', scalar_value=eval_status["PSNR"], global_step=com_round)
        writer.add_scalar(tag='server ssim', scalar_value=eval_status["SSIM"], global_step=com_round)
        writer.add_scalar(tag='server loss', scalar_value=eval_status["loss"], global_step=com_round)
        if eval_status['PSNR'] > server_best_status['PSNR']:
            server_best_status.update(eval_status)
            server_best_status.update({'bestround': com_round})
            server_best_checkpoint = {
                'server_model': server_model.state_dict(),
                'bestround': com_round,
                'args': cfg,
            }

            if not os.path.exists(ckpt_root):
                ckpt_root = Path(outputdir) / 'ckpt'
                ckpt_root.mkdir(parents=True, exist_ok=True)
            checkpoint_path = os.path.join(ckpt_root, f'checkpoint-epoch_{(com_round):04}.pth')
            torch.save(server_best_checkpoint, checkpoint_path)

        logger.info(f'********* Best_round: {server_best_status["bestround"]}  '
                    f'SERVER PSNR: {server_best_status["PSNR"]:.3f} '
                    f'SSIM: {server_best_status["SSIM"]:.3f} '
                    f'NMSE: {server_best_status["NMSE"]:.3f} ')
        logger.info('*************' * 5 + '\n')

    # log the best score!
    logger.info("Best Results ----------")
    logger.info('The best round for Server  is {}'.format(server_best_status['bestround']))
    logger.info("PSNR: {:.4f}".format(server_best_status['PSNR']))
    logger.info("NMSE: {:.4f}".format(server_best_status['NMSE']))
    logger.info("SSIM: {:.4f}".format(server_best_status['SSIM']))
    logger.info("------------------")

    checkpoint_final_path = os.path.join(ckpt_root, 'best.pth')
    shutil.copy(checkpoint_path, checkpoint_final_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info(logger.handlers[0].baseFilename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="a unit Cross Multi modity transformer")
    parser.add_argument(
        "--config", default="different_dataset", help="choose a experiment to do")
    args = parser.parse_args()

    cfg = build_config(args.config)

    main(cfg)

    print('OK!')
