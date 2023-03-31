import os
import time
import datetime
import random
import numpy as np
import argparse
import h5py
import torch
import logging
import scipy.io as sio
from collections import defaultdict
from tqdm import tqdm
import time
from config import build_config
from data import build_different_dataloader, build_server_dataloader
from models.vit_models import Swin
from models.loss import Criterion
from engine import server_evaluate
from util.metric import nmse, psnr, ssim, AverageMeter

def save_reconstructions(reconstructions, out_dir):
    """
    Save reconstruction images.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input
            filenames to corresponding reconstructions (of shape num_slices x
            height x width).
        out_dir (pathlib.Path): Path to the output directory where the
            reconstructions should be saved.
    """
    os.makedirs(str(out_dir), exist_ok=True)
    print(out_dir, len(list(reconstructions.keys())))
    idx = min(len(list(reconstructions.keys())), 10)
    for fname in list(reconstructions.keys())[-idx:]:
        f_output = torch.stack([v for _, v in reconstructions[fname].items()])

        basename = fname.split('/')[-1]
        with h5py.File(str(out_dir) + '/' + str(basename) + '.hdf5', "w") as f:
            print(fname)
            f.create_dataset("reconstruction", data=f_output.cpu())


def create_all_model(args):
    device = torch.device('cpu')
    server_model = Swin(args).to(device)
    return server_model

@torch.no_grad()
def server_evaluate(model, criterion, data_loaders, device):
    model.eval()
    criterion.eval()
    criterion.to(device)

    nmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    output_dic = defaultdict(dict)
    target_dic = defaultdict(dict)

    start_time = time.time()

    loss_all = 0
    count = 0
    for idx, data_loader in enumerate(data_loaders):
        for i, data in enumerate(data_loader):
            count += 1
            image, target, mean, std, fname, slice_num = data
            image = image.unsqueeze(1)
            image = image.to(device)
            target = target.to(device)

            mean = mean.unsqueeze(1).unsqueeze(2)
            std = std.unsqueeze(1).unsqueeze(2)
            mean = mean.to(device)
            std = std.to(device)

            outputs = model(image)
            outputs = outputs.squeeze(1)
            outputs = outputs * std + mean
            target = target * std + mean

            loss = criterion(outputs, target)
            loss_all += loss['loss'].item()

            for i, f in enumerate(fname):
                output_dic[f][slice_num[i].item()] = outputs[i]
                target_dic[f][slice_num[i].item()] = target[i]

        for name in output_dic.keys():
            f_output = torch.stack([v for _, v in output_dic[name].items()])
            f_target = torch.stack([v for _, v in target_dic[name].items()])

            our_nmse = nmse(f_target.cpu().numpy(), f_output.cpu().numpy())
            our_psnr = psnr(f_target.cpu().numpy(), f_output.cpu().numpy())
            our_ssim = ssim(f_target.cpu().numpy(), f_output.cpu().numpy())
            nmse_meter.update(our_nmse, 1)
            psnr_meter.update(our_psnr, 1)
            ssim_meter.update(our_ssim, 1)

    total_time = time.time() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    loss_avg = loss_all / count

    return {'total_time': total_time, 'loss': loss_avg, 'PSNR': psnr_meter.avg,
            'SSIM': ssim_meter.avg, 'NMSE': nmse_meter.avg}


def main_fed(args):
    device = torch.device('cuda:0')
    cfg.DISTRIBUTION_TYPE = 'in-distribution' # 'out-of-distribution'  # 'in-distribution'  #

    outputdir = os.path.dirname(os.path.dirname(cfg.MODEL.WEIGHT_PATH))
    outputdir = os.path.join(outputdir, 'evaluation')
    os.makedirs(outputdir, exist_ok=True)
    dirs = [d for d in os.listdir(outputdir) if os.path.isdir(os.path.join(outputdir, d))]
    experiments_num = max([int(k.split('_')[0]) + 1 for k in dirs]) if os.path.exists(outputdir) and not len(dirs) == 0 else 0
    outputdir = os.path.join(outputdir, f'{experiments_num:02d}_' + time.strftime('%y-%m-%d_%H-%M'))
    if outputdir:
        os.makedirs(outputdir, exist_ok=True)

    server_model = create_all_model(cfg)

    seed = args.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if cfg.DISTRIBUTION_TYPE == 'in-distribution':
        dataloader_val, lens_val = build_different_dataloader(cfg, mode='val')
    elif cfg.DISTRIBUTION_TYPE == 'out-of-distribution':
        dataloader_val, lens_val = build_server_dataloader(cfg, mode='val')
    else:
        raise ValueError("cfg.DISTRIBUTION_TYPE should be in ['in-distribution', 'out-of-distribution']")

    checkpoint = torch.load(cfg.MODEL.WEIGHT_PATH, map_location='cpu')
    server_model.load_state_dict(checkpoint['server_model'])
    server_model.to(device)

    eval_status = server_evaluate(server_model, Criterion(), dataloader_val, device)
    with open(os.path.join(os.path.dirname(outputdir), 'testlog.txt'), 'a') as f:
        print(f'outputdir: {outputdir} \n', file=f)
        print('{} Evaluate time {} PSNR: {:.3f} SSIM: {:.4f} NMSE: {:.4f} \n\n'.format(time.strftime('%Y-%m-%d %H:%M'),
            eval_status['total_time'], eval_status['PSNR'], eval_status['SSIM'], eval_status['NMSE']), file=f)
        print('{} Evaluate time {} PSNR: {:.3f} SSIM: {:.4f} NMSE: {:.4f} \n\n'.format(time.strftime('%Y-%m-%d %H:%M'),
            eval_status['total_time'], eval_status['PSNR'], eval_status['SSIM'], eval_status['NMSE']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="a swin prompter transformer")
    parser.add_argument("--config", default="different_dataset", help="choose a experiment to do")
    args = parser.parse_args()
    cfg = build_config(args.config)

    main_fed(cfg)

    print('OK!')