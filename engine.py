import os
import time
import hashlib
from typing import Iterable
import imageio
import util.misc as utils
import datetime
import numpy as np
import matplotlib.pyplot as plt
from util.metric import nmse, psnr, ssim, AverageMeter
from collections import defaultdict

import torch
import torch.nn.functional as F



def train_one_epoch_null_nohead(model, criterion,data_loader, optimizer, device):
    model.train()
    loss_all = 0
    count=0
    for _, data in enumerate(data_loader):
        count+=1
        image, target, mean, std, fname, slice_num = data  # NOTE

        image = image.unsqueeze(1)  # (8,1,320,320)
        target = target.unsqueeze(1)
        image = image.to(device)
        target = target.to(device)

        outputs = model(image)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss['loss'].backward()

        optimizer.step()

        loss_all += loss['loss'].item()

    loss_avg = loss_all / len(data_loader)
    global_step = count

    return {"loss": loss_avg, "global_step": global_step}


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
            image = image.unsqueeze(1)  # (8,1,320,320)
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

            for k, f in enumerate(fname):
                output_dic[f][slice_num[k].item()] = outputs[k]
                target_dic[f][slice_num[k].item()] = target[k]

        for name in output_dic.keys():
            f_output = torch.stack([v for _, v in output_dic[name].items()])  # (34,320,320)
            f_target = torch.stack([v for _, v in target_dic[name].items()])  # (34,320,320)
            our_nmse = nmse(f_target.cpu().numpy(), f_output.cpu().numpy())
            our_psnr = psnr(f_target.cpu().numpy(), f_output.cpu().numpy())
            our_ssim = ssim(f_target.cpu().numpy(), f_output.cpu().numpy())

            nmse_meter.update(our_nmse, 1)
            psnr_meter.update(our_psnr, 1)
            ssim_meter.update(our_ssim, 1)

    total_time = time.time() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    loss_avg = loss_all / count

    return {'total_time': total_time, 'loss': loss_avg, 'PSNR': psnr_meter.avg, 'SSIM': ssim_meter.avg, 'NMSE': nmse_meter.avg}







