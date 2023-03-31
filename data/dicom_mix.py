import pydicom
import os
import torch
import numpy as np
import random
from scipy.io import loadmat
from .transforms import center_crop,normalize_instance, normalize
from torch.utils.data import Dataset
from data.transforms import to_tensor
from .math import *


class FastMRIDicom(Dataset):
    def __init__(self, list_file, mask, crop_size=(320, 320), mode='train', sample_rate=1, totalsize=None):
        data_root = os.path.join(os.path.expanduser('~'), 'Apollo2022/AAA-Download-Datasets')
        paths = []
        with open(list_file) as f:
                for idx, line in enumerate(f):
                    path = os.path.join(data_root, line.strip())
                    name = line.strip().split('/')[-1]
                    slices = sorted([s.split('.')[0] for s in os.listdir(path)])
                    for idx,slice in enumerate(slices[:16]):
                        paths.append((path, slice, name))
                    if len(paths) > totalsize:
                        break

        paths = paths[: totalsize]
        self.crop_size = crop_size
        self.mask = loadmat(mask)['mask']

        if sample_rate < 1:
            if mode == 'train':
                random.shuffle(paths)
            num_examples = round(len(paths) * sample_rate)
            self.examples = paths[0:num_examples]
        else:
            self.examples = paths

    def __getitem__(self, item):
        path, slice, fname = self.examples[item]
        img_path = os.path.join(path, str(slice)+'.mat')

        img = loadmat(img_path)['img']

        kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
        kspace = center_crop(kspace, self.crop_size)
        maskedkspace = kspace * self.mask

        maskedkspace = to_tensor(maskedkspace)

        subsample = complex_abs(ifft2c(maskedkspace))

        kspace = to_tensor(kspace)
        target = complex_abs(ifft2c(kspace))

        subsample, mean, std = normalize_instance(subsample, eps=1e-11)
        target = normalize(target, mean, std, eps=1e-11)

        subsample = subsample.float()
        target = target.float()

        return subsample, target, mean, std, fname, slice

    def __len__(self):
        return len(self.examples)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


def build_fastmri_dataset(args, mode='train'):
    mask = os.path.join(args.FASTMRI_MASK_DIR, args.FASTMRI_MASK_FILE)
    if mode == 'train':
        data_list = os.path.join(args.FASTMRIROOT, 'train.txt')
    elif mode == 'val':
        data_list = os.path.join(args.FASTMRIROOT, 'val.txt')
    else:
        raise ValueError('mode error')

    return FastMRIDicom(data_list, mask, mode=mode, sample_rate=args.DATASET.SAMPLE_RATE[0],
                        totalsize=0)

