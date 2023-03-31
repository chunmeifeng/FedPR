import os
import random
from .transforms import normalize, normalize_instance
from torch.utils.data import Dataset
from scipy.io import loadmat
import pickle
from matplotlib import pyplot as plt
from .math import *


class FedTS(Dataset):
    def __init__(self, data_root, list_file, mask_path, mode='train', sample_rate=1., crop_size=(320, 320), pattern='T2',
                 totalsize=None, sample_delta=4):
        self.mask = loadmat(mask_path)['mask']
        self.crop_size = crop_size

        paths = []
        with open(list_file) as f:
            for line in f:
                line = os.path.join(data_root, line.strip())
                name = line.split('/')[-1]
                for slice in range(44, 124, sample_delta):
                    if not pattern=='T1+T2':
                        if pattern=='T1':
                            path1 = os.path.join(line, name + '_t1' + '-%03d' % (slice) + '.mat')
                            paths.append((path1, slice, name+'_t1'+'-%03d'%(slice)))
                        if pattern=='T2':
                            path2 = os.path.join(line, name + '_t2' + '-%03d' % (slice) + '.mat')
                            paths.append((path2, slice, name+'_t2'+'-%03d'%(slice)))
                    else:
                        path1 = os.path.join(line, name + '_t1' + '-%03d' % (slice) + '.mat')
                        path2 = os.path.join(line, name + '_t2' + '-%03d' % (slice) + '.mat')
                        paths.append((path1, slice, name + '_t1' + '-%03d' % (slice)))
                        paths.append((path2, slice, name + '_t2' + '-%03d' % (slice)))
                if len(paths) > totalsize:
                    break

        if sample_rate < 1:
            if mode == 'train':
                random.shuffle(paths)
            num_examples = round(len(paths) * sample_rate)
            self.examples = paths[0:num_examples]
        else:
            self.examples = paths
        self.examples = self.examples[:totalsize]

    def __getitem__(self, item):
        path, slice, fname = self.examples[item]
        img = loadmat(path)
        img = img['img'].transpose(1, 0)
        kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
        kspace = self.pad_toshape(kspace, pad_shape=self.crop_size)

        maskedkspace = kspace * self.mask
        subsample = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(maskedkspace)))
        subsample = abs(subsample)
        target = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
        target = abs(target)

        subsample = torch.from_numpy(subsample).float()
        target = torch.from_numpy(target).float()

        subsample, mean, std = normalize_instance(subsample, eps=1e-11)
        target = normalize(target, mean, std, eps=1e-11)
        subsample = subsample.clamp(-6, 6)
        target = target.clamp(-6, 6)

        return subsample, target, mean, std, fname, slice

    def __len__(self):
        return len(self.examples)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

    def pad_toshape(self, data, pad_shape):
        assert 0 < data.shape[-2] <= pad_shape[0], 'Error: pad_shape: {}, data.shape: {}'.format(pad_shape, data.shape)  # 556...556
        assert 0 < data.shape[-1] <= pad_shape[1]   # 640...640
        k = np.zeros(shape=pad_shape, dtype=np.complex64)
        h_from = (pad_shape[0] - data.shape[-2]) // 2
        w_from = (pad_shape[1] - data.shape[-1]) // 2
        h_to = h_from + data.shape[-1]
        w_to = w_from + data.shape[-2]
        k[...,h_from:h_to, w_from:w_to] = data
        return k


def build_fed_dataset(data_root, list_file, mask_path, mode='train',  sample_rate=1.0, crop_size=320, pattern='T2',
                      totalsize=None, sample_delta=4):
    if mode in ['train', 'val']:
        return FedTS(data_root, list_file, mask_path, mode, sample_rate, crop_size=crop_size, pattern=pattern,
                     totalsize=totalsize, sample_delta=sample_delta)
    else:
        raise ValueError('mode in fedts2022 dataset setup process error')


