from os.path import splitext
import os
from os import listdir, path
from torch.utils.data import Dataset
import random
from scipy.io import loadmat, savemat
import scipy.io as sio
import torch
import cv2
import numpy as np


class IXIdataset(Dataset):
    def __init__(self, data_root, list_file, maskdir, mode='train', pattern='T2', sample_rate=1.0, totalsize=None,
                 crop_size=(224,224), ):
        self.list_file = list_file
        self.mode = mode
        self.maskdir = maskdir
        self.examples = []
        self.img_size = crop_size

        file_names = open(list_file).readlines()

        if not pattern == 'T1+T2':
            if pattern == 'T1':
                idx = 0
            elif pattern == 'T2':
                idx = 1
            for file_name in file_names:
                splits = file_name.split()
                for slice_id in [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 119]:
                    name = os.path.basename(splits[idx]).split('.')[0]
                    path = os.path.dirname(splits[idx]) + '_mat'
                    self.examples.append((os.path.join(data_root, path), name, slice_id))
                    if len(self.examples) > totalsize:
                        break
        else:
            for idx, file_name in enumerate(file_names):
                splits = file_name.split()
                for slice_id in [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 119]:
                    name = os.path.basename(splits[0]).split('.')[0]
                    path = os.path.dirname(splits[0]) + '_mat'
                    self.examples.append((os.path.join(data_root, path), name, slice_id))
                    name = os.path.basename(splits[1]).split('.')[0]
                    path = os.path.dirname(splits[1]) + '_mat'
                    self.examples.append((os.path.join(data_root, path), name, slice_id))
                    if len(self.examples) > totalsize:
                        break

        self.mask = loadmat(maskdir)['mask']
        self.examples = self.examples[:totalsize]

        if sample_rate < 1:
            if mode == 'train':
                random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fpath, fname, slice_num = self.examples[i]
        file_name = '%s-%03d.mat' % (fname, slice_num)
        file_name = os.path.join(fpath, file_name)
        img = loadmat(file_name)
        img = img['img']

        if 'T2' in fname:
            img_height, img_width  = img.shape
            img_matRotate = cv2.getRotationMatrix2D((img_height * 0.5, img_width * 0.5), 90, 1)
            img = cv2.warpAffine(img, img_matRotate, (img_height, img_width))

        kspace = self.fft2(img).astype(np.complex64)

        cropped_kspace = self.pad_toshape(kspace, pad_shape=self.img_size)
        # target image:
        target = self.ifft2(cropped_kspace).astype(np.complex64)
        target = np.absolute(target)
        masked_cropped_kspace = cropped_kspace * self.mask
        subsample = self.ifft2(masked_cropped_kspace).astype(np.complex64)
        subsample = np.absolute(subsample)

        subsample, target = torch.from_numpy(subsample).float(), torch.from_numpy(target).float()

        subsample, mean, std = self.normalize_instance(subsample, eps=1e-11)
        target = (target - mean) / std
        subsample = subsample.clamp(-6, 6)
        target = target.clamp(-6, 6)

        return subsample, target, mean, std, fname, slice_num

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img), norm=None))

    def ifft2(self, kspace):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace), norm=None))

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

    def normalize_instance(self, data, eps=0.0):
        mean = data.mean()
        std = data.std()
        return (data - mean) / (std + eps), mean, std


def build_ixi_dataset(data_root, list_file, mask_path, mode, type=None, pattern='T2', sample_rate=1.0, totalsize=None,
                      crop_size=(320, 320)):
    assert type in ['HH', 'Guys', 'IOP'], "IXI type should choosen from ['HH','Guys','IOP'] "

    return IXIdataset(data_root, list_file=list_file, maskdir=mask_path, mode=mode, pattern=pattern, sample_rate=sample_rate,
                      totalsize=totalsize, crop_size=crop_size)

