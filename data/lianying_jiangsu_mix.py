import csv
import os

import logging
import pickle
import random

import scipy.io as sio
from scipy.io import loadmat, savemat
from os import listdir, path
from os.path import splitext
from types import SimpleNamespace

from .transforms import build_transforms
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset



class LianYingdataset(Dataset):
    def __init__(self, data_dir,  transforms,  crop_size, challenge, sample_rate=1, mode='train', pattern='T2',
            totalsize=None, slice_range=(0, 18)):
        
        self.transform = transforms
        self.data_dir = data_dir
        self.img_size = crop_size
        self.examples = []

        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )

        #make an image id's list
        if mode == 'train':
            f = open(path.join(str(data_dir),'lianying_train.txt'),'r')
        elif mode == 'val':
            f = open(path.join(str(data_dir),'lianying_val.txt'),'r')
        elif mode == 'test':
            f = open(path.join(str(data_dir),'lianying_test.txt'),'r')
        else: 
            raise ValueError("No mode like this, please choose one in ['train', 'val', 'test'].")
        
        file_names = f.readlines()

        metadata = {
            'acquisition': pattern,
            'encoding_size': (640, 556, 1),
            'max': 0,
            'norm': 0,
            'padding_left': 0,
            'padding_right': 0,
            'patient_id':'0',
            'recon_size': (320, 320, 1),
        }

        if not pattern == 'T1+T2':

            if pattern == 'T1':
                idx = 0
            elif pattern == 'T2':
                idx = 1

            for file_name in file_names:
                splits = file_name.split()  # 分离空格
                for slice_id in range(slice_range[0], slice_range[1]+1):  # 0:19==20
                    self.examples.append((os.path.join(self.data_dir, splits[idx]), slice_id, metadata))
                    if len(self.examples) > totalsize:
                        break
        else:
            for file_name in file_names:
                splits = file_name.split()
                for slice_id in range(slice_range[0], slice_range[1]+1):  # 0:18==19
                    self.examples.append((os.path.join(self.data_dir, splits[0]), slice_id, metadata))
                    self.examples.append((os.path.join(self.data_dir, splits[1]), slice_id, metadata))
                    if len(self.examples) > totalsize:
                        break

        self.examples = self.examples[:totalsize]
        if mode == 'train':
            logging.info(f'Creating training dataset with {len(self.examples)} examples')
        elif mode == 'val':
            logging.info(f'Creating validation dataset with {len(self.examples)} examples')
        elif mode == 'test':
            logging.info(f'Creating test dataset with {len(self.examples)} examples')
        
        if sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, slice_id):
        fname_nii, slice_idx, metadata =self.examples[slice_id]
        slice_path = self.niipath2matpath(fname_nii, slice_idx)
        image = loadmat(slice_path)['img']

        mask = None
        attrs = metadata

        image = np.rot90(image)
        
        kspace = self.fft2(image).astype(np.complex64)
        target = image.astype(np.float32)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname_nii, slice_idx)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname_nii, slice_idx)

        return sample

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

    def niipath2matpath(self, T1,slice_id):
        filedir,filename = path.split(T1)
        filedir,_ = path.split(filedir)
        mat_dir = path.join(filedir,'mat_320')
        basename, ext = path.splitext(filename)
        file_name = '%s-%03d.mat'%(basename,slice_id)
        T1_file_path = path.join(mat_dir,file_name)
        return T1_file_path


class JiangSudataset(Dataset):
    def __init__(self,  data_dir, transforms, crop_size, challenge, sample_rate=1, mode='train',  pattern='T2',
                 totalsize=None, slice_range=(0, 19)):
        
        self.transform = transforms
        self.data_dir = data_dir
        self.img_size = crop_size
        self.examples = []

        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )

        #make an image id's list
        if mode == 'train':
            f = open(path.join(str(data_dir), 'jiangsu_train.txt'),'r')
        elif mode == 'val':
            f = open(path.join(str(data_dir), 'jiangsu_val.txt'),'r')
        elif mode == 'test':
            f = open(path.join(str(data_dir), 'jiangsu_test.txt'),'r')
        else: 
            raise ValueError("No mode like this, please choose one in ['train', 'val', 'test'].")
        
        file_names = f.readlines()

        metadata = {
            'acquisition': pattern,
            'encoding_size': (640, 556, 1),
            'max': 0,
            'norm': 0,
            'padding_left': 0,
            'padding_right': 0,
            'patient_id':'0',
            'recon_size': (320, 320, 1),
        }

        if not pattern == 'T1+T2':

            if pattern == 'T1':
                idx = 0
            elif pattern == 'T2':
                idx = 1

            for file_name in file_names:
                splits = file_name.split()
                for slice_id in range(slice_range[0], slice_range[1]+1):  # 0:19==20
                    self.examples.append((os.path.join(self.data_dir, splits[idx]), slice_id, metadata))
                    if len(self.examples) > totalsize:
                        break
        else:
            for file_name in file_names:
                splits = file_name.split()
                for slice_id in range(slice_range[0], slice_range[1]+1):  # 0:19==20
                    self.examples.append((os.path.join(self.data_dir, splits[0]), slice_id, metadata))
                    self.examples.append((os.path.join(self.data_dir, splits[1]), slice_id, metadata))
                    if len(self.examples) > totalsize[mode]:
                        break

        self.examples = self.examples[:totalsize]
        if sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, slice_id):

        fname_nii, slice_idx, metadata =self.examples[slice_id]

        slice_path = self.niipath2matpath(fname_nii, slice_idx)
        image = loadmat(slice_path)['img']

        mask = None
        attrs = metadata

        image = np.rot90(image)
        
        kspace = self.fft2(image).astype(np.complex64)
        target = image.astype(np.float32)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname_nii, slice_idx)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname_nii, slice_idx)

        return sample

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

    def niipath2matpath(self, T1,slice_id):
        filedir, filename = path.split(T1)
        filedir,_ = path.split(filedir)
        mat_dir = path.join(filedir, 'mat320')
        basename, ext = path.splitext(filename)
        # base_name = basename[:-1]
        file_name = '%s-%03d.mat'%(basename, slice_id)
        T1_file_path = path.join(mat_dir, file_name)
        return T1_file_path



def create_datasets(args, data_root=None, mode='train', sample_rate=1, client_name='fastMRI', pattern='pd',
                    crop_size=None, totalsize=None, slice_range=None):
    assert mode in ['train', 'val', 'test'], 'unknown mode'
    transforms = build_transforms(args, mode, client_name=client_name)

    if client_name == 'JiangSu':
        return JiangSudataset(data_root, transforms=transforms, crop_size=crop_size, challenge=args.DATASET.CHALLENGE,
                            sample_rate=sample_rate, mode=mode, pattern=pattern,
                            totalsize=totalsize, slice_range=slice_range)

    elif client_name == 'lianying':
        return LianYingdataset(data_root, transforms=transforms, crop_size=crop_size, challenge=args.DATASET.CHALLENGE,
                               sample_rate=sample_rate, mode=mode, pattern=pattern,
                               totalsize=totalsize, slice_range=slice_range)


