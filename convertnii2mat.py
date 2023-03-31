import h5py
import os
import scipy.io as sio
from os.path import splitext
from tqdm import tqdm
import argparse

import nibabel as nib


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

def convert(niipath, matpath):
    niifiles = os.listdir(niipath)
    os.makedirs(matpath, exist_ok=True)

    for nif in tqdm(niifiles):
        nifile = os.path.join(niipath, nif)
        fname = nif.split('.')[0]
        f = nib_load(nifile)
        slices = f.shape[2]
        for slice in range(slices):
            img = f[..., slice]
            matfile = os.path.join(matpath, fname + '-{:03d}.mat'.format(slice))
            sio.savemat(matfile, {'img':img})

def main(args):

    os.makedirs(args.dst_root, exist_ok=True)

    convert(args.src_root, args.dst_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert h5 to mat")
    parser.add_argument(
        "--src_root", default="", help="choose a experiment to do")
    parser.add_argument(
        "--dst_root", default="", help="choose a experiment to do")

    args = parser.parse_args()
    main(args)