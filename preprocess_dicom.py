import pydicom
import os
import json
from scipy.io import savemat
import numpy as np
from tqdm import tqdm
import glob
from collections import defaultdict
import scipy.io as sio
import nibabel as nib


root = os.path.join(os.path.expanduser('~'), r'data/AAA-Download-Datasets/')


def make_mat():
    sequence = 'T1'
    src_root = os.path.join(root, f'Fed/Data/IXI-NIFTI/{sequence}')
    dst_root = os.path.join(root, f'Fed/Data/IXI-NIFTI/{sequence}_mat')
    if not os.path.exists(dst_root):
        os.makedirs(dst_root, exist_ok=True)
    names = sorted(os.listdir(src_root))
    for name in tqdm(names):
        niipath = os.path.join(src_root, name)
        data = nib.load(niipath)
        print(f'data orientation {nib.aff2axcodes(data.affine)}')
        img = data.get_fdata().astype(np.float32)
        for slice in range(img.shape[1]):
            matpath = os.path.join(dst_root, name.split('.')[0]+f'-{slice:03d}.mat')
            sio.savemat(file_name=matpath, mdict={'img' : img[:,  slice]})


def ixi_txt():
    dst_sub = 'Fed/Data/IXI-NIFTI'
    path = os.path.join(root, dst_sub)
    for client in ['HH', 'Guys', 'IOP']:
        lines1 = glob.glob(os.path.join(path, f'T1/*{client}*'))
        lines2 = glob.glob(os.path.join(path, f'T2/*{client}*'))
        names1 = [os.path.basename(l1).split('-T')[0] for l1 in lines1]
        names2 = [os.path.basename(l1).split('-T')[0] for l1 in lines2]
        names = [n for n in names1 if n in names2]

        print(f'there are {len(names)}  {client} intersect files')

        split = int(len(names) * 0.7)
        p1 = os.path.join(dst_sub, 'T1')
        p2 = os.path.join(dst_sub, 'T2')
        with open(os.path.join(path, f'{client}_train.txt'), 'w') as f:
            for name in names[:split]:
                print(os.path.join(p1, name+'-T1.nii.gz') + '    ' + os.path.join(p2, name+'-T2.nii.gz'), file=f)

        with open(os.path.join(path, f'{client}_val.txt'), 'w') as f:
            for name in names[split:]:
                print(os.path.join(p1, name+'-T1.nii.gz') + '    ' + os.path.join(p2, name+'-T2.nii.gz'), file=f)


if __name__ == '__main__':

    # make_mat()
    # ixi_txt()


    print('ok')
