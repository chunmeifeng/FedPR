
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def nmse(gt, pred):
    """Compute Normalized Mean Squared Error (NMSE)"""
    result = 0

    for idx in range(gt.shape[0]):
        result+=np.linalg.norm(gt[idx] - pred[idx]) ** 2 / np.linalg.norm(gt[idx]) ** 2

    result = result/gt.shape[0]

    return result


def psnr(gt, pred):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""

    maxval = gt.max()

    result = 0

    for idx in range(gt.shape[0]):
        result += peak_signal_noise_ratio(gt[idx], pred[idx], data_range = maxval)

    result = result / gt.shape[0]

    return result


def ssim(gt, pred, maxval=None):
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    ssim = ssim / gt.shape[0]

    return ssim


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count