'''Create sampling patterns for Cartesian k-space trajectories.'''

import numpy as np

def cartesian_pe(shape, undersample=.5, reflines=20):
    '''Randomly collect Cartesian phase encodes (lines).

    Parameters
    ----------
    shape : tuple
        Shape of the image to be sampled.
    undersample : float, optional
        Undersampling factor (0 < undersample <= 1).
    reflines : int, optional
        Number of lines in the center to collect regardless.

    Returns
    -------
    mask : array_like
        Boolean mask of sample locations on Cartesian grid.

    Raises
    ------
    AssertionError
        If undersample factor is outside of interval (0, 1].
    '''

    assert 0 < undersample <= 1, (
        'Undersampling factor must be in (0,1]!')

    M, _N = shape[:]
    k = int(undersample*M)
    idx = np.random.permutation(M)[:k]

    mask = np.zeros(shape)*False
    mask[idx, :] = True

    # Make sure we grab center of kspace regardless
    mask[int(M/2-reflines/2):int(M/2+reflines/2), :] = True

    return mask

def cartesian_gaussian(shape, undersample=(.5, .5), reflines=20):
    '''Undersample in Gaussian pattern.

    Parameters
    ----------
    shape : tuple
        Shape of the image to be sampled.
    undersample : tuple, optional
        Undersampling factor in x and y (0 < ux, uy <= 1).
    reflines : int, optional
        Number of lines in the center to collect regardless.

    Returns
    -------
    mask : array_like
        Boolean mask of sample locations on Cartesian grid.

    Raises
    ------
    AssertionError
        If undersample factors are outside of interval (0, 1].
    '''

    assert 0 < undersample[0] <= 1 and 0 < undersample[1] <= 1, \
        'Undersampling factor must be in (0,1]!'

    M, N = shape[:]
    km = int(undersample[0]*M)
    kn = int(undersample[1]*N)

    mask = np.zeros(N*M).astype(bool)
    idx = np.arange(mask.size)
    np.random.shuffle(idx)
    mask[idx[:km*kn]] = True
    mask = mask.reshape(shape)

    # Make sure we grab the reference lines in center of kspace
    mask[int(M/2-reflines/2):int(M/2+reflines/2), :] = True

    return mask


import matplotlib.pyplot as plt
def imshow(img, title=""):
    """ Show image as grayscale.
    imshow(np.linalg.norm(coilImages, axis=0))
    """
    if img.dtype == np.complex64 or img.dtype == np.complex128:
        print('img is complex! Take absolute value.')
        img = np.abs(img)

    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    #plt.show()
    plt.savefig('mask_show.jpg')


if __name__ == "__main__":
    import scipy.io as sio
    Accrate=5
    H=256
    W=256
    undersample=1/Accrate
    a=cartesian_pe((W,H),undersample=undersample, reflines=20)
    sio.savemat('1D-Cartesian_{}X_{}{}.mat'.format(Accrate,H,W),{'mask':np.rot90(a)})
