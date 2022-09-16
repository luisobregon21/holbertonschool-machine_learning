#!/usr/bin/env python3
''' custom padding convolution on grayscale images '''

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    '''
    performs a custom padding convolution on grayscale images
    :images: a numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
        m: is the number of images
        h: is the height in pixels of the images
        w: is the width in pixels of the images
    :kernel: is a numpy.ndarray with shape
    (kh, kw) containing the kernel for the convolution
        kh: is the height of the kernel
        kw: is the width of the kernel
    :padding: is a tuple of (ph, pw)
        ph: is the padding for the height of the image
        pw: is the padding for the width of the image
        the image should be padded with zeros
    '''
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding[0], padding[1]
    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1
    dim = (m, oh, ow)
    out = np.zeros(dim)
    padded = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                    mode='constant', constant_values=0)
    for i in range(dim[1]):
        for j in range(dim[2]):
            x = i + kh
            y = j + kw
            M = padded[:, i:x, j:y]
            out[:, i, j] = np.tensordot(M, kernel)
    return out
