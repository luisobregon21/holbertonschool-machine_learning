#!/usr/bin/env python3
''' convolution on images using multiple kernels '''

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    '''
    performs a convolution on images using multiple kernels
    :images: a numpy.ndarray with shape (m, h, w) containing
    multiple images with channels
        m: is the number of images
        h: is the height in pixels of the images
        w: is the width in pixels of the images
        c: is the number of channels in the image
    :kernel: is a numpy.ndarray with shape
    (kh, kw, c) containing the kernel for the convolution
        kh: is the height of the kernel
        kw: is the width of the kernel
        nc: is the number of kernels
    :padding: is a tuple of (ph, pw)
        ph: is the padding for the height of the image
        pw: is the padding for the width of the image
        the image should be padded with zeros
    :stride: is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    '''
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride

    if padding == 'valid':
        ph = pw = 0
    elif padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)
    else:
        ph, pw = padding

    oh = int(((h + 2 * ph - kh) / sh) + 1)
    ow = int(((w + 2 * pw - kw) / sw) + 1)

    # convolutional dimension
    conv_image = np.zeros((m, oh, ow, nc))
    # padded images
    padded_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw),
                                              (0, 0)), mode='constant')
    for y in range(oh):
        for x in range(ow):
            image_slice = padded_images[:,
                                        (y*sh):(y*sh)+kh,
                                        (x*sw):(x*sw)+kw, :]
            for z in range(nc):
                # change values of zero's array
                conv_image[:, y, x, z] = np.tensordot(
                    image_slice, kernels[:, :, :, z],
                    axes=c
                    )
    return conv_image
