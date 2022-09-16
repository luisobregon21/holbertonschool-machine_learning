#!/usr/bin/env python3
''' same convolution on grayscale images '''

import numpy as np


def convolve_grayscale_same(images, kernel):
    '''
    performs a same convolution on grayscale images
    :images: a numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
        m: is the number of images
        h: is the height in pixels of the images
        w: is the width in pixels of the images
    :kernel: is a numpy.ndarray with shape
    (kh, kw) containing the kernel for the convolution
        kh: is the height of the kernel
        kw: is the width of the kernel
    '''
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = int(np.ceil((kh - 1) / 2))
    pw = int(np.ceil((kw - 1) / 2))

    # convolutional dimension
    conv_image = np.zeros(images.shape)

    # padded images
    padded_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')
    for y in range(h):
        for x in range(w):
            image_slice = padded_images[:, y:y+kh, x:x+kw]
            # change values of zero's array
            conv_image[:, y, x] = np.tensordot(image_slice, kernel)
    return conv_image
